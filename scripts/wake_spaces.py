#!/usr/bin/env python3
"""Wake the TTCC and TRIBE Hugging Face Spaces if they are paused or asleep, and optionally wait until they answer /health.

    .venv-models/bin/python scripts/wake_spaces.py [--which ttcc tribe] [--no-wait] [--timeout 900]

Uses the cached Hugging Face login (a token that may restart your Spaces) for the restart, and TTCC_ENDPOINT_URL / TRIBE_ENDPOINT_URL plus
HF_TOKEN from the environment or .env for the health check. Nothing is uploaded. Prints one line per event; exit 0 when every requested Space is ready
(or, with --no-wait, once the wake-up request was sent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SPACES = {"ttcc": ("TTCC_SPACE_ID", "wlddhq/hooklinewldd-ttcc", "TTCC_ENDPOINT_URL"), "tribe": ("TRIBE_SPACE_ID", "wlddhq/hooklinewldd", "TRIBE_ENDPOINT_URL")}
ASLEEP = {"PAUSED", "SLEEPING", "STOPPED"}


def load_env() -> None:
    f = PROJECT_ROOT / ".env"
    if f.is_file():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def healthy(url: str, token: str | None, name: str) -> bool:
    req = urllib.request.Request(url.rstrip("/") + "/health", headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status != 200:
                return False
            body = json.loads(r.read().decode() or "{}")
            return body.get("checkpoint_ready", True) is not False if name == "ttcc" else True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", nargs="+", choices=list(SPACES), default=list(SPACES))
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    load_env()
    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except ImportError:
        print("huggingface_hub is not installed in this environment; run this with .venv-models/bin/python")
        return 2
    ok = True
    for name in args.which:
        env_id, default_id, env_url = SPACES[name]
        space = os.environ.get(env_id, default_id)
        try:
            stage = api.get_space_runtime(space).stage
        except Exception as exc:  # noqa: BLE001 - report, do not crash the pipeline
            print(f"{name}: could not read the Space state ({type(exc).__name__}); continuing")
            continue
        if stage in ASLEEP:
            try:
                api.restart_space(space)
                print(f"{name}: was {stage}; wake-up requested")
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: could not restart ({type(exc).__name__}: {str(exc)[:120]})")
                ok = False
                continue
        else:
            print(f"{name}: {stage}")
        if args.no_wait:
            continue
        url, token = os.environ.get(env_url), os.environ.get("HF_TOKEN")
        t0 = time.time()
        while url and not healthy(url, token, name):
            if time.time() - t0 > args.timeout:
                print(f"{name}: not ready after {args.timeout:.0f} s")
                ok = False
                break
            time.sleep(10)
        else:
            print(f"{name}: ready after {time.time() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
