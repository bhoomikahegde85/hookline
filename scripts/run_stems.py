#!/usr/bin/env python3
"""Music vs voice levels for the opening seconds of a video, using Demucs (htdemucs) on this machine. Nothing is uploaded.

    .venv-demucs/bin/python scripts/run_stems.py --input VIDEO.mp4 --artifacts artifacts/e2e [--seconds 10]

Writes <artifacts>/features/<video_id>/stems.json. Needs the separate .venv-demucs (uv venv .venv-demucs --python 3.11;
uv pip install --python .venv-demucs/bin/python demucs soundfile). The model weights download on first use (about 80 MB).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.stems import STEMS_FILE, summarise  # noqa: E402


def separate(video: Path, seconds: int) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-t", str(seconds), "-vn", "-ac", "2", "-ar", "44100", str(wav)],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not wav.is_file():
            return {"status": "unavailable", "reason": "no audio stream could be read from this video"}
        x, sr = sf.read(str(wav), dtype="float32")
    if x.size == 0:
        return {"status": "unavailable", "reason": "the audio is empty"}
    x = torch.from_numpy(np.ascontiguousarray(x.T if x.ndim == 2 else np.stack([x, x])))
    model = get_model("htdemucs")
    model.eval()
    with torch.no_grad():
        y = apply_model(model, x[None], device="cpu", split=True, progress=False)[0].numpy()
    return summarise({name: y[i] for i, name in enumerate(model.sources)}, sr, window_s=5, max_s=seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--seconds", type=int, default=10, help="opening seconds to separate (default 10)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    video = args.input.expanduser().resolve()
    out_dir = args.artifacts / "features" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / STEMS_FILE
    if out.is_file() and not args.force:
        print(f"{video.stem}: stems.json exists (cached)")
        return 0
    res = separate(video, args.seconds) | {"video_id": video.stem, "model": "htdemucs"}
    out.write_text(json.dumps(res, indent=1))
    print(f"{video.stem}: {res['status']}" + (f" music-voice {res.get('music_minus_voice_db_0_5')} dB" if res["status"] == "ok" else f" ({res.get('reason')})"))
    return 0 if res["status"] in ("ok", "unavailable") else 1


if __name__ == "__main__":
    sys.exit(main())
