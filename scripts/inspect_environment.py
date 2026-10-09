#!/usr/bin/env python3
"""Report the local environment for Hookline. Uses only the standard library.

    python scripts/inspect_environment.py [--offline] [--skip-torch]

Writes reports/environment_report.md and reports/environment_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 10)
HF_DATASET_API = "https://huggingface.co/api/datasets/tcm03/SnapUGC"

# distribution name -> why it matters
DEPENDENCIES = {
    "pytest": "tests (this phase)",
    "numpy": "Task 2: arrays / features",
    "opencv-python-headless": "Task 2: frame sampling, scene cuts",
    "opencv-python": "alternative OpenCV build",
    "pandas": "optional: tabular analysis",
    "huggingface_hub": "optional: HF downloads",
    "torch": "later: neural models",
    "torchvision": "later: vision models",
    "transformers": "later: text/vision encoders",
    "faster-whisper": "later: speech transcription",
    "openai-whisper": "later: alternative ASR",
    "easyocr": "later: on-screen text OCR",
    "librosa": "later: audio features",
    "soundfile": "later: audio I/O",
    "psutil": "optional: memory reporting",
}


def run(cmd: list[str], timeout: int = 10) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"machine": platform.machine(), "logical_cores": os.cpu_count()}
    name = None
    if sys.platform == "darwin":
        name = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    info["name"] = name or platform.processor() or None
    return info


GIB = 1024**3


def memory_info() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return {"total_gb": round(vm.total / GIB, 1), "available_gb": round(vm.available / GIB, 1), "via": "psutil"}
    except ImportError:
        pass
    if sys.platform == "darwin":
        total = run(["sysctl", "-n", "hw.memsize"])
        vmstat = run(["vm_stat"])
        avail = None
        if vmstat:
            page = int(re.search(r"page size of (\d+)", vmstat).group(1)) if "page size" in vmstat else 4096
            pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in re.findall(r"^(Pages [^:]+):\s+(\d+\.?)", vmstat, re.M)}
            free = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative"))
            avail = round(free * page / GIB, 1)
        return {"total_gb": round(int(total) / GIB, 1) if total else None, "available_gb": avail, "via": "sysctl/vm_stat (approx.)"}
    if sys.platform.startswith("linux"):
        try:
            mem = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())}
            return {"total_gb": round(mem["MemTotal"] * 1024 / GIB, 1), "available_gb": round(mem.get("MemAvailable", 0) * 1024 / GIB, 1), "via": "/proc/meminfo"}
        except (OSError, KeyError, ValueError):
            pass
    return {"total_gb": None, "available_gb": None, "via": "unavailable (pip install psutil)"}


def gpu_info(skip_torch: bool) -> dict[str, Any]:
    info: dict[str, Any] = {"nvidia_smi": None, "torch": None}
    smi = shutil.which("nvidia-smi")
    if smi:
        out = run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        info["nvidia_smi"] = out or "nvidia-smi present but query failed"
    if skip_torch:
        info["torch"] = "skipped (--skip-torch)"
        return info
    try:
        import torch  # type: ignore
    except ImportError:
        info["torch"] = "not installed"
        return info
    except Exception as exc:  # broken installs should not crash the report
        info["torch"] = f"import failed: {exc}"
        return info
    t: dict[str, Any] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    t["cuda_version"] = getattr(torch.version, "cuda", None)
    if t["cuda_available"]:
        t["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    t["mps_available"] = bool(mps and mps.is_available())
    info["torch"] = t
    return info


def tool_version(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"available": False}
    out = run([path, "-hide_banner", "-version"]) or run([path, "-version"])
    first = out.splitlines()[0] if out else None
    return {"available": True, "path": path, "version": first}


def ffmpeg_features(ffmpeg: str | None) -> dict[str, bool]:
    if not ffmpeg:
        return {}
    filters = run([ffmpeg, "-hide_banner", "-filters"]) or ""
    encoders = run([ffmpeg, "-hide_banner", "-encoders"]) or ""
    return {
        "libx264": " libx264 " in encoders,
        "aac": " aac " in encoders,
        "drawtext": " drawtext " in filters,
        "scdet (scene detection)": " scdet " in filters,
        "silencedetect": " silencedetect " in filters,
        "ebur128 (loudness)": " ebur128 " in filters,
    }


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dist in DEPENDENCIES:
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def hf_check(offline: bool) -> dict[str, Any]:
    token_present = bool(os.environ.get("HF_TOKEN")) or (Path.home() / ".cache/huggingface/token").exists()
    result: dict[str, Any] = {"token_present": token_present}
    if offline:
        result["status"] = "skipped (--offline)"
        return result
    try:
        with urllib.request.urlopen(HF_DATASET_API, timeout=8) as resp:
            data = json.load(resp)
        result.update(status="reachable", dataset=data.get("id"), gated=data.get("gated"), private=data.get("private"))
    except urllib.error.HTTPError as exc:
        result["status"] = f"HTTP {exc.code} ({'auth required' if exc.code in (401, 403) else exc.reason})"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result["status"] = f"unreachable: {exc}"
    return result


def recommendations(report: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    py = tuple(int(x) for x in report["python"]["version"].split(".")[:2])
    if py < MIN_PYTHON:
        recs.append(
            f"Python {report['python']['version']} is older than {MIN_PYTHON[0]}.{MIN_PYTHON[1]}. "
            "Create the project venv with Python 3.11 (see README)."
        )
    if not report["ffmpeg"]["available"] or not report["ffprobe"]["available"]:
        recs.append("Install FFmpeg (macOS: `brew install ffmpeg`; Ubuntu: `sudo apt install ffmpeg`). Media probing depends on it.")
    torch_info = report["gpu"]["torch"]
    cuda = isinstance(torch_info, dict) and torch_info.get("cuda_available")
    if not report["gpu"]["nvidia_smi"] and not cuda:
        mps = isinstance(torch_info, dict) and torch_info.get("mps_available")
        recs.append(
            "No NVIDIA GPU/CUDA: CUDA-only research code (e.g. SnapUGC ECR inference) will not run locally. "
            + ("Apple MPS is available for small PyTorch models. " if mps else "")
            + "Prefer CPU-friendly models (faster-whisper int8, small CLIP/OCR) or a rented GPU."
        )
    mem = report["memory"].get("total_gb")
    if mem and mem < 24:
        recs.append(f"{mem} GiB RAM: keep to one model loaded at a time; avoid multi-GB checkpoints.")
    if report["disk"]["free_gb"] < 30:
        recs.append(f"Only {report['disk']['free_gb']} GB free disk: do not download full datasets or large checkpoints.")
    if not report["dependencies"].get("pytest"):
        recs.append("Install dev requirements: `pip install -r requirements.txt`.")
    return recs


def build_report(offline: bool, skip_torch: bool) -> dict[str, Any]:
    disk = shutil.disk_usage(PROJECT_ROOT)
    ffmpeg = tool_version("ffmpeg")
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": {"version": platform.python_version(), "executable": sys.executable, "in_venv": sys.prefix != sys.base_prefix},
        "os": {"system": platform.system(), "release": platform.release(), "platform": platform.platform()},
        "cpu": cpu_info(),
        "memory": memory_info(),
        "disk": {"free_gb": round(disk.free / 1e9, 1), "total_gb": round(disk.total / 1e9, 1)},
        "gpu": gpu_info(skip_torch),
        "ffmpeg": ffmpeg,
        "ffprobe": tool_version("ffprobe"),
        "ffmpeg_features": ffmpeg_features(ffmpeg.get("path")),
        "dependencies": dependency_versions(),
        "huggingface": hf_check(offline),
    }
    report["recommendations"] = recommendations(report)
    return report


def to_markdown(r: dict[str, Any]) -> str:
    def yn(v: Any) -> str:
        return "yes" if v is True else "no" if v is False else str(v)

    lines = [
        "# Hookline environment report",
        "",
        f"Generated: {r['generated_at']}",
        "",
        "## System",
        "",
        f"- Python: {r['python']['version']} ({r['python']['executable']}; venv: {yn(r['python']['in_venv'])})",
        f"- OS: {r['os']['platform']}",
        f"- CPU: {r['cpu']['name']} ({r['cpu']['machine']}, {r['cpu']['logical_cores']} logical cores)",
        f"- Memory: {r['memory']['total_gb']} GiB total, ~{r['memory']['available_gb']} GiB available ({r['memory']['via']})",
        f"- Disk: {r['disk']['free_gb']} GB free of {r['disk']['total_gb']} GB",
        "",
        "## GPU",
        "",
        f"- nvidia-smi: {r['gpu']['nvidia_smi'] or 'not found'}",
        f"- PyTorch: {json.dumps(r['gpu']['torch']) if isinstance(r['gpu']['torch'], dict) else r['gpu']['torch']}",
        "",
        "## FFmpeg",
        "",
        f"- ffmpeg: {r['ffmpeg'].get('version') or 'NOT FOUND'}",
        f"- ffprobe: {r['ffprobe'].get('version') or 'NOT FOUND'}",
    ]
    lines += [f"- {k}: {yn(v)}" for k, v in r["ffmpeg_features"].items()]
    lines += ["", "## Python packages", "", "| Package | Version | Purpose |", "|---|---|---|"]
    lines += [f"| {k} | {v or 'not installed'} | {DEPENDENCIES[k]} |" for k, v in r["dependencies"].items()]
    hf = r["huggingface"]
    lines += [
        "",
        "## Hugging Face",
        "",
        f"- API check ({HF_DATASET_API}): {hf['status']}",
        f"- Token configured: {yn(hf['token_present'])} (value never printed)",
    ]
    if "gated" in hf:
        lines.append(f"- tcm03/SnapUGC gated: {yn(hf['gated'])}, private: {yn(hf['private'])}")
    lines += ["", "## Recommendations", ""]
    lines += [f"- {x}" for x in r["recommendations"]] or ["- None."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="skip network checks")
    parser.add_argument("--skip-torch", action="store_true", help="do not import torch (faster)")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    report = build_report(args.offline, args.skip_torch)
    md = to_markdown(report)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "environment_report.md").write_text(md, encoding="utf-8")
    (args.report_dir / "environment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(md)
    print(f"Saved: {args.report_dir / 'environment_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
