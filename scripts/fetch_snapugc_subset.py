#!/usr/bin/env python3
"""Fetch SnapUGC metadata and a tiny, ECR-stratified video subset.

    python scripts/fetch_snapugc_subset.py                 # dry run: metadata + plan only
    python scripts/fetch_snapugc_subset.py --yes --n 12    # download 12 videos (~25 MB)
    python scripts/fetch_snapugc_subset.py --repo-samples /path/to/SnapUGC_Engagement/ECR_inference/dataset/test_videos

Sources (verified 2026-10-09; see reports/snapugc_feasibility.md):
  - ECR labels: train_data.csv / val_data.csv on Google Drive (linked from dasongli1/SnapUGC_Engagement).
  - Videos: Hugging Face mirror tcm03/SnapUGC, folder train/ only. The Snapchat CDN
    links inside the CSVs return 404.
Caveat: the hosted videos are almost all from the bottom of the ECR range (<= ~0.52).

Never downloads the full dataset. Existing files are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline import paths  # noqa: E402
from hookline.manifest import format_bool, write_manifest  # noqa: E402
from hookline.media import STATUS_OK, probe_video  # noqa: E402

GDRIVE = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
METADATA_FILES = {
    "train_data.csv": "1Mv5Esq5gGuxRTayabRUb5NmHwEN3JdbD",  # Id, Title, Description, Download_link, ECR
    "val_data.csv": "1iY7qUfqivYzUqUID501dKQDpI3NQil7y",  # Id, Title, Description, Download_link (no labels)
}
HF_API = "https://huggingface.co/api/datasets/tcm03/SnapUGC?blobs=true"
HF_FILE = "https://huggingface.co/datasets/tcm03/SnapUGC/resolve/main/{path}"
LABEL_SOURCE = "SnapUGC train_data.csv (Google Drive link in dasongli1/SnapUGC_Engagement); ECR per Li et al., ECCV 2024"
LICENSE = (
    "research use only: Snapchat UGC from SnapUGC (no license file in original repo; HF mirror tcm03/SnapUGC "
    "self-tagged apache-2.0). Do not redistribute or show publicly."
)
MAX_VIDEOS = 50
TIMEOUT = 60


def download(url: str, dest: Path, expected_size: int | None = None) -> bool:
    """Download to dest via a .part file. Returns True if a download happened."""
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hookline/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    if expected_size is not None and tmp.stat().st_size != expected_size:
        tmp.unlink()
        raise IOError(f"size mismatch for {dest.name}")
    with tmp.open("rb") as f:
        head = f.read(200).lower()
    if b"<html" in head or b"<!doctype" in head:
        tmp.unlink()
        raise IOError(f"{dest.name}: got an HTML page instead of data (Google Drive quota/permission?)")
    tmp.replace(dest)
    return True


def load_ecr(train_csv: Path) -> dict[str, float]:
    csv.field_size_limit(2**31 - 1)
    with train_csv.open(newline="", encoding="utf-8") as f:
        return {r["Id"]: float(r["ECR"]) for r in csv.DictReader(f) if r.get("ECR")}


def hosted_index(meta_dir: Path, ecr: dict[str, float], refresh: bool) -> list[dict]:
    index_path = meta_dir / "hf_hosted_index.csv"
    if index_path.exists() and not refresh:
        with index_path.open(newline="") as f:
            return [{"id": r["id"], "path": r["path"], "size": int(r["size"]), "ecr": float(r["ecr"])} for r in csv.DictReader(f)]
    with urllib.request.urlopen(HF_API, timeout=TIMEOUT) as resp:
        siblings = json.load(resp)["siblings"]
    rows = []
    for s in siblings:
        name = s["rfilename"]
        vid = Path(name).stem
        if name.endswith(".mp4") and vid in ecr:
            rows.append({"id": vid, "path": name, "size": s.get("size") or 0, "ecr": ecr[vid]})
    rows.sort(key=lambda r: r["ecr"])
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "path", "size", "ecr"])
        w.writeheader()
        w.writerows(rows)
    return rows


def stratified_pick(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Split rows (sorted by ECR) into n equal-count bins and pick one per bin."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["ecr"])
    picks = []
    for i in range(n):
        lo, hi = i * len(rows) // n, (i + 1) * len(rows) // n
        if hi > lo:
            picks.append(rows[rng.randrange(lo, hi)])
    return picks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=12, help=f"number of videos (max {MAX_VIDEOS})")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--yes", action="store_true", help="actually download the videos")
    parser.add_argument("--refresh-index", action="store_true", help="re-query the Hugging Face file list")
    parser.add_argument("--repo-samples", type=Path, help="register the 10 unlabeled val videos from a cloned SnapUGC repo")
    parser.add_argument("--out-dir", type=Path, default=paths.snapugc_dir())
    args = parser.parse_args()
    if not 0 < args.n <= MAX_VIDEOS:
        parser.error(f"--n must be between 1 and {MAX_VIDEOS}")

    out_dir: Path = args.out_dir
    meta_dir = out_dir / "metadata"
    try:
        for name, file_id in METADATA_FILES.items():
            if download(GDRIVE.format(id=file_id), meta_dir / name):
                print(f"downloaded {name}")
        ecr = load_ecr(meta_dir / "train_data.csv")
        index = hosted_index(meta_dir, ecr, args.refresh_index)
    except (urllib.error.URLError, IOError, KeyError, ValueError) as exc:
        print(f"ERROR fetching metadata: {exc}\nSee README 'Manual SnapUGC fallback'.", file=sys.stderr)
        return 1

    print(f"ECR labels: {len(ecr)} train videos. Hosted on HF with ECR: {len(index)} "
          f"(ECR {min(r['ecr'] for r in index):.3f}-{max(r['ecr'] for r in index):.3f}).")
    picks = stratified_pick(index, args.n, args.seed)
    total_mb = sum(r["size"] for r in picks) / 1e6
    print(f"Plan: {len(picks)} videos, {total_mb:.1f} MB")
    for r in picks:
        print(f"  {r['id']}  ECR={r['ecr']:.4f}  {r['size'] / 1e6:.2f} MB")

    video_dir = out_dir / "videos"
    rows = []
    for r in picks:
        dest = video_dir / f"{r['id']}.mp4"
        if args.yes:
            try:
                if download(HF_FILE.format(path=r["path"]), dest, r["size"]):
                    print(f"downloaded {dest.name}")
            except (urllib.error.URLError, IOError) as exc:
                print(f"  failed {r['id']}: {exc}", file=sys.stderr)
                continue
        if dest.exists() and dest.stat().st_size == r["size"]:
            rows.append((r, dest))
    if not args.yes:
        print(f"Dry run: {len(rows)}/{len(picks)} already on disk. Re-run with --yes to download the rest.")

    manifest_rows = []
    for r, dest in rows:
        probe = probe_video(dest)
        ok = probe.status == STATUS_OK
        manifest_rows.append({
            "video_id": r["id"],
            "file_path": str(dest.relative_to(out_dir)),
            "duration_seconds": probe.duration_seconds if ok else "",
            "has_audio": format_bool(probe.has_audio if ok else None),
            "has_video": format_bool(probe.has_video if ok else None),
            "source": "SnapUGC train split via huggingface.co/datasets/tcm03/SnapUGC",
            "license_or_permission": LICENSE,
            "dataset_split": "train",
            "notes": "" if ok else f"probe: {probe.error}",
            "label_name": "ECR",
            "label_value": f"{r['ecr']:.6f}",
            "label_source": LABEL_SOURCE,
        })

    if args.repo_samples:
        src = args.repo_samples
        if not src.is_dir():
            print(f"ERROR: --repo-samples {src} is not a directory", file=sys.stderr)
            return 1
        dest_dir = out_dir / "repo_val_samples"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.mp4")):
            dest = dest_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
            probe = probe_video(dest)
            ok = probe.status == STATUS_OK
            manifest_rows.append({
                "video_id": f.stem,
                "file_path": str(dest.relative_to(out_dir)),
                "duration_seconds": probe.duration_seconds if ok else "",
                "has_audio": format_bool(probe.has_audio if ok else None),
                "has_video": format_bool(probe.has_video if ok else None),
                "source": "SnapUGC val sample bundled in github.com/dasongli1/SnapUGC_Engagement",
                "license_or_permission": LICENSE,
                "dataset_split": "val",
                "notes": "no public label (challenge validation set)" if ok else f"probe: {probe.error}",
            })
        print(f"registered {len(list(src.glob('*.mp4')))} repo val samples")

    if manifest_rows:
        path = write_manifest(out_dir / "manifest.csv", manifest_rows, include_labels=True)
        print(f"Wrote {path} ({len(manifest_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
