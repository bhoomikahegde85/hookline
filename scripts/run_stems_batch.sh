#!/bin/bash
# Demucs music/voice levels for every reel under 01-Hookline/{face,faceless,outliers} (local, nothing uploaded); finished reels are skipped.
cd "$(dirname "$0")/.."
for f in 01-Hookline/face/videos/*.mp4 01-Hookline/faceless/videos/*.mp4 01-Hookline/outliers/videos/*.mp4; do
  .venv-demucs/bin/python scripts/run_stems.py --input "$f" --artifacts artifacts/stems 2>&1 | grep -v Warning
done
echo "stems batch done"
