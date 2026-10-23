# Hookline

Hookline analyses a short-form video before it's published. It looks at the
opening hook (especially the first 5 seconds), produces a timestamped
attention-risk timeline, gathers multimodal evidence (visuals, audio, speech,
on-screen text) and gives format-aware edit suggestions.

Design rules:
- A missing modality is never negative evidence.
- Hookline's own scores never claim to predict retention. The only retention
  curve is TTCC's: a pretrained model's prediction, labelled as such, kept separate
  from the heuristics, and not validated on these reels (there are no retention
  labels for them).

**Status: Phase 1, Task 3.** Task 1 set up the project foundation, the dataset
inspection tools and the test fixtures. Task 2 adds cached preprocessing
(analysis proxy, mono 16 kHz WAV, timestamp map) and timestamped low-level
visual and audio features in 0.5 s segments. Task 3 adds timestamped speech
(Whisper) and on-screen text (PaddleOCR) as a separate, optional stage, plus
opening-5-second summaries that combine all four modalities. There's no
scoring engine or frontend yet.

## TTCC predicted retention (opt-in)

`scripts/run_ttcc.py` runs the pretrained TTCC model
(`liangyuch/ttcc-sft-qwen25omni-3b-v8`, pinned revision) on videos you name with `--input`, and
writes a per-video `video_report.md`. That report keeps four things in separate sections:
measured features, heuristic scores, TTCC-predicted retention and the edit recommendations.

- TTCC is off unless you set `ENABLE_TTCC=true` or pass `--enable-ttcc`.
- It needs its own environment (`.venv-ttcc`: `requirements-ttcc.txt` plus the upstream `go_viral`
  checkout). It runs on this Mac (Apple GPU via MPS, or CPU) as well as on CUDA GPUs; the device is
  detected. Setup, the verified inference protocol, the differences from the upstream GPU setup and
  the limitations are in [docs/ttcc.md](docs/ttcc.md).
- With TTCC off, the script only rewrites the reports.

```bash
.venv/bin/python scripts/run_ttcc.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/company             # report only
.venv/bin/python scripts/run_ttcc.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/company --enable-ttcc
```

## TRIBE v2 brain-response predictions (opt-in, remote)

`run_speech_ocr.py --tribe` sends a video to **your own Hugging Face Inference Endpoint** running
TRIBE v2 (`facebook/tribev2`, CC BY-NC 4.0, non-commercial only). It stores the model's predicted
fMRI responses (an average subject on the fsaverage5 cortex, at 1 s steps) in `tribe.json` and
`tribe_preds.npy`.

- TRIBE is off by default. Nothing runs locally.
- The endpoint URL and token come from `TRIBE_ENDPOINT_URL` and `HF_TOKEN` (environment or the
  untracked `.env`).
- Uploading needs `--tribe-allow-remote`. Company reels also need `--tribe-approve <video_id>`.
- These outputs are not retention or engagement predictions.
- The endpoint handler is in `endpoint/tribev2/`. Deployment steps, licences and status are in
  [docs/tribe.md](docs/tribe.md).



## Feature outputs


Empty CSV cells and JSON `null` mean "not measured". They're never zero.
Columns, units, status rules and the timeline convention are documented in
[docs/feature_schema.md](docs/feature_schema.md). The features describe the
video. They're not engagement predictions, and more cuts, motion or loudness
isn't treated as better.

Caching: the proxy and WAV are keyed by the source's SHA-256 plus the
preprocessing settings, and features by the full config. Re-running skips
unchanged work, and changing only feature settings (for example `--window`)
reuses the proxies. To use a separate output root (for example for the
company dataset), pass `--artifacts artifacts/company`.

`--strict` makes `inspect_dataset.py` exit with code 1 if it finds any error.
Don't use it on `data/sample/`: two of those fixtures are deliberately corrupt.

## Adding your own videos

Copy clips you have permission to use into `data/raw/` and add a row to
`data/raw/manifest.csv`. See [data/raw/README.md](data/raw/README.md) for the
columns and an example. Then run `python scripts/inspect_dataset.py`. Don't
scrape Instagram, TikTok or YouTube.


## SnapUGC in one paragraph

ECR is the share of viewers who watched at least 5 seconds. It's one number per
video, not a retention curve. Labels exist for 106k training videos, but the
Snapchat CDN links have expired. Only about 9.7k videos are mirrored on Hugging
Face, and almost all of those are low-ECR. The pretrained model needs about
13 GB of checkpoints and CUDA, so it won't run on this Mac. Use SnapUGC as
research context and a small real-world test set, not as training data this
weekend. Full details: [reports/snapugc_feasibility.md](reports/snapugc_feasibility.md).

