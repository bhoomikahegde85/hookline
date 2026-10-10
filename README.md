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

## Layout

```text
hookline/
├── data/
│   ├── raw/        your own permitted videos + manifest.csv (videos git-ignored)
│   ├── sample/     synthetic edge-case fixtures (generated) + manifest.csv
│   └── snapugc/    tiny SnapUGC research subset (git-ignored, see its README)
├── artifacts/      cache/ (proxies, WAVs) and features/ (git-ignored)
├── docs/feature_schema.md       every output column, unit and status rule
├── reports/        generated inspection reports + snapugc_feasibility.md
├── scripts/
│   ├── inspect_environment.py   system, GPU, FFmpeg, packages, HF access
│   ├── inspect_dataset.py       probe videos, validate manifests, summarise metadata
│   ├── make_synthetic_samples.py
│   ├── fetch_snapugc_subset.py
│   ├── run_pipeline.py          preprocess + extract features
│   ├── run_speech_ocr.py        speech + on-screen text on top of the cached features
│   ├── run_ttcc.py              TTCC predicted retention for named videos + video_report.md (opt-in)
│   ├── ttcc_infer.py            TTCC runner; runs only in the TTCC environment (docs/ttcc.md)
│   └── show_features.py         readable view of one video's features
├── src/hookline/   probing, manifests, preprocess, segments, visual, audio, features, pipeline,
│                   speech, ocr, multimodal (Task 3 orchestration), context, ttcc, video_report
├── requirements-models.txt      optional: faster-whisper + PaddleOCR (Task 3)
├── requirements-ttcc.txt        optional: TTCC inference, in its own GPU environment
└── tests/
```

## Setup (macOS / Linux)

Requirements:
- Python 3.10 or newer (3.11 recommended).
- FFmpeg on `PATH` (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg`
  on Ubuntu).

Python dependencies are `numpy` and `opencv-python-headless` (feature
extraction), `pytest`, and `librosa` (so the librosa tests also run in this venv;
the pipeline in this venv doesn't use it). No pretrained models; everything runs on CPU.

```bash
cd hookline
python3.11 -m venv .venv            # or: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt     # or: uv pip install -r requirements.txt
```

On this machine the default `python3` is pyenv's 3.8.11, which is too old. The
venv was created with:
`uv venv --python ~/.pyenv/versions/3.11.9/bin/python3.11 .venv`.

### Optional: speech and on-screen text models (Task 3)

Only `run_speech_ocr.py` needs these; the tests mock them. They go in a
**separate venv**: PaddleOCR pins `opencv-contrib-python==4.10.0.84` and
`numpy<2.4`, which would replace the OpenCV/numpy that the features in `.venv`
were built with. On this Mac the models venv is 934 MB. Models are fetched on
first use: Whisper `small` 464 MB (`base` ≈ 145 MB, `tiny` ≈ 75 MB) and
PP-OCRv5 mobile det + English rec 12.5 MB.

```bash
uv venv --python ~/.pyenv/versions/3.11.9/bin/python3.11 .venv-models
uv pip install --python .venv-models/bin/python -r requirements-models.txt
python scripts/run_pipeline.py --input <videos>                 # features from the main .venv first
.venv-models/bin/python scripts/run_speech_ocr.py --input <videos> --limit 1 --download-models   # one-time model fetch
```

Models are never downloaded without `--download-models`. Without the
packages or models, the stage still runs and records `failed` with the
reason. It never reports "no speech" or "no text" in that case.

The same venv also runs the librosa audio features (`librosa.json`, on by default,
`--skip-librosa` to skip). Edit observations (`recommendations.json`) and the heuristic scores
are computed from the measured features with fixed rules; no language model or paid API is involved.

`--siglip` (off by default) adds SigLIP 2 frame embeddings and description scores
(`google/siglip2-base-patch16-224`, pinned revision). It needs `torch` + `transformers` from
`requirements-models.txt` (about 555 MB installed). The first run needs `--download-models`
to fetch about 1.5 GB into `~/.cache/huggingface`. No Hugging Face token is needed for this
public model. If you want one anyway (it raises download rate limits), run
`.venv-models/bin/hf auth login` or `export HF_TOKEN=...`. For SigLIP, `HF_TOKEN` in `.env` has no
effect: `.env` is read only for the TRIBE endpoint (`TRIBE_ENDPOINT_URL`, `HF_TOKEN`). The model runs on MPS (Apple GPU) when
available, else CPU.

## Commands

```bash
# 1. Environment report  -> reports/environment_report.md/.json
python scripts/inspect_environment.py              # add --offline to skip the Hugging Face check

# 2. Generate synthetic edge-case fixtures  -> data/sample/*.mp4 + manifest.csv  (~3.5 MB)
python scripts/make_synthetic_samples.py

# 3. (Optional) SnapUGC metadata + tiny subset. Dry run by default.
python scripts/fetch_snapugc_subset.py                       # ~33 MB of label CSVs, prints plan
python scripts/fetch_snapugc_subset.py --yes --n 12          # + 12 videos (~28 MB)
#   add --repo-samples <clone>/ECR_inference/dataset/test_videos to register the 10 repo val clips

# 4. Dataset report  -> reports/dataset_report.md/.json
python scripts/inspect_dataset.py                  # data/raw, data/sample, data/snapugc
python scripts/inspect_dataset.py --decode-check   # also fully decode every file
python scripts/inspect_dataset.py --videos data/raw --manifests data/raw/manifest.csv --strict

# 5. Preprocess + extract features  -> artifacts/cache/, artifacts/features/
python scripts/run_pipeline.py                         # data/raw + data/sample
python scripts/run_pipeline.py --input path/to/videos  # any dir or file (repeatable)
python scripts/run_pipeline.py --force                 # rebuild instead of using caches

# 6. Inspect features
python scripts/show_features.py --list
python scripts/show_features.py syn_text_slides_9s --segments 0

# 7. Speech + on-screen text (run with .venv-models/bin/python, see setup; reuses step 5's cache)
python scripts/run_speech_ocr.py                                 # data/raw + data/sample
python scripts/run_speech_ocr.py --input 01-Hookline --artifacts artifacts/company --limit 10
python scripts/run_speech_ocr.py --skip-ocr --asr-model base     # speech only, smaller model
python scripts/run_speech_ocr.py --skip-speech --ocr-fps 1 --ocr-max-seconds 10
python scripts/run_speech_ocr.py --language hi --ocr-lang hi     # force Hindi ASR, Devanagari OCR
# Estimate OCR sampling misses: compare a 1 fps run against a denser reference run (e.g. --ocr-fps 4 into another root)
python scripts/compare_ocr_runs.py --ref artifacts/company_ocr_ref --test artifacts/company FC003 FL004 FL019
# One company reel end to end on this machine (features first, then the model stage; nothing is uploaded)
.venv/bin/python scripts/run_pipeline.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/demo
.venv-models/bin/python scripts/run_speech_ocr.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/demo --ocr-fps 1 --siglip
# SigLIP 2 visual-semantic signals (first run downloads ~1.5 GB; later runs read the local cache)
python scripts/run_speech_ocr.py --input data/sample/syn_av_testpattern_8s.mp4 --siglip --download-models
# Rebuild one reel's evidence timeline from cached speech/OCR (same flags as the original run; no model inference)
python scripts/run_speech_ocr.py --input 01-Hookline/face/videos/FC003.mp4 --artifacts artifacts/company --ocr-fps 1 --ocr-max-seconds 10

# 8. Tests
python -m pytest -q
```

`run_pipeline.py` and `run_speech_ocr.py` exit with code 1 if any video
failed. On `data/sample/` that's expected, because the two `syn_corrupt_*`
fixtures fail on purpose. `run_speech_ocr.py` also exits with 1 when the
models aren't installed, since every video's speech/text is then `failed`.
Pass the same `--window` / `--proxy-fps` to both scripts so the features cache
is shared.

Order matters: Task 2 features come only from `.venv/bin/python scripts/run_pipeline.py`. It
refuses `.venv-models` (exit 3) and refuses to mix library versions within one artifacts root
(`feature_env.json`). `run_speech_ocr.py` only reads those features; a video without them is
reported `failed` with the command to run. Run-level CSVs and summaries are rebuilt from every
video in the root, so re-running one video never drops the others.

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

### TTCC drop-off analysis

`scripts/show_dropoffs.py <video_id> --artifacts <root>` prints, second by second, where TTCC's predicted retention falls and what was
measured at that moment (words, text, cuts, quiet audio), with unvalidated rule-based candidate causes. It reads saved files only. See
[docs/ttcc.md](docs/ttcc.md).

## Feature outputs

| File | Contents |
|---|---|
| `artifacts/features/<video_id>/segments.csv` | one row per segment: times, scene index, modality status, visual + audio values, warnings |
| `artifacts/features/<video_id>/frames.csv` | per proxy frame (10 fps) visual measurements and cut flags |
| `artifacts/features/<video_id>/report.json` | provenance, probe, timestamp map, config + versions, warnings/errors, opening summary, segments |
| `artifacts/features/first5s_summary.csv` | one row per video: valid windows and status coverage per modality in the first 5 s (or the whole clip if shorter) |
| `artifacts/features/segments_all.csv` | all segment rows of every video in the artifacts root (rebuilt each run) |
| `artifacts/features/run_summary.json` | per-video ok / partial / failed for every video in the root, plus `last_run` |
| `artifacts/features/feature_env.json` | OpenCV / NumPy / FFmpeg / Python that built this root's features |
| `artifacts/features/<video_id>/speech.json` | language, Whisper segments and words with timestamps, confidence flags, status and reason |
| `artifacts/features/<video_id>/ocr.json` | sampled frame times, deduplicated text events (first/last seen, box, repeats), per-frame detections, status |
| `artifacts/features/<video_id>/segments_merged.csv` | evidence timeline: every `segments.csv` column plus speech and text columns per segment, with `text_evidence` (observed/inferred) and links into `words.csv` / `text_observations.csv` |
| `artifacts/features/<video_id>/words.csv` | one row per word: word times, probability, segment |
| `artifacts/features/<video_id>/text_observations.csv` | one row per OCR detection/sample: sample time, box, event id, segment, failed samples |
| `artifacts/features/<video_id>/opening_multimodal.json` | first 5 s: visual, audio, speech (first word, transcript, words/s) and text (first text, events) |
| `artifacts/features/<video_id>/librosa.json` | librosa per-segment audio features, low-energy intervals, global tempo (when reliable) |
| `artifacts/features/<video_id>/siglip.json`, `siglip_embeddings.npy` | SigLIP 2 (opt-in): sampled frame times/ids, description scores, frame-to-frame change; float16 embeddings |
| `artifacts/features/<video_id>/scores.json`, `segment_scores.csv` | per-segment `attention_risk` and `moment_quality` (0–100 heuristics) with every point's source; top spans |
| `artifacts/features/<video_id>/recommendations.json` | rule-based edit observations, each with interval, evidence ids and reason |
| `artifacts/features/<video_id>/ttcc_status.json`, `ttcc_retention.json`, `ttcc_raw.json`, `ttcc_retention.svg` | TTCC (opt-in): latest attempt status; predicted R(t) at 1 s steps with provenance; raw 60-value head output; plot. See [docs/ttcc.md](docs/ttcc.md) |
| `artifacts/features/<video_id>/tribe.json`, `tribe_preds.npy` | TRIBE v2 (opt-in, remote endpoint): status, provenance, segment timestamps; float32 predicted brain responses (n_segments × 20484 fsaverage5 vertices). See [docs/tribe.md](docs/tribe.md) |
| `artifacts/features/<video_id>/video_report.md` | readable per-video report: measured features, heuristic scores, TTCC prediction, recommendations (written by `run_ttcc.py`) |
| `artifacts/features/segments_merged_all.csv`, `first5s_multimodal.csv`, `speech_ocr_run_summary.json` | run-level versions of the above, rebuilt from every video in the root (older-schema videos listed, not padded) |

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

## Synthetic fixtures

`make_synthetic_samples.py` generates the following clips with FFmpeg, so they
contain no third-party content:

| ID | Exercises |
|---|---|
| `syn_av_testpattern_8s` | baseline audio + video |
| `syn_silent_no_audio_stream_6s` | silent video, no audio stream |
| `syn_silent_audio_track_5s` | audio stream present but silent |
| `syn_text_slides_9s` | text-led slideshow, no audio |
| `syn_static_image_tone_7s` | static image, audio-led |
| `syn_low_quality_visual_audio_6s` | dark, noisy, low-res visuals with clear audio |
| `syn_very_short_0p4s` | very short clip |
| `syn_audio_only_5s` | MP4 with no video stream |
| `syn_corrupt_truncated` | truncated MP4 (unreadable) |
| `syn_corrupt_zero_bytes` | empty file |

They're pipeline fixtures, not realistic content. Real talking-head, meme,
gaming and montage clips still need to go in `data/raw/`.

## SnapUGC in one paragraph

ECR is the share of viewers who watched at least 5 seconds. It's one number per
video, not a retention curve. Labels exist for 106k training videos, but the
Snapchat CDN links have expired. Only about 9.7k videos are mirrored on Hugging
Face, and almost all of those are low-ECR. The pretrained model needs about
13 GB of checkpoints and CUDA, so it won't run on this Mac. Use SnapUGC as
research context and a small real-world test set, not as training data this
weekend. Full details: [reports/snapugc_feasibility.md](reports/snapugc_feasibility.md).

## Speech and OCR on a GPU (Kaggle)

The M2 runs Whisper `small` (int8) and PP-OCRv5 mobile on CPU, which is
enough for a pilot. For larger models (`large-v3`) or all reels at higher OCR
rates, use a GPU notebook:

```bash
pip install faster-whisper paddleocr      # plus the paddlepaddle-gpu build matching the CUDA version (see paddlepaddle.org.cn install guide);
                                          # if `import cv2` breaks, keep only one of opencv-python / opencv-contrib-python
python scripts/run_pipeline.py --input <videos>
python scripts/run_speech_ocr.py --input <videos> --download-models \
    --asr-model large-v3 --asr-device cuda --asr-compute-type float16 --ocr-device gpu:0
```

Copy `artifacts/features/<video_id>/{speech,ocr}.json` back into the same
local folders. They're reused if you pass the same model and config flags
(`--asr-device` isn't part of the cache key, `--asr-compute-type` is).

**Company reels:** the dataset README says "Provided for this hackathon
only. Please don't publish or redistribute." Uploading the reels to Kaggle,
even as a private dataset, copies them to a third party. Ask the organisers
before doing that. Otherwise, use Kaggle only for your own or licensed clips
and run the company reels locally.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `speech=failed ... faster-whisper is not installed` / `paddleocr/paddlepaddle are not installed` | Run with `.venv-models/bin/python` (see setup), not `.venv`. |
| `wrong environment for Task 2 features` (exit 3) | Run `run_pipeline.py` with `.venv/bin/python`. If the root's features came from other library versions on purpose, add `--new-feature-env` (recomputes that root's features). |
| `no Task 2 features for <id>` / `features predate dependency provenance` | Run the printed `.venv/bin/python scripts/run_pipeline.py ...` command first, with the same `--artifacts`/`--window`/`--proxy-fps`. |
| `run-level files exclude N videos with older per-video outputs` | Those videos were processed by an older version; re-run them to bring them back into the aggregates. |
| Speech/OCR slow on the Mac | Measured on 3 company reels: Whisper `small` ran at 0.14–1.42× real time (slowest on music-heavy audio) and OCR took 0.6–1.1 s per 720p frame. Use `--ocr-fps 1`, `--ocr-max-seconds 10`, or a GPU. |
| `could not load Whisper model ... --download-models` / `OCR models not in ~/.paddlex/...` | The model isn't cached locally. Re-run once with `--download-models`. |
| Speech text on music-only reels | Whisper can hallucinate on music. Such segments are flagged (`low_confidence`, `possible_non_speech`, `repetitive`) and make the file `uncertain` if nothing else is confident. |
| Hindi/other captions read as garbage | The default OCR recogniser is English/Latin. Use `--ocr-lang hi` (or the relevant code). |
| `ffprobe` / `ffmpeg` NOT FOUND in reports | Install FFmpeg and reopen the terminal. Without it, probing falls back to OpenCV (if installed) and audio is reported as unknown. |
| `SyntaxError` or odd type errors | You're on Python older than 3.10. Activate `.venv` (`which python` should point inside it). |
| `No module named pytest` | `pip install -r requirements.txt` inside the venv. |
| `manifest not found` | Paths are relative to the project root. Run from `hookline/` or pass `--manifests`. |
| `file not found` errors in a manifest | `file_path` is resolved relative to the manifest's own folder. |
| SnapUGC fetch says "got an HTML page" | Google Drive quota or permission page. Use the manual fallback in `reports/snapugc_feasibility.md`. |
| Hugging Face check `unreachable` | Network or proxy issue. Run with `--offline`; nothing else depends on it. |
| Corrupt files listed as errors | Expected for `syn_corrupt_*`. For your own files, re-export or re-download them. |

## Data hygiene

`.gitignore` excludes:
- All media, SnapUGC data and the large metadata CSVs.
- The hackathon company dataset (`01-Hookline/`) and its zip. They're for
  this hackathon only and mustn't be committed or redistributed.
- Checkpoints (`*.pth`, `*.safetensors`, ...) and `artifacts/` (including
  `speech.json`/`ocr.json`, which contain transcripts and on-screen text from the videos).
  Whisper and PaddleOCR models are cached outside the repo (`~/.cache/huggingface`, `~/.paddlex`).
- Generated reports, `.env` files and tokens (`.env.example` is tracked and holds no secrets).

Only code, READMEs, manifests and `reports/snapugc_feasibility.md` are tracked.
