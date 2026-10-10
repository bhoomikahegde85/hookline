# TTCC retention predictions

TTCC is Hookline's first pretrained audience-retention model. For a 5–60 s window of video it predicts
**R(t)**: the fraction of viewers still watching at second *t*, for each whole second. Videos longer than one window
(30 s by default) are predicted in overlapping windows and merged into one curve (see "Videos longer than 30 s").

What TTCC is not:
- It is not measured retention.
- It has not been validated on the company reels, because no ground-truth retention exists for them.
- It is not a virality score.
- It is independent of Hookline's heuristic `attention_risk` / `moment_quality` scores and of the
  edit recommendations. Nothing combines them.

## Model and verified inference protocol

| | |
|---|---|
| Checkpoint | [`liangyuch/ttcc-sft-qwen25omni-3b-v8`](https://huggingface.co/liangyuch/ttcc-sft-qwen25omni-3b-v8) at revision `1686d5f00408bfd8f6a32be12e71248b6f8ff447` (checkpoint-225 of run v19, about 9.4 GB in 2 bf16 shards) |
| Architecture | Qwen2.5-Omni-3B (thinker only, talker off) plus `RetentionHead`: `Linear(2048 → 60)`, softplus hazards, `R(t) = exp(-cumsum(λ))` |
| Loading | only through `examples/custom/qwen2_5_omni_retention/register.py` from [go_viral](https://github.com/cliangyu/go_viral/tree/ttcc-rl) on branch `ttcc-rl` (written against commit `893da03b`), with `model_type='qwen2_5_omni_retention'` |
| Protocol | the one behind the published metric (`rl/srcc_eval.py`, cross-ad SRCC 0.5142, n=158): template `qwen2_5_omni_retention` in train mode; short V8 system/user prompts with `T = round(duration)`; `videos=[mp4]`, `audios=[]`; assistant span `<cot></cot>`; one forward pass; read `out.r_pred` |
| Video settings | `FPS=1`, `FPS_MAX_FRAMES=60`, `VIDEO_MAX_TOKEN_NUM=8192`, `VIDEO_MAX_PIXELS=MAX_PIXELS=49152`, `USE_AUDIO_IN_VIDEO=false`, `ENABLE_AUDIO_OUTPUT=False`, `RETENTION_HEAD_TYPE=hazard`, `max_length=32768`, bf16, `attn_impl=flash_attn` |
| Output | 60 values R(1..60). R(0)=1 is a convention and is prepended. The prediction keeps t = 0..T. The model was trained only on 5 ≤ T ≤ 60 |

Notes on the upstream material:
- The repo `github.com/cliangyu/ttcc-inference` returned 404 on 2026-10-10, so the protocol above
  comes from `go_viral` itself.
- `INFERENCE.md` in `go_viral` describes the withdrawn V7 checkpoints, which leaked labels into the
  input. It does not describe this v8 checkpoint.
- The v8 `config.json` says `model_type: qwen2_5_omni`. The retention head is only attached when
  you load it with `model_type='qwen2_5_omni_retention'` and the plugin.

No fallback to another model. Before every load, the runner checks that:
- the checkpoint files match the pinned sizes and SHA-256 hashes (in
  [config.py](../src/hookline/config.py)). The two weight shards are only hashed with
  `--verify-weights`; their sizes are always checked.
- `swift` is imported from the given go_viral checkout, not from another ms-swift install.
- the retention-head tensors in memory are exactly the ones stored in the checkpoint. This check
  matters because `register.py` silently skips missing head weights, which would leave a random head.
- the tokenizer has the multimodal tokens.

## How it fits into Hookline

```text
.venv / any env                              TTCC env (CUDA, Apple MPS or CPU)
scripts/run_ttcc.py ──► hookline.ttcc ──subprocess──► scripts/ttcc_infer.py ──► go_viral register.py + checkpoint
        │                 (validate, cache, normalise, plot)
        └─► video_report.md (measured | heuristic | TTCC | recommendations)
```

- TTCC is off by default (`ENABLE_TTCC=false`).
- `run_pipeline.py` and `run_speech_ocr.py` are unchanged. They never import TTCC code or need its
  dependencies.
- TTCC needs only the video file. Task 2 features are reused or built (`--features build|reuse|skip`)
  only so the report can show them.

Per video, in `<artifacts>/features/<video_id>/`:

| File | Contents |
|---|---|
| `ttcc_status.json` | Latest attempt. `success`, `failed` (bad input, inference error or invalid output) or `unavailable` (environment, checkpoint, or duration outside 5–60 s). Also stage, reason, cache key, and whether the stored prediction is current. |
| `ttcc_retention.json` | Last successful prediction: `timestamps_s` (0..T), `retention` (fraction 0–1), convention and units, summary, inference settings, provenance (go_viral commit, plugin hash, library versions, checkpoint files, head check, GPU), and the SHA-256 of the raw output. A later failure never replaces it. |
| `ttcc_raw.json` | The runner's raw output: all 60 head values, the exact input row, token count, video grid, and the `</cot>` anchor. |
| `ttcc_raw_rejected.json` | Raw output that failed validation, kept for inspection. No curve is produced from it. |
| `ttcc_retention.svg` | Plot of the predicted curve. One dot per whole second, axes labelled, no interpolated points. Only written on success. |
| `video_report.md` | The per-video report, with the four kinds of output in separate sections. |

Run level: `<artifacts>/features/ttcc_run_summary.json` has per-video status, reason, durations,
artifact paths, batch model-load and runner times, and the probe result.

The cache key covers:
- source SHA-256
- T
- every inference setting and prompt
- pinned checkpoint files
- go_viral commit, dirty flag and `register.py` hash
- runner hash
- versions of torch, transformers, qwen-omni-utils, video readers, flash-attn and ms-swift
- GPU model

Failed and unavailable attempts are always retried. `--force` re-runs a successful one.

Resolution: TTCC predicts on a 1 s grid. It is **not** resampled onto Hookline's 0.5 s segments,
because that would invent values between the model's outputs. Comparing it with segments is left
to the reader of the report.

## Running

Every video must be named with `--input`. There is no default set, so the 110 reels are never all
processed by accident.

```bash
# Report only (TTCC off): rewrites video_report.md from the artifacts on disk
.venv/bin/python scripts/run_ttcc.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/company
# One video, TTCC on (needs the TTCC environment; CUDA, Apple MPS or CPU)
.venv/bin/python scripts/run_ttcc.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/company --enable-ttcc
# A chosen set: repeat --input. One model load serves the whole batch, and a failed video doesn't stop the others.
.venv/bin/python scripts/run_ttcc.py --input a.mp4 --input b.mp4 --artifacts artifacts/company --enable-ttcc
python scripts/show_features.py FC001 --artifacts artifacts/company   # terminal view, now with the TTCC curve
```

Exit codes:
- `0`: no failure
- `1`: at least one video failed
- `2`: bad input, or FFmpeg is missing
- `3`: the TTCC environment, checkpoint or requested device is unavailable

## Recommended: run TTCC in your own Hugging Face Space (nothing heavy on your machine)

`endpoint/ttcc_space/` is a Docker Space with the same job API as the TRIBE Space. It clones go_viral
at the pinned commit, downloads the pinned checkpoint (about 9.4 GB, on every start: the disk is
wiped when a Space sleeps or is paused), and runs `ttcc_infer.py` for each video. Hookline sends the video,
polls the job and receives the raw output; validation, caching, plots and reports run locally and are unchanged.

1. Create a **private** Docker Space (Blank), for example `wlddhq/hooklinewldd-ttcc`, on an **L4 (24 GB) or L40S (48 GB)** GPU. No
   secrets are needed (the checkpoint is public). Set a sleep time you can live with, and pause it when idle.
2. Upload `Dockerfile`, `app.py`, `requirements.txt`, `README.md` from `endpoint/ttcc_space/` and `scripts/ttcc_infer.py` to the Space root.
3. Wait for the build, then for `https://<owner>-<name>.hf.space/health` (logged in) to show `"checkpoint_ready": true`.
4. In your untracked `.env`: `TTCC_ENDPOINT_URL=https://<owner>-<name>.hf.space` and `HF_TOKEN=<token that can read the Space>`.
5. Run (own clips need only `--ttcc-allow-remote`; company reels also need `--ttcc-approve <video_id>`, only after the organisers allow it):
   ```bash
   .venv/bin/python scripts/run_ttcc.py --input my_clip.mp4 --artifacts artifacts/demo --enable-ttcc --ttcc-remote --ttcc-allow-remote
   ```

Status: the Space service, the remote client, the permission gate and a real localhost HTTP round trip are
tested with a fake runner. **The Docker image has never been built and the model has never run on a Space.**
Expect to read the build and container logs on the first deploy. Likely trouble spots: dependency versions (the image
has torch 2.6; upstream used newer ones), GPU memory for long videos, and the missing `processor_config.json`.
Attention uses `sdpa` (flash-attn is not installed), recorded in each prediction as a deviation from upstream.

## Running locally without CUDA (MacBook Air M2, 16 GB)

TTCC runs on Apple Silicon through PyTorch's MPS backend (the Apple GPU), and on any machine's CPU.
You don't need to choose: `--device-map detect` (the default) picks CUDA, then MPS, then CPU.
`--attn-impl detect` picks `flash_attn` only where FlashAttention 2 can run, and `sdpa` everywhere else.

One-time setup (about 3 GB of packages plus the 9.4 GB checkpoint; check `df -h` first):

```bash
git clone -b ttcc-rl https://github.com/cliangyu/go_viral && git -C go_viral checkout 893da03bccee6a18d221e96c5de55c9b35a8e350
uv venv --python ~/.pyenv/versions/3.11.9/bin/python3.11 .venv-ttcc
uv pip install --python .venv-ttcc/bin/python torch==2.11.0 torchvision==0.26.0
uv pip install --python .venv-ttcc/bin/python -e ./go_viral
uv pip install --python .venv-ttcc/bin/python -r requirements-ttcc.txt
.venv-ttcc/bin/python scripts/ttcc_infer.py probe --repo go_viral     # expect "inference_possible": true, "video_reader": "decord"
```

> This route needs about 13 GB of disk and was killed by macOS memory pressure on a 16 GB Mac. Prefer the Space above.

Run it. `--download-models` is needed only the first time, to fetch the checkpoint into
`~/.cache/huggingface`:

```bash
.venv/bin/python scripts/run_ttcc.py --input 01-Hookline/face/videos/FC001.mp4 --artifacts artifacts/company \
    --enable-ttcc --download-models
```

How the local run differs from the checkpoint's published evaluation (CUDA, bf16, FlashAttention 2,
decord):
- MPS or CPU instead of a CUDA GPU. On MPS, `PYTORCH_ENABLE_MPS_FALLBACK=1` sends any operation MPS
  lacks to the CPU.
- `sdpa` attention instead of `flash_attn`.
- `eva-decord` (a decord build with macOS wheels) instead of `decord`. `torchvision` 0.24 and later
  can no longer decode video.

The prompts, frame sampling (1 frame/s, at most 60 frames, 8192 video tokens), audio-off setting,
checkpoint and head are identical. Each prediction lists these differences under
`inference.deviations_from_upstream_eval`. The device is part of the cache key, so a Mac prediction
and a GPU prediction never stand in for each other. Upstream observed that changing the attention
kernel shifts the head's input slightly, so treat small differences from a CUDA run as expected.

Memory: the bf16 weights take about 9.4 GB of the 16 GB of unified memory. With these video settings,
a 60 s reel is about 2,000 video tokens, so activations are small. Close other heavy apps before a
run. If MPS runs out of memory, `--device-map cpu` is slower but uses ordinary RAM.

## Kaggle or another CUDA GPU (optional, closest to upstream)

> **Company reels:** `01-Hookline/` is hackathon data that must not be redistributed. Uploading reels
> to Kaggle (even as a private dataset) needs the organisers' approval, the same rule as for TRIBE.

GPU notes:
- Kaggle offers T4 x2 (16 GB each, compute capability 7.5) and P100 (16 GB, 6.0).
- Neither supports FlashAttention 2, so attention detection picks `sdpa`. Upstream observed that the attention
  kernel shifts the head's input, so SDPA predictions can differ slightly from the published
  evaluation. The kernel is recorded in the prediction.
- Neither GPU has native bf16. If bf16 fails, `--dtype float16` is a deviation from upstream. It is
  recorded and changes the cache key.
- An L4, A10G or newer (24 GB, bf16 + FA2) matches upstream most closely.
- Upstream reports about 13 GB of GPU memory per ad at batch 1. `--device-map auto` splits the model
  over both T4s.

Notebook steps (GPU on, internet on):

```bash
git clone https://github.com/bhoomikahegde85/hookline.git          # this repo, without company data
git clone -b ttcc-rl https://github.com/cliangyu/go_viral
git -C go_viral checkout 893da03bccee6a18d221e96c5de55c9b35a8e350    # the commit the adapter was verified against
pip install -e go_viral                                                # ms-swift fork + its requirements
pip install -r hookline/requirements-ttcc.txt                          # keeps Kaggle's CUDA torch
python hookline/scripts/ttcc_infer.py probe --repo go_viral            # must print "inference_possible": true
hf download liangyuch/ttcc-sft-qwen25omni-3b-v8 --revision 1686d5f00408bfd8f6a32be12e71248b6f8ff447 \
    --local-dir /kaggle/working/ttcc-v8                                # ~9.4 GB (/kaggle/working limit is 20 GB)

# one explicitly selected reel first
python hookline/scripts/run_ttcc.py --input /kaggle/input/<dataset>/FC001.mp4 \
    --artifacts /kaggle/working/artifacts --features skip --enable-ttcc \
    --ttcc-python "$(which python)" --ttcc-repo go_viral --checkpoint-dir /kaggle/working/ttcc-v8 \
    --device-map auto     # attention detected: sdpa on T4/P100, flash_attn on Ampere+ with flash-attn installed
# optional real-model test on the same reel
cd hookline && HOOKLINE_TTCC_VIDEO=/kaggle/input/<dataset>/FC001.mp4 TTCC_PYTHON="$(which python)" \
    TTCC_REPO=../go_viral TTCC_CHECKPOINT_DIR=/kaggle/working/ttcc-v8 \
    python -m pytest -q tests/test_ttcc_integration.py
```

Bring results home:
1. Copy `/kaggle/working/artifacts/features/<id>/ttcc_*` into `artifacts/company/features/<id>/`.
2. Run `run_ttcc.py --input <reel> --artifacts artifacts/company` without `--enable-ttcc`. This
   rebuilds `video_report.md` with the prediction.

The status file from Kaggle stays as it was. A later local run with TTCC enabled would mark the
prediction "not confirmed current", because the Mac can't re-check it. It does not delete it.

## The editing assistant (`review.html`)

`scripts/run_e2e.py` (or `scripts/build_review.py <id> --artifacts <root> --video <file>`) writes one local HTML file per video. Open it on the
machine that has the video. It has:
- the video player, a **clickable timeline strip** (segments coloured by predicted drop / decline / hold) and the predicted retention curve,
  all **synchronised**: a playhead follows the video and the active segment is highlighted;
- a card per candidate segment: timestamps, the **observable evidence** measured in it, and **edit proposals** typed as `TRIM`, `ADD` or `REVIEW`;
- per proposal: **Jump to** the timestamp, an editable trim range with **Preview trim** (the video skips the range, 2 s of context each side),
  and **Accept / Reject / Unsure** with a note;
- **Play edited version**: playback skips every accepted trim, and the page shows the resulting duration;
- **Export edit decisions** as JSON (accepted trims, duration after trims, every proposal and decision) or CSV (one row per proposal).
  Decisions are kept in the browser until exported. Nothing is uploaded and the page loads nothing from the network.

Trim proposals come only from rules that mean "remove this" (`late_first_word`, `quiet_stretch`, `static_stretch`); the range is the flagged stretch and is
meant to be adjusted before accepting. Other rules propose `ADD` or `REVIEW` and cannot be previewed as a trim. A video may have no trim proposals
(FC002 has none). The page labels the curve as a **prediction** and says its accuracy on these videos has **not been verified**; every exported row carries
`data_kind = predicted_retention_unvalidated`. Preview skipping is checked every 0.1 s, so it can overshoot a trim edge slightly.

## Hook analysis (default): one pass over the opening

`long_videos="hook"` (default; `--long-videos hook`, `--hook-s 30`): the model sees the first `min(duration, 30)` s, is told the ad is that
long, and predicts R(t) for viewers who start at 0. `ttcc_retention.json -> hook` holds R(1), R(3), R(5), R(10) (None when the video is
shorter, never extrapolated) and the steepest one-second drop in the first 10 s. One pass per video, nothing stitched; the rest of a longer
video is marked "not predicted". If 30 s does not fit in GPU memory the runner shrinks the window (recorded in `inference.window_fit`).
`hook_prompt_s` (sensitivity checks only) tells the model a different length than it sees; such runs are flagged and never used for evaluation.
Sensitivity checks (`scripts/ttcc_sanity.py`): re-encoded, black and 1-second-shuffled openings, told-length variations, and different reels.

## Videos longer than 30 s: overlapping windows (experimental, `--long-videos windows`)

**Where the 30 s limit comes from (inspected, not assumed).** It is not a model limit. The retention head emits 60 values (`T_MAX = 60`), training used 5 to 60 s horizons
(rows over 32,768 tokens were dropped, so ads up to about 54 s were kept), and the backbone has a 32,768-token context with no sliding window. The practical limit is GPU memory
with `sdpa` attention (memory grows with the square of the input): on the L40S (48 GB) 25 s (14,434 tokens) fits, 37 s and 54 s ran out of memory (measured on FC003, 82.5 s); a
24 GB L4 runs out at about 14k tokens. FlashAttention (not installed) or a larger GPU would lift it, which is why `window_s` is configurable up to 60. Nothing is trained beyond 60 s.

**What Hookline does.** `long_videos="windows"` (the default; `--long-videos windows`) with `window_mode="auto"` (the default; `--window-mode auto|fixed`): the window size follows the video length.

| Video | Behaviour |
|---|---|
| shorter than 5 s | `unavailable` (the model's trained minimum; windows under 5 s are not possible) |
| 5 s up to `window_s` (30 s) | one native pass over the full video, exactly as before (same cache key, same artifact) |
| over 30 s, up to 45 s | 20 s windows, stride 10 s |
| over 45 s, up to 90 s | 15 s windows, stride 8 s |
| over 90 s | 10 s windows, stride 5 s (never under 5 s; the out-of-memory fallback stops at `min_window_s`, 10 s) |

The stride keeps the 50 % overlap, and in auto mode the warm-up is at most a quarter of a window (5 s, 3.75 s, 2.5 s for the three window sizes). The last window is moved back so it
ends exactly at the end of the video and is a full window, never a short remainder. `window_mode="fixed"` uses `window_s` for every window (30 s every 15 s by default).
The sizes are a rule of thumb from the model's structure (`ttcc_windows.AUTO_TIERS`); they have **not** been tuned or validated on real reels. Shorter windows give finer local reading and need less memory,
but each sees less context and the start-of-window artifact is a larger share of it. Examples (auto): 31 s → (0,20), (10,30), (11,31); 60 s → 7 windows of 15 s; FL031 (106.6 s, rounds to 107) → 10 s windows every 5 s.
Each window is decoded from its own part of the original file (`video_start`/`video_end`) and prompted as an ad of the window's length, so all windows are inside the model's trained
range. Predictions are put back on the original timeline: window output second *k* of a window starting at *s* is original second *s + k*.

**How windows are merged into one curve.** Each window is a standalone prediction (it behaves as if the ad starts at the window's start, with `R(0) = 1`). Chaining the windows
as they are would repeat the steep "opening drop" of an ad at every boundary, which is an artifact. So the merge works on hazards, `λ(t) = -ln(R(t)/R(t-1))`, the share of the
remaining viewers lost in second *t*:

1. the first window counts fully; a later window's first `warmup_s` seconds (default 5) are down-weighted with a quadratic ramp, because they describe a fresh start, not the middle of a video;
2. the merged hazard at second *t* is the weighted mean over the windows that cover *t*;
3. the curve is rebuilt as `R(t) = exp(-Σλ)`: continuous, non-increasing, anchored at `R(0) = 1`, no jumps at window boundaries.

If every window agrees about a second, the merged curve equals that prediction exactly (tested). `windowing.diagnostics` reports how much overlapping windows disagree (mean and max
hazard difference after warm-up), so the stitching can be judged per video.

**Failures, memory and caching.**
- Each window's raw prediction is cached on its own (`ttcc_windows/<start>-<end>.json`, keyed by video hash, settings, window start and end). A rerun repeats only missing or failed windows; a changed stride reuses the windows that still match.
- After a CUDA out-of-memory error or a `max_length` error the video is replanned with 30 % shorter windows (stride scaled with them) down to `min_window_s` (10 s). `windowing.attempts` records every plan. `--no-shrink-window` turns this off.
  A video that does not fit as a single pass (a 28 s video on a small GPU) falls back to windows the same way.
- Every window's raw output must echo the seconds it decoded (`row.chat_template_kwargs`). An outdated runner that ignores `video_start` is refused (`stage: protocol`, "outdated"), never cached; upload the current `ttcc_infer.py` and `app.py`.
- A window that fails for another reason is listed in `windowing.failed_windows`. If the neighbouring windows overlap it, the curve still covers the whole video; otherwise the curve **ends at the gap** (`partial_coverage`, `coverage.unpredicted_s`), nothing is interpolated.
  Without a successful first window there is no curve, and an earlier successful prediction is never overwritten. The next run retries only the failed windows.
- Cost: the number of windows is about `(n - W) / stride + 1` for window `W` (auto: 60 s: 7 windows of 15 s, 106 s: 21 of 10 s; fixed 30/15: 60 s: 3, 82 s: 5). Smaller windows mean more passes, though each is cheaper. Remote runs send the video once and all windows in one request (one model load); the Space handles up to 16 windows per request and the client splits longer lists.

**Settings:** `--window-mode`, `--window-s`, `--stride-s`, `--warmup-s`, `--min-window-s`, `--no-shrink-window`, and `--long-videos windows|first-60s|skip` (`first-60s` is the older behaviour: only the first 60 s, the rest unpredicted).

**Artifact additions** (`ttcc_retention.json`): `stitched: true`, `horizon_s` = seconds covered, `windowing` (window plan and sizes, per-window tokens/time, failed windows, per-second window coverage, merged hazards,
diagnostics, a `method` and a `limitation` text), `raw_output.windows`; `ttcc_raw.json` holds every window's raw output. The plot is labelled "stitched from overlapping windows (approximate)".

**Limitations (read before using a long-video curve).**
- A stitched curve is an **approximation**. Window *k* predicts viewers who start watching at its start; the merged curve describes viewers who started at 0. Late in a long video the curve is therefore the model's reading of local content,
  not a survival estimate for the whole video. The warm-up weighting reduces start-of-window artifacts; it does not remove the conceptual gap.
- The model was trained on ads of 5 to 60 s; stitched curves beyond 60 s are outside anything it was evaluated on. Accuracy is **unvalidated** (there is no retention ground truth for these reels).
- Like every TTCC curve this is a **model prediction, not audience data**.
- Window-start behaviour on real reels (how large the "fresh start" artifact is, and whether 5 s of warm-up is the right size) has not been measured; the defaults were chosen from the model's structure and tested with synthetic windows.
- The runner's `video_start` windowing relies on upstream's `qwen_omni_utils.fetch_video` honouring `video_start`/`video_end`; this has **not yet been checked with the real model** (the Space was paused and the updated runner not deployed when this was built). Until it is, treat window timestamps as unverified on real inference: run a 60 s clip and compare a window against the same seconds cut out as its own clip.

## Scores per segment

`ttcc_dropoffs.json -> segments` splits the predicted span into consecutive segments (the flagged intervals plus the stretches between them). Each has the TTCC predicted
fall (and its share of the total, and a rank) and, separately, the heuristic `attention_risk` / `moment_quality` (mean and max, 0 to 100) from measured features. The two
kinds of number are never combined into one score. The page shows them in a table and exports them as CSV.

## Drop-off analysis (`ttcc_dropoffs.json`)

After a successful prediction, `run_ttcc.py` (or `scripts/show_dropoffs.py <id> --artifacts <root> --refresh`) derives where the
predicted curve falls, from the files on disk only (no Space, no model, no upload).

- **Detection** (fixed, documented thresholds in `src/hookline/dropoffs.py`, chosen for readability and not validated): every predicted
  second is `steep` (at least 15 % of the remaining viewers lost and at least 0.02 absolute), `decline` (at least 5 % per second),
  `hold` (under 2 %) or unlabelled. Runs become intervals: `steep_drop`, `sustained_decline` (3 s or more) and `holds` (3 s or more),
  each with its share of the curve's total predicted fall.
- **Observed** evidence per interval and per second: Whisper words, PaddleOCR text, scene cuts, motion, quiet audio. A source that was
  not run says `not run`; it is never reported as "nothing happened".
- **Candidate causes** are only the existing rule-based observations (late first word, no early visual change, static/long shots, repeated
  text, quiet stretches) that overlap the interval, labelled "inferred, unvalidated". TTCC does not say why it predicts a fall.
- **No confidence value:** TTCC outputs a point curve with no calibrated uncertainty on your videos.
- The per-video report gets a "Predicted drop-offs" section under the TTCC section. The curve and the heuristic scores are not changed.

## Limitations and open verification

- **Real inference status:** see the latest local verification note at the end of this file.
- The v8 repo has no `processor_config.json`, which upstream's `validate_ckpt.py` lists as required.
- The model was trained on TikTok ads with audio off. The company reels are a different domain, and
  no ground truth exists here to measure accuracy. Do not report accuracy on them.
- Videos shorter than 4.5 s get `unavailable`. Longer videos are predicted in windows (see above); with `--long-videos first-60s` or `skip` the older rules apply
  (only the first 60 s, or no prediction beyond 60 s). A single window near 60 s can exceed `max_length` (upstream drops about 15% of training rows longer than 54 s);
  the window fallback handles that.
- `views` and `follower_band` are not used anywhere in this stage.
