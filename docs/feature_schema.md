# Feature schema (Phase 1, Tasks 2 and 3)

Task 2: low-level visual and audio features (sections up to "Caching").
Task 3: timestamped speech and on-screen text, a separate stage on top of
Task 2's cached outputs (see [Speech and on-screen text](#speech-and-on-screen-text-task-3)).

Low-level, timestamped, descriptive measurements. None of these values is an
engagement or retention score, and none should be read as "more is better":
more cuts, more motion, louder audio or higher sharpness are not treated as
evidence of a stronger hook. A missing modality is never negative evidence.

Versions: `schema_version` 1, `preprocess_version` 1, `feature_version` 2
(see `src/hookline/config.py`). Bump `feature_version` whenever extraction
code changes behaviour, so cached features are rebuilt.

## Timeline

All times are **seconds from the start of the original's playback**
(`original_pts - format_start_time`).

- The proxy is constant-frame-rate (`proxy_fps`, default 10) and keeps the
  original timeline: proxy frame time == timeline time. Temporal resolution is
  `1 / proxy_fps` (0.1 s). Each proxy frame shows the source frame on screen at
  that instant.
- WAV sample `n` is at `timestamp_map.audio.offset_seconds + n / 16000`.
- If a stream starts late (e.g. audio starting 1 s in), windows before it are
  `absent` for that modality, not silent.
- `report.json -> preprocess.timestamp_map` records expected stream spans,
  source fps, a variable-frame-rate flag, the proxy frame count and the maximum
  deviation of proxy frames from the constant-rate grid.

## Segments

- 0.5 s grid (`window_seconds`) starting at 0; detected scene cuts become
  boundaries so a segment never spans a cut where practical.
- Pieces shorter than `min_segment_seconds` (0.2 s) are merged into a
  neighbour across grid boundaries; cuts closer than 0.2 s keep the first
  boundary, and both cut times are listed in that segment.
- Segments are contiguous, cover `[0, duration]`, and are never empty. The
  first and last segment may be shorter or longer than 0.5 s (`is_partial`).
- A clip shorter than one window gives exactly one segment.

## Modality status (per segment and per stream)

| status | meaning |
| --- | --- |
| `available` | measured with adequate coverage |
| `absent` | the stream does not exist, or the window lies outside the stream's span |
| `failed` | the stream exists and data was expected here, but decoding/extraction produced nothing |
| `uncertain` | inspection failed (presence unknown), or the window is only partly covered (< 50 %) |
| `not_applicable` | used per feature in `notes`, e.g. motion for a window holding only the first frame |

An inspection failure (corrupt/unreadable file) gives `uncertain`, never `absent`.
Values are empty (CSV) / `null` (JSON) unless measured. They are never filled with 0.

## `segments.csv` columns

| column | unit / range | notes |
| --- | --- | --- |
| `video_id` | | manifest `video_id`, else file stem |
| `segment_index` | int | 0-based |
| `start_s`, `end_s`, `duration_s` | s | timeline |
| `is_partial` | bool | duration differs from `window_seconds` |
| `scene_index` | int | increments at each detected cut; empty without video |
| `scene_cut_at_start` | bool | empty for the first segment (`not_applicable`) and without video |
| `scene_cut_count` | int | cut times within the segment (includes merged close cuts) |
| `visual_status` | status | |
| `visual_coverage` | 0-1 | proxy frames found / frames expected |
| `visual_frames` | int | proxy frames in the segment |
| `brightness_mean` | 0-255 | mean luma |
| `contrast_std` | 0-255 | luma standard deviation |
| `sharpness_lapvar` | ≥ 0 | variance of the Laplacian at proxy resolution; inflated by noise |
| `noise_sigma` | 0-255 scale | Immerkær noise estimate; fine texture and small text also raise it |
| `motion_mean`, `motion_max` | 0-255 | mean absolute luma difference between consecutive (blurred) frames; cut frames excluded; conflates object/camera motion, lighting change and noise |
| `scene_change_max` | 0-1 | max frame-to-frame dissimilarity (max of colour-histogram and structural change) |
| `visual_warnings` | `\|`-separated | `dark`, `overexposed`, `low_contrast`, `low_sharpness`, `possible_noise`, `partial_visual_coverage` |
| `audio_status` | status | |
| `audio_coverage` | 0-1 | decoded samples / samples expected |
| `rms` | 0-1 | linear RMS of the mono 16 kHz signal |
| `rms_dbfs`, `peak_dbfs` | dBFS | floored at -100 (digital silence) |
| `near_silence_ratio` | 0-1 | share of 20 ms frames below -50 dBFS |
| `energy_variation_db` | dB | std of 20 ms frame levels |
| `onset_strength_mean` | dB | mean spectral flux (mean positive per-bin dBFS increase per 10 ms hop); an activity proxy |
| `onset_count` | int | flux peaks with non-falling energy (syllable/note/beat-onset proxy, not a beat tracker) |
| `clipping_ratio` | 0-1 | share of samples ≥ 0.999 full scale (after down-mix/resample, so approximate) |
| `audio_warnings` | `\|`-separated | `digital_silence`, `possible_clipping`, `partial_audio_coverage` |
| `notes` | text | per-feature status, e.g. `scene_cut_at_start=not_applicable: first segment` |

## Scene cuts

A proxy frame is a cut if either:

- the HSV colour-histogram Bhattacharyya distance to the previous frame is ≥ 0.4, or
- the structural change (1 − max(NCC, 0) of 64-px grayscale thumbnails) is
  ≥ 0.2 **and** exceeds the median of the ±5 surrounding frames by ≥ 0.15
  (so pans and noise are not cuts, but text cards on one background are).

Runs of adjacent candidate frames (dip-to-black, flash or whip transitions)
count as one cut at the frame with the largest change. Slow dissolves are not
detected. Cut times have 0.1 s resolution. Talking-head jump cuts and punch-in
zooms are detected as cuts (they are edits).

## `frames.csv` (per proxy frame)

`time_s, brightness, contrast, sharpness, noise, motion, hist_change, struct_change, is_cut`.
The first frame has empty `motion`, `hist_change` and `struct_change`.

## `report.json`

`schema_version`, `generated_at`, `video_id`, `source_path`, `source_sha256`,
`processing_status` (`ok` | `partial` | `failed`), `duration_s`,
`modalities` (stream status per modality, plus `not_extracted_in_this_phase`:
speech, on-screen text, semantic visual; speech and text are written by the
Task 3 stage to their own files, so this report and its cache stay unchanged), `probe` (FFprobe facts),
`preprocess` (cache key, proxy/WAV paths, `timestamp_map`), `extractor`
(Hookline/preprocess/feature versions, `feature_key`, full config, `dependencies` =
OpenCV/NumPy/FFmpeg versions that computed the features, `preprocess_ffmpeg` = the FFmpeg that
made the proxy/WAV (`null` for caches older than this field), `runtime` = Python version and
executable), `scene_cut_times`, `warnings`,
`errors`, `opening` (below), `segments` (same fields as the CSV plus a
`feature_status` map), `outputs`.

Processing status: `ok` when every stream that exists was extracted with no
failed windows; `partial` when something was extracted but a stream or some
windows failed; `failed` when nothing usable was extracted (including
unreadable files).

## Opening summary (`report.json -> opening`, `first5s_summary.csv`)

Covers `[0, min(5 s, duration)]`. For clips shorter than 5 s only the real
duration is summarised (`shorter_than_opening_window`).

- `valid_windows`: segments overlapping the opening with status `available`, per modality.
- `coverage`: share of the opening's seconds in each status, per modality.
- `visual`: frames, mean brightness/contrast/sharpness/noise/motion, number and times of cuts, first cut.
- `audio`: level, peak, near-silence ratio, onset count and rate, first non-silent time (first 20 ms frame ≥ -50 dBFS), clipping, digital silence.

These describe the opening; they are not engagement predictions.

## Run-level files (`artifacts/features/`)

Rebuilt after every run from **all** per-video folders in the artifacts root, sorted by
`video_id`. Re-running one video therefore updates only that video's rows; other videos stay,
unchanged and without duplicates. A per-video file whose header differs from the current schema
is left out, not padded with blanks. Its video id is listed in `excluded_stale_schema` and in
the log; re-run that video.

- `run_summary.json`: `counts` and `videos` cover every video in the root. A video's record is
  replaced when it's re-run, and rebuilt from `report.json` (`record_source`) if no earlier
  summary had it. `last_run` holds this invocation's inputs, config, video ids and counts (the
  exit code uses these). Also holds `feature_environment` and `excluded_stale_schema`.
- `segments_all.csv`: every video's `segments.csv` rows.
- `first5s_summary.csv`: one row per video with a `report.json`.
- `feature_env.json`: the OpenCV/NumPy/FFmpeg versions, Python version and executable that built
  this root's features (see [Caching](#caching)).

CSV is the only tabular format written (no Parquet dependency); load with
`pandas.read_csv(..., keep_default_na=True)` so empty cells become NaN.

## Caching

`artifacts/cache/<video_id>/<key16>/` holds `proxy.mp4`, `audio.wav` and
`preprocess.json`. The key hashes the source bytes plus the preprocessing
fields of the config (`proxy_fps`, `proxy_short_side`, `audio_sample_rate`,
`preprocess_version`). A cached proxy/WAV is also rebuilt when the FFmpeg that made it
(`preprocess.json -> ffmpeg_version`) differs from the installed one. Caches written before this
field existed are reused and reported as `preprocess_ffmpeg: null`.

Features are recomputed, from the cached proxy/WAV, when any config field, `feature_version`, or
the OpenCV, NumPy or FFmpeg version changes (`extractor.feature_key` hashes the preprocessing
key, the config and those versions). Changing the source file changes the key; originals are
never written.

Environment rules:
- Task 2 features are computed only by `scripts/run_pipeline.py` in the feature environment
  (`.venv`).
- It exits with code 3 when run with this project's `.venv-models` (whose PaddleOCR pins an older
  OpenCV/NumPy), or when the root's `feature_env.json` names different library versions. In the
  second case it prints the recorded interpreter; `--new-feature-env` deliberately rebuilds the
  root's features in the current environment.
- `run_speech_ocr.py` never computes Task 2 features. It reuses a video's `report.json` only if
  the source hash, config, feature/preprocess versions match and dependency provenance is
  present. Otherwise that video is `failed` with the exact `run_pipeline.py` command to run.

---

## Speech and on-screen text (Task 3)

`scripts/run_speech_ocr.py` (`hookline.multimodal`) reads each video's cached
Task 2 report (which must exist: build it first with `.venv/bin/python scripts/run_pipeline.py`)
and never re-runs or changes
preprocessing or visual/audio extraction. Speech uses the cached mono 16 kHz
WAV; OCR samples frames from the original. Versions: `speech_version` 1,
`ocr_version` 2 (`src/hookline/config.py`). The same five statuses apply.

### Models

| stage | backend | default model | where it is cached |
| --- | --- | --- | --- |
| speech | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 build of OpenAI Whisper) | `small`, `int8` on CPU | Hugging Face cache (`~/.cache/huggingface`, repo `Systran/faster-whisper-small`) |
| text | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 3.x | `PP-OCRv5_mobile_det` + `en_PP-OCRv5_mobile_rec` | `~/.paddlex/official_models/` (or `$PADDLE_PDX_CACHE_HOME`) |

Install `requirements-models.txt` into a separate venv (`.venv-models`):
PaddleOCR pins `opencv-contrib-python==4.10.0.84` and `numpy<2.4`, which
conflict with the main environment. Models are only downloaded when
`--download-models` is passed; otherwise a missing model is a `failed` stage
with the reason recorded. Silero VAD (bundled with faster-whisper) drops
non-speech before decoding. Document unwarping, orientation and text-line
orientation models are off. The English recogniser reads Latin script only;
use `--ocr-lang hi` (Devanagari), `--ocr-lang ch`, etc. One recogniser runs per
pass, so mixed-script captions need a second pass with another language.

### Timeline

- **Speech**: Whisper times are relative to the WAV's first sample, shifted by
  `timestamp_map.audio.offset_seconds`, clamped to `[0, duration_s]` and
  rounded to 1 ms. Word times come from Whisper's cross-attention alignment,
  not forced alignment, so treat them as approximate (not frame-accurate).
- **Text**: FFmpeg's `fps` filter resamples the original (including
  variable-frame-rate sources) to a constant `sample_fps` grid (default 2,
  i.e. every 0.5 s) on the project timeline, the same convention as the proxy.
  Frame times are read back with FFprobe and stored. Sample `t` stands for
  `[t, t + 1/sample_fps)`. Text shown for less than one interval can be missed,
  and the real onset of an event lies in `(first_seen_s − interval, first_seen_s]`.
  Frames are scaled so the short side is at most `frame_short_side` (720,
  never upscaled), in display orientation, and are deleted after OCR.

### File-level status (`speech.json`, `ocr.json` → `status`)

| status | speech | on-screen text |
| --- | --- | --- |
| `available` | transcribed; or the audio is below -50 dBFS throughout (model skipped, `method: skipped_silent_audio`) | OCR ran on at least one sampled frame |
| `absent` | the file has no audio stream | the file has no video stream |
| `failed` | audio exists but is undecodable, cached WAV missing, model unavailable, or transcription raised | video undecodable, frame sampling failed, model unavailable, or OCR raised on every frame |
| `uncertain` | stream presence unknown (inspection failed), or every transcribed segment is flagged | stream presence unknown |

`speech_detected` / `text_detected` are `true`/`false` only when the stage
measured the file, and `null` otherwise. A failure, a missing stream or a
skipped stage is never evidence that there is no speech or text. `measured`
is `true` when values may be read (`available`, or speech `uncertain` because
of low confidence).

### `speech.json`

`schema_version`, `speech_version`, `generated_at`, `video_id`, `source_path`,
`source_sha256`, `cache_key`, `status`, `measured`, `speech_detected`,
`speech_result`, `flags` (file level), `method` (`asr` | `skipped_silent_audio`), `reason`, `language` and
`language_probability` (detected per video unless `--language` is given; the
task is always `transcribe`, never translate), `word_timestamps`, `model`
(backend, version, model, revision, compute type, device), `config`,
`timeline` (`offset_seconds`, `duration_s`), `timing` (`audio_seconds`,
`processing_seconds`, `realtime_factor`), `stats` (`segments`, `words`,
`speech_seconds`), `warnings`, `errors`, and `segments[]`:

| field | unit | notes |
| --- | --- | --- |
| `id`, `start_s`, `end_s`, `text` | s | Whisper segment on the project timeline |
| `avg_logprob`, `no_speech_prob`, `compression_ratio` | | Whisper's own confidence signals |
| `flags` | list | `low_confidence` (avg_logprob < −1.0); `possible_non_speech` (low_confidence and no_speech_prob > 0.6); `high_no_speech_prob` (no_speech_prob > 0.6 although avg_logprob ≥ −1.0); `low_word_probability` (mean word probability < 0.5); `repetitive` (compression_ratio > 2.4). These are typical of hallucinations on music or noise. Flagged segments and their words are kept, never deleted |
| `words[]` | | `start_s`, `end_s`, `word`, `probability` (0–1); `null` if the backend gave no word timestamps |

`speech_result` (speech version 2) separates the outcomes:

| `speech_result` | meaning |
| --- | --- |
| `speech` | at least one unflagged segment (`speech_detected: true`) |
| `no_speech` | measured, nothing transcribed, or audio below −50 dBFS throughout (`speech_detected: false`) |
| `uncertain_speech` | text was transcribed but every segment is flagged (`status: uncertain`, `speech_detected: null`); words kept |
| `failed` | audio exists but couldn't be measured (WAV, model or transcription failure); never evidence of no speech |
| `absent` | no audio stream |
| `unknown` | stream presence unknown (inspection failed) |

File-level `flags`: `low_language_confidence` when the language was auto-detected with
probability < 0.5 and something was transcribed (the text may be in the wrong script, or
hallucinated). It adds a warning but doesn't change the status by itself. Use `--language` to
force a language. All thresholds are in `SpeechConfig` and part of the cache key.

Whisper detects one language per file, so code-switched speech (e.g.
Hindi–English) may come out in one script. Lyrics in music can be transcribed
as speech; check `flags` and `no_speech_prob`.

### `ocr.json`

`schema_version`, `ocr_version`, `generated_at`, `video_id`, `source_path`,
`source_sha256`, `cache_key`, `status`, `measured`, `text_detected`, `reason`,
`model` (backend and Paddle versions, detection/recognition model names,
device), `config`, `sampling` (`fps`, `interval_s`, `frame_size`, `range_s`,
`frames`, `failed_frames`, `first_frame_s`, `max_deviation_from_grid_s`,
`source_fps`, `source_variable_frame_rate`), `timing` (`processing_seconds`,
`seconds_per_frame`), `warnings`, `errors`, `events[]` and `frames[]`.

`events[]`: one entry per distinct piece of text on screen, not per frame.

| field | unit | notes |
| --- | --- | --- |
| `event_id` | int | 0-based |
| `text` | | most frequent recognised variant |
| `norm_text` | | NFKC-normalised, whitespace collapsed, case-folded |
| `first_seen_s`, `last_seen_s` | s | first and last sample showing the text |
| `visible_until_s` | s | `last_seen_s + interval_s`, capped at the end of the OCR range |
| `frames` | int | samples in which it was detected |
| `score_mean` | 0–1 | mean recognition confidence |
| `box_norm` | 0–1 | median `[x0, y0, x1, y1]` as fractions of frame width/height |
| `repeat_of` | int | earlier event with the same text that had disappeared, else `null` |

Linking rule: a detection joins an active event when the normalised text
similarity is ≥ 0.8 (absorbs OCR jitter such as `50%`/`5O%`) and either the
event was seen in the previous sample (text may move with the camera) or the
box IoU is ≥ 0.3 (needed to bridge one missed sample). Text that disappears
for longer and returns is a new event with `repeat_of`. The same caption
shown twice at once (in different places) is two events. Detections with
score < 0.5, or with no letter or digit (`-`, `***`: edges and textures), are
dropped.

On real reels, burned-in captions are long, stable events (high `frames`,
`score_mean` ≈ 0.99). Scene text such as signs and logos, and misreads,
mostly show up as single-sample events (`frames` = 1). They are kept, not
filtered, so consumers can decide.

`frames[]`: `t_s`, `status` (`available` or `failed` with `error`), and
`detections[]` with `score`, `box_norm`, `poly_px` (polygon in sampled-frame
pixels), `event_id`, and `text` only when it differs from the event's text.
An `available` frame with no detections is a measured "no text"; a `failed`
frame is not.

### Evidence timeline: `segments_merged.csv` (per video) and `segments_merged_all.csv` (run)

The creator-facing timeline. One row per Task 2 segment (the 0.5 s grid split
at scene cuts), in playback order. It holds every `segments.csv` column
(times, `scene_index`, `scene_cut_at_start`, visual and audio values and
statuses, unchanged), then the speech columns, the word links, the text
columns, and a combined `notes`. Join keys are `video_id` + `segment_index`.
`segments_merged_all.csv` is rebuilt from every video's `segments_merged.csv` in the artifacts
root (see [Run-level files](#run-level-files-artifactsfeatures)). Videos whose merged file
predates the current columns are listed in `speech_ocr_run_summary.json ->
excluded_stale_schema` until they're re-run.

Values are empty (CSV) / `null` unless the window status is `available` or
`uncertain`. They are never 0 for "not measured".

How to read the evidence in each row:

| evidence | how it shows | meaning |
| --- | --- | --- |
| observed | `*_status = available`; text also has `text_evidence = observed` | measured inside this segment. `false` / 0 here are measured negatives *at the sampling rate used* |
| inferred / uncertain | `*_status = uncertain`; text has `text_evidence = inferred` when no OCR sample lies in the segment | values carried from the preceding OCR sample (`[t, t + interval)`), a low-confidence transcription, a partly covered window, or a skipped stage (`notes`) |
| absent / not applicable | `absent` (no stream, or outside the stream's span); `not_applicable` per feature in `notes` | nothing to measure. This isn't negative evidence |
| failed | `failed` | the stream exists but extraction failed here. The values are empty and are never read as "no speech" or "no text" |

| column | unit / range | notes |
| --- | --- | --- |
| `video_id` ... `audio_warnings` | | every `segments.csv` column, see above |
| `speech_status` | status | the window's `audio_status` if it is `absent`/`failed`; `uncertain` if partly covered or low-confidence; the file status if speech was not measured |
| `speech_present` | bool | any word (or segment, without word timestamps) overlaps the window |
| `speech_coverage` | 0–1 | share of the window covered by word spans (0 is a measured zero) |
| `speech_word_count` | int | words whose midpoint falls in the window (each word counted once); empty without word timestamps |
| `speech_word_prob_mean` | 0–1 | mean word probability of those words |
| `speech_text` | | those words joined |
| `speech_word_index_first`, `speech_word_index_last` | int | range of `word_index` in `words.csv` for this segment's words. Empty if none or if speech isn't measured here. Exact word times live only in `words.csv` / `speech.json` |
| `text_status` | status | `absent` without video; `failed` if every sample in the window failed; `uncertain` if no sample lies in the window (inferred), the window is partly sampled (< 50 %), or it's beyond `--ocr-max-seconds` |
| `text_evidence` | `observed` / `inferred` | `observed`: values come from OCR samples inside the segment. `inferred`: from the preceding sample's hold interval. Empty when text isn't measured |
| `text_samples` | int | successful OCR samples inside the segment (0 for inferred rows) |
| `text_present` | bool | any text event visible in the window |
| `text_coverage` | 0–1 | share of the window backed by successfully OCR'd samples |
| `text_event_count` | int | events visible in the window |
| `text_new_count` | int | events first appearing in the window, excluding repeats |
| `text_area_ratio` | 0–1 | max over samples of the summed detection-box area (fraction of frame) |
| `on_screen_text` | | visible events' text, `" \| "`-separated, in order of appearance |
| `text_event_ids` | `\|`-separated ints | `event_id`s of those events, for joining with `text_observations.csv` and `ocr.json -> events` (boxes, first/last seen) |
| `librosa_*` | | per-segment librosa values, see [librosa audio features](#librosa-audio-features-librosajson) |
| `siglip_*` | | per-segment SigLIP 2 values (opt-in), see [SigLIP 2](#siglip-2-visual-semantic-signals-siglipjson) |
| `recommendation_ids` | `\|`-separated ids | recommendations in `recommendations.json` whose interval overlaps the segment |
| `notes` | | segment notes plus e.g. `speech=not extracted (stage skipped)`, `text=inferred between samples (no OCR sample in segment)`, `text=not sampled (beyond OCR range)` |

When a stage is skipped (`--skip-speech` / `--skip-ocr`), its columns are
`uncertain` with a note, never `absent`. Brief text between two OCR samples
can be missed entirely. At 1 fps, 1 of 18 persistent text episodes was
missed in the 3-reel pilot (`scripts/compare_ocr_runs.py`).

### `words.csv` (per video)

One row per transcribed word with timestamps. It's empty when speech isn't
measured.

| column | unit | notes |
| --- | --- | --- |
| `video_id` | | |
| `word_index` | int | 0-based, in transcript order; referenced by `speech_word_index_first/last` |
| `segment_index` | int | segment containing the word's midpoint (words past the last segment's end go to the last segment) |
| `start_s`, `end_s` | s | word times on the playback timeline, as stored in `speech.json` (Whisper alignment: approximate, not frame-accurate) |
| `word`, `probability` | –, 0–1 | |
| `asr_segment_id`, `asr_flags` | | Whisper segment and its flags (`low_confidence`, `possible_non_speech`, `repetitive`) |

A word keeps its row even if its segment's audio window is `failed`. That
segment's speech columns are still empty.

### `text_observations.csv` (per video)

One row per OCR detection, plus one row per sample with no detections or a
failed sample. It's empty when OCR didn't run.

| column | unit | notes |
| --- | --- | --- |
| `video_id`, `t_s` | –, s | sample time on the playback timeline; the sample stands for `[t_s, t_s + interval_s)` |
| `segment_index` | int | segment containing `t_s` |
| `sample_status` | status | `available` or `failed` (a failed sample is not an empty one) |
| `event_id` | int | links to `ocr.json -> events` and `text_event_ids`; empty for a sample with no detections |
| `text`, `score` | –, 0–1 | text as read in this sample |
| `x0`, `y0`, `x1`, `y1` | 0–1 | box, fraction of frame width/height, display orientation |
| `error` | | error for a failed sample |

### Regenerating the timeline

`segments_merged.csv`, `words.csv` and `text_observations.csv` are rebuilt
from `report.json`, `speech.json` and `ocr.json` on every run. When the
cache keys match, no model is loaded. Use the same flags as the original run.
For the company pilot (1 fps, first 10 s):

```bash
# one reel
.venv-models/bin/python scripts/run_speech_ocr.py --input 01-Hookline/face/videos/FC003.mp4 \
    --artifacts artifacts/company --ocr-fps 1 --ocr-max-seconds 10
# a batch (segments_merged_all.csv then holds exactly these reels)
.venv-models/bin/python scripts/run_speech_ocr.py --input 01-Hookline/face/videos/FC003.mp4 \
    --input 01-Hookline/faceless/videos/FL004.mp4 --input 01-Hookline/faceless/videos/FL019.mp4 \
    --artifacts artifacts/company --ocr-fps 1 --ocr-max-seconds 10
```

The log line shows `cached speech,ocr` when nothing was re-inferred. Python
access: `hookline.multimodal.process_video_multimodal` (one video) and
`run_multimodal` (batch, one video in memory at a time).

### Librosa audio features (`librosa.json`)

Computed by `run_speech_ocr.py` (skip with `--skip-librosa`) from the cached mono 16 kHz WAV,
alongside, not instead of, the NumPy audio features in `segments.csv`. Needs `librosa`
(`requirements-models.txt`). The source video is never re-decoded.

- **Frames:** centred STFT frames, `n_fft` 512 (32 ms), hop 160 (10 ms). Frame `i` is at
  `timestamp_map.audio.offset_seconds + i × 0.01` s on the playback timeline. A frame belongs to the
  segment that contains its centre (the last segment also takes a frame exactly at its end). Frames
  centred after the video's duration (AAC padding) belong to no segment. Near a boundary, a centred
  frame still hears up to 16 ms of the neighbouring segment.
- **Segment status:** `librosa_status` follows the segment's `audio_status` (same WAV, same
  coverage rule). If the stage itself didn't measure the file, it takes the file status:
  `absent` without audio, `uncertain` if inspection failed, `failed` if the WAV is missing or
  unreadable, librosa isn't installed, or extraction raised. It's `uncertain` with a note when the
  stage was skipped.

| column (`segments_merged.csv`) | unit | aggregation over the segment's frames |
| --- | --- | --- |
| `librosa_frames` | count | frames centred in the segment |
| `librosa_rms_db` | dBFS | mean of per-frame RMS level (floored at −100) |
| `librosa_low_energy_ratio` | 0–1 | share of frames below −50 dBFS (`silence_dbfs`) |
| `librosa_centroid_hz` | Hz | mean spectral centroid of frames at or above −50 dBFS |
| `librosa_bandwidth_hz` | Hz | mean spectral bandwidth of those frames |
| `librosa_rolloff_hz` | Hz | mean 85 % spectral rolloff of those frames (bin resolution 31.25 Hz) |
| `librosa_zcr` | 0–1 | mean zero-crossing rate per sample of those frames |
| `librosa_onset_strength` | relative | mean librosa onset strength (64-band log-mel spectral flux, all frames). Comparable within a video, not across loudness-normalised and non-normalised sources |

Spectral shape and zero-crossing rate are left empty (not 0) when every frame in the segment is
below −50 dBFS: on near-silence they describe noise.

`librosa.json` also holds:
- `low_energy_intervals`: `{interval_id, start_s, end_s}` runs of frames below −50 dBFS lasting at
  least 0.3 s.
- `global`: `audio_seconds`, `low_energy_ratio`, `rms_db_mean`, `centroid_hz_mean`, and the tempo
  fields:
  - `tempo_status` is `available` only when the audio is at least 6 s long, at least 20 % of it
    is above −50 dBFS, at least 8 beats were tracked, and at least 80 % of tracked beats fall within
    50 ms of a detected onset (`beat_onset_agreement`). The beat tracker always returns evenly
    spaced beats, even for random onsets, so beat regularity alone isn't evidence.
  - Otherwise `tempo_status` is `not_applicable` (too short or too quiet) or `uncertain` with
    `tempo_bpm` null.
- `frames` (count, hop, first/last time), `versions` (librosa, numpy), `config`, `cache_key`.

Cache key: preprocessing key (source bytes + WAV settings) + audio offset + `LibrosaConfig` +
`librosa_feature_version` + installed librosa and numpy versions + segment boundaries.

Limitations: the WAV is a 16 kHz mono down-mix, so nothing above 8 kHz and no stereo cues. A
bright or loud window isn't "exciting" and a quiet one isn't "boring". Tempo is one global value,
not a beat grid, and isn't reported for speech-only or short clips.

### SigLIP 2 visual-semantic signals (`siglip.json`)

Opt-in with `--siglip` (`run_speech_ocr.py`, `.venv-models`). Without the flag, nothing is
loaded or written, and the `siglip_*` columns are `uncertain`/empty with note
`siglip=not run (disabled)`. These are pretrained-model measurements, used as-is with no
training. They're kept apart from the recommendations, except that one rule
(`no_early_visual_change`) and the visual-change score point read `siglip_within_shot_change`.

- **Model:** `google/siglip2-base-patch16-224`, pinned to revision
  `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, loaded through `transformers` (`SiglipModel`) in
  eval mode with no gradients, float32.
  - It's public, so no Hugging Face token is needed.
  - The first run needs `--download-models`: about 1.43 GiB into `~/.cache/huggingface/hub`
    (`model.safetensors` 1.50 GB + tokenizer 34 MB). Without that flag it only reads the local cache.
  - Device: `--siglip-device auto` uses MPS when PyTorch supports it, else CPU. An unsupported MPS
    operation moves the model to the CPU once (recorded in `model.mps_fallback`). The model is
    loaded once per run, not per frame or per video.
- **Frames:** the cached Task 2 proxy (10 fps, short side 360). The source video isn't read.
  - Sampling: every `round(10 / --siglip-fps)`-th proxy frame (default 2 fps), plus the frame
    nearest the midpoint of any segment the grid misses, so every segment that has proxy frames
    gets at least one.
  - `frame_index` is the row of `frames.csv` (proxy frame) and `t_s` its `time_s` on the original
    playback timeline.
  - The processor resizes frames to 224 × 224 without cropping, so portrait frames are squashed.
- **Per frame (`siglip.json -> frames`):**
  - `similarity_to_previous`: cosine with the previous sampled frame.
  - `change` = 1 − similarity (0 = same embedding; about 0.01–0.03 between near-identical
    frames).
  - `crosses_cut`: a detected scene cut lies between the two frames.
  - `scores`: one per description, `sigmoid(cos(image, text) × exp(logit_scale) + logit_bias)` with
    the model's own scale and bias. The prompt is `"This is a photo of {description}."` with text
    padded to 64 tokens, as in SigLIP's zero-shot pipeline. Scores are independent (they don't sum
    to 1), are often small in absolute terms (bias ≈ −16.8), and are similarity scores, not
    calibrated probabilities or ground truth. Compare them across frames and segments of a video
    rather than reading them as "x % likely".
  - Embeddings are in `siglip_embeddings.npy`: float16 (n_frames × 768), L2-normalised, in frame
    order. `embedding.sha256` identifies the file. Like all of `artifacts/`, it's git-ignored.
- **Descriptions** (keys are column suffixes): `talking_to_camera` "a person speaking directly to
  the camera", `product_demo` "a product demonstration", `person_closeup` "a close-up of a person",
  `montage` "a montage of rapidly changing scenes", `text_screen` "a screen full of text",
  `static_shot` "a static shot", `outdoor_scene` "an outdoor scene". A single frame can't show
  editing rhythm or camera motion, so `montage` and `static_shot` only measure how much one frame
  resembles such imagery. Use the cut and motion features for the real thing.

| column (`segments_merged.csv`) | unit | meaning |
| --- | --- | --- |
| `siglip_status` | status | the segment's `visual_status` when measured; `uncertain` if no sampled frame lies in it; else the file status |
| `siglip_frames` | count | sampled frames in the segment |
| `siglip_frame_ids` | `\|`-separated ints | their `frame_index` (rows of `frames.csv`, entries in `siglip.json -> frames`) |
| `siglip_within_shot_change` | 0–2 | largest `change` into a frame of this segment from the previous sampled frame, **excluding** pairs across a detected cut: content change the cut detector doesn't already mark |
| `siglip_cut_change` | 0–2 | largest `change` across a detected cut ending in this segment: how different the new shot is |
| `siglip_top_label` | key | description with the highest mean score in the segment (relative, not a classification) |
| `siglip_score_<key>` | 0–1 | mean score of each description over the segment's sampled frames |

Semantic change isn't attention, and repeated imagery isn't boredom.

- **Status:**
  - `absent`: no video stream.
  - `uncertain`: inspection failed.
  - `failed`: proxy missing, no frames listed, nothing decodable, torch/transformers missing,
    model not cached, or inference/validation failed.
  - Validation rejects embeddings that aren't (n × d), aren't finite, or aren't unit-norm, as well
    as image and text sizes that differ, and non-finite scale or bias.
- **Cache key:** source SHA-256 + preprocessing key (proxy settings) + `SiglipConfig` (model id,
  revision, sample fps, coverage rule, prompt template; not device or batch size) +
  `siglip_version` + sampled frame times + torch/transformers/numpy versions. A cache hit loads no
  model. A failed run never replaces a usable `siglip.json` / `siglip_embeddings.npy`.
- **Cost on the M2 (8 s clip, 16 frames):**
  - First run: 254 s, almost all of it the download.
  - Warm run: model load 2.7 s, inference 0.035 s/frame on MPS, about 7 s end to end, peak RSS
    1.2–1.3 GB.
  - Cache hit: 0.2 s.
- **Known warning:** transformers prints `bos_token_id must be None or an integer within the
  vocabulary ... got 49406` (and the same for eos). It comes from the checkpoint's minimal
  `config.json` and doesn't affect pooled embeddings. Scores were checked against
  `SiglipModel.forward` logits (maximum difference 6e-9), and MPS against CPU embeddings (1e-6).

### Recommendations (`recommendations.json`)

Each item has `id` (`r*`), `source` (always `rule`), `basis` (always `measured`), `rule`,
`start_s`/`end_s`, `suggestion`, `why`, `evidence` (a list of `{type, id}`) and `caveat`.
They are transparent rule-based observations over measured features: no language model or
external API is involved. Evidence types and where to find them:

- `segment`: `segment_index`
- `word`: `words.csv` `word_index`
- `text_event`: `ocr.json` / `text_observations.csv` `event_id`
- `librosa_interval`: `librosa.json` `interval_id`

`evidence_problems` lists any id that doesn't exist. It's checked on every run and should be empty.

Rules (heuristic thresholds, not validated against retention data). Each fires only when the
evidence it needs was measured, never on missing, failed or uncertain modalities:

| rule | fires when |
| --- | --- |
| `late_first_word` | first transcribed word starts at 2 s or later, with speech measured (`available`) before it |
| `opening_without_words` | speech measured with no word, and every OCR sample in the first 3 s succeeded and read nothing |
| `quiet_stretch` | a librosa low-energy interval of at least 1.5 s, audio `available`, no transcribed word overlaps it, not the clip's final 0.25 s |
| `static_stretch` | at least 3 s in one scene with `motion_mean` < 1.0 in every segment (about the lowest 7 % of company-reel segments) |
| `no_early_visual_change` | clip of at least 6 s, vision `available` for every segment of the first 3 s, no scene cut after 0 s and no SigLIP `siglip_within_shot_change` of 0.15 or more in that window (SigLIP is optional: without it the observation says so) |
| `long_shot` | a scene (between detected cuts) of at least 10 s with vision `available` in every segment |
| `repeated_text` | an OCR text event that reappears after disappearing (`repeat_of` points to the earlier event); cites both events |

### Segment scores (`scores.json`, `segment_scores.csv`, run-level `segment_scores_all.csv`)

Two separate 0–100 heuristics per segment, rebuilt on every run from the merged timeline and the
rule-based recommendations (scoring version 2: measured evidence only):
- `moment_quality`: evidence that the stretch works.
- `attention_risk`: evidence that a viewer might lose interest.

A segment can be high on both, or low on both. 0 means no evidence either way, not "bad".

| `attention_risk` points | | `moment_quality` points | |
| --- | --- | --- | --- |
| rule recommendation (measured) | 15 each | transcribed speech | 5 |
| motion_mean < 1.0 | 5 | on-screen text observed | 5 |
| near-silent + no speech (measured) | 5 | scene cut or SigLIP within-shot change ≥ 0.15 | 5 |

- Each total is capped at 100.
- Every point is listed in `risk_contributions` / `quality_contributions` (`source`: rule /
  measured, `reason`, `points`, `ref`).
- A recommendation overlaps a segment when its interval intersects it (end-exclusive).
- Missing, failed or uncertain modalities contribute nothing.
- `basis` is always `measured_only`. Scores written by scoring version 1 (which could include
  language-model judgments) are not comparable; re-run to rebuild them.
- `highest_attention_risk` / `highest_moment_quality` rank spans: adjacent segments with the same
  top reason, scored by their peak.
- The weights are hand-set, not calibrated or trained on outcomes. These aren't retention,
  engagement or virality predictions.

### Opening summary (`opening_multimodal.json`, `first5s_multimodal.csv`)

Covers `[0, min(5 s, duration)]` and combines Task 2's opening values with
speech and text:

- `visual`, `audio`: stream status, per-status coverage, scene cuts, first
  cut, brightness/motion, first non-silent time, near-silence ratio, level,
  onsets (copied from `report.json -> opening`).
- `speech`: `status`, `status_coverage`, `language`, `speech_detected`,
  `first_word_s`, `word_count`, `words_per_s` (words / opening seconds),
  `speech_coverage`, `transcript`.
- `text`: `status`, `status_coverage`, `text_detected`, `first_text_s`,
  `events`, `new_events`, `single_sample_events` (events seen in only one
  sample, often scene text or misreads), `screen_coverage` (share of the
  opening with any text visible), `texts` (up to 10 distinct texts, most
  persistent in the opening first).
- `librosa` (JSON only): `status`, low-energy intervals starting in the opening, `tempo_bpm`,
  `tempo_status` (whole-clip tempo).
- `stage_status`, `features_status`, `note`.

`speech_detected` / `text_detected` is `false` only if every opening window
was measured as `available`; otherwise it is `null` when nothing was found.
The CSV joins lists with `|`. Descriptive only; not an engagement or
retention prediction.

### Stage status (`stage_status`, `speech_ocr_run_summary.json`)

Requested modalities are speech, OCR, librosa (unless `--skip-librosa`) and SigLIP (with `--siglip`).
TRIBE v2 is not part of the stage status:
- `ok`: every requested modality is `available` or `absent` and no window `failed`.
- `failed`: every requested modality is `failed` or `uncertain` (for example, models not
  installed, or an unreadable file).
- `partial`: anything else. For example, an uncertain or partly failed modality makes a video
  `partial`.

`run_speech_ocr.py` exits with 1 if any video failed. The run summary also records
`tribe_requests_sent` (videos sent to the TRIBE endpoint in that run).

### Caching

- Speech key: Task 2 preprocessing key (source bytes + preprocessing
  settings) + every `SpeechConfig` field except `device` + `speech_version`.
- OCR key: source SHA-256 + every `OcrConfig` field except `device` +
  `ocr_version`.
- An artifact is reused when its `cache_key` matches and its status is not
  `failed`. Failed artifacts are always retried, so installing a model later
  just works. `--force` recomputes speech and OCR only (features stay cached).
- Model names are in the key; model weights are not hashed. Pin
  `model_revision` in `SpeechConfig`, or use `--force`, if a hub model changes.
- Keys contain no paths, so `speech.json`/`ocr.json` computed on another
  machine (e.g. a Kaggle GPU) are reused when copied into the same
  `artifacts/features/<video_id>/` and the same flags are used locally
  (device is not part of the key; compute type is).

## TTCC predicted retention (`run_ttcc.py`, opt-in)

These are not features. They are a pretrained model's prediction, kept apart from every measured
column and every heuristic score. Full documentation is in [ttcc.md](ttcc.md).

- `ttcc_retention.json`:
  - `timestamps_s`: whole seconds 0..T, where T = round(duration_s) and 5 ≤ T ≤ 60.
  - `retention`: R(t), the predicted fraction of viewers (0–1) still watching at second t.
    R(0) = 1 by convention. The values are non-increasing.
  - The record also holds `convention`, `summary`, `inference` (settings, token count, attention
    anchor), `provenance` (checkpoint revision and files, go_viral commit and `register.py` hash,
    library versions, verified retention head, GPU) and `raw_output` (path and SHA-256 of
    `ttcc_raw.json`).
- The 1 s grid is TTCC's own. It is not resampled onto the 0.5 s segments, and no TTCC column is
  added to `segments_merged.csv`.
- `ttcc_status.json`: `status` is `success`, `failed` or `unavailable`, plus `stage`, `reason`,
  `cache_key` and `prediction_is_current`. Without a successful run there is no curve; nothing is
  filled in.

