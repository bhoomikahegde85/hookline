# Gemini: explanations and edit suggestions for the predicted drop-off segments

**Division of labour.** TTCC *predicts* a retention curve. Gemini *interprets the video content* around the segments TTCC flags and proposes
timestamped edits, citing measured evidence. Gemini does not predict retention and does not know why real viewers leave. Every observation-based
explanation is phrased as a hypothesis, and **nothing is validated against real audience-retention data**.

## Privacy and cost: read first

- **The video is uploaded to Google** (the Files API) for each new analysis, and deleted from Google right after the call.
- Nothing is sent unless you pass `--gemini-allow-remote`. Company reels under `01-Hookline/` also need `--gemini-approve <video_id>`,
  and only after the organisers allow uploading them.
- The key comes from `GEMINI_API_KEY` (environment or the untracked `.env`) and is never printed or stored in any artifact.
  **Use a new key.** A key that was ever committed (this project's earlier key was) must be revoked and never reused.
- Cost scales with the video seconds analysed, `--gemini-fps` (default 1) and `--gemini-resolution` (default medium). **A cap keeps long videos affordable**:
  at most `--max-analysed-s` seconds (default 120) of frames are analysed. A video whose predicted span fits is analysed whole. A longer one is analysed as
  clips: the opening (`--hook-s`, default 5 s) plus every flagged segment with `--context-margin-s` (default 3 s) around it, largest predicted fall first, adding a
  segment only if the total stays within the cap. Segments outside the clips are **not interpreted** (listed in `analysis_scope.not_analysed`, never invented);
  their predicted retention is still given as context. The whole file is uploaded to Google for the call (deleted afterwards); only the clips are given to the model as frames.
  The artifact records `analysis_scope` (clips, seconds, approximate frames) and the real token usage. An identical result (same video, settings and evidence) is reused: no call, no cost.
  I have **not** measured cost or latency of this version; check the usage tokens in `gemini_edit.json` after the first call.
- Default model: `gemini-3.8-flash` (what this project used before). If your key serves another Flash model, set `GEMINI_MODEL` or `--gemini-model`.
  The model name has not been re-verified against the API.

## Setup and run

```bash
uv pip install --python .venv-models/bin/python -r requirements-models.txt     # includes google-genai
# put GEMINI_API_KEY=<new key> in .env (untracked)
# with the whole pipeline (optional step before the review page):
python3 scripts/run_e2e.py <video> --allow-remote --gemini --gemini-allow-remote [--approve <id>] --open
# or on an analysed video:
.venv-models/bin/python scripts/run_gemini_edit.py <id> --artifacts artifacts/e2e --video <file> --gemini-allow-remote [--gemini-approve <id>]
```

## What Gemini receives and must return

An **evidence pack** with citable ids, restricted to the analysed clips (the whole predicted span for a video within the cap): TTCC's curve and segments (with predicted fall, rank, heuristic
risk/quality), Whisper words (`w<n>`), OCR text (`t<n>`), scene cuts (`c<n>`), quiet audio (`q<n>`), rule-based observations (`r<n>`),
segments (`s<n>`), and `ttcc:t=<second>`. It returns JSON per TTCC segment: observations (each `observation` must cite evidence,
an `interpretation` may not), a hedged hypothesis with a confidence (how well the evidence supports it, not a retention probability),
and up to 3 suggestions. Each suggestion has a category (`tighten_hook`, `remove_dead_space`, `adjust_pacing`, `change_cuts`, `add_caption`,
`fix_audio`, `other`), an action, an edit (`trim` a range, `add` at a time, or `review` a range), a rationale citing evidence, a hedged
expected effect and a confidence.

For a multi-clip request each clip is preceded by a label ("CLIP 2 of 3: original video seconds 97 to 109") and the prompt requires **original** timestamps
(never time inside a clip). TTCC's whole-video retention trend is in the pack, so the model can prioritise by predicted fall even where it sees no frames.
TTCC stays responsible for the retention prediction; Gemini only reads content and proposes edits.

## Validation (strict; nothing is repaired)

Dropped and counted in `validation`: unknown or duplicate segments, invented evidence ids, observations without evidence, suggestions
without cited evidence, timestamps outside the predicted span or the analysed clips, entries for segments that were not analysed, trims shorter than 0.2 s or longer than 30 s, edits that do not overlap
their segment, unknown categories, and suggestions beyond the cap. The artifact status is `available`, `uncertain` (some items dropped) or
`failed`. A failed call never replaces an earlier usable result, and never touches any other artifact.

## Outputs

- `gemini_edit.json`: status, reason, model, usage tokens, validation, evidence digest, the validated analysis.
- `segment_recommendations.json`: per segment, TTCC's predicted fall (a model prediction), the heuristic scores, the measured evidence, the rule proposals and Gemini's
  observations, hypothesis and suggestions, side by side with a `source` on every proposal. Nothing is merged into one score.
- `review.html` (the editing assistant): Gemini's observations and hypotheses per segment, source-labelled suggestions (trims can be previewed and accepted
  like the rule-based ones), and an overview box. The CSV/JSON export carries `source`, `rationale`, `confidence` and `evidence_refs`.

## Not verified

- The clip selection and the multi-clip request were tested with a fake SDK. Whether Gemini maps clip time back to original seconds correctly is **unverified** (the prompt asks for it and
  validation drops edits outside the clips, but a consistent offset error inside a clip would not be caught).

- No real call has been made with this version (no new key was available when it was built). The SDK glue is tested against a fake SDK; the
  prompt, the model name, cost, latency and the quality of Gemini's explanations are untested.
- Whether Gemini's suggestions improve retention: not tested. Treat them as hypotheses to try and compare against real audience data.
- Video understanding of Hindi speech and on-screen text depends on the model; the evidence pack's transcript and OCR are rough on some reels.
