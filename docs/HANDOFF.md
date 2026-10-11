# Hookline handoff (state as of 2026-10-11 ~01:15 local)

Paste this file into a new chat to continue. Repo: /Users/bhoomikarhegde/Documents/hackathon/hookline. Nothing is committed (do not `git add .`; 01-Hookline/ is
company data and gitignored). Use `.venv/bin/python` (features, tests), `.venv-models/bin/python` (Whisper/OCR/SigLIP/Gemini SDK). zsh does not word-split
variables: use `bash -c` with arrays when building long `--input` lists.

## Goal
Hackathon Problem 01 "Hookline": take a video, predict second-by-second viewer response, say how strong the hook is and why, where attention holds/drops,
strongest/weakest moments, what to change and where; judge the VIDEO not the account; show how we know it works (real outcomes), edge cases, future scope.
Judges upload unseen videos. User wants to win; time left ~3-4 h from 01:15.

## What exists (all tested, 546 tests pass, ruff clean except old E741 in scripts/inspect_environment.py)
- Upload page: `python3 scripts/serve_app.py --open` (127.0.0.1:8770; tick "send to my Space" (required) and Gemini (also required)). Runs `scripts/run_e2e.py`
  with `--hook-s 10 --full-curve`. Result page = review.html (src/hookline/review_page.py; dark theme, WLDD colours).
- Pipeline steps (run_e2e): features (OpenCV/FFmpeg) -> speech+OCR+librosa+SigLIP (local) -> TTCC on Space -> [Gemini] -> scores (scripts/score_video.py)
  -> explain (scripts/explain_video.py: hook_report.json + moments.json) -> review page.
- TTCC: src/hookline/ttcc.py (+ttcc_windows.py). Default policy "hook" (one native pass over first min(dur,30) s); upload app uses fixed 10 s windows every 5 s merged in
  hazard space (approximate, window 0 = native hook, `hook.native=true`). 20 s OOMs on the 24 GB L4, 10 s fits. Non-h264/hevc/mpeg4 (VP9) are converted to
  540 px H.264 first. TTCC unavailable now fails the e2e step with the reason.
- New modules: hook_report.py, moments.py, whatif.py (apply accepted trims, re-predict, compare; endpoints in app_server.py `/api/whatif/<id>`), hook_score.py
  (0-100 hook/breakout percentiles + measured accuracy; NO account features), views_eval.py (evaluation), app_server.py, gemini_edit.py (Gemini explainer,
  max_analysed_s=120 clip budget, strict validation).
- Evaluation: `.venv/bin/python scripts/eval_views.py [--ttcc-artifacts artifacts/hook] [--tribe-artifacts artifacts/tribe]` writes timestamped runs to
  01-Hookline/evaluation/ (README there). Label = log(views) - log(median views of follower band over 100 regular reels); above-median = binary label; fold-wise
  AUC/Spearman, creator-grouped CV, bootstrap CI, permutation tests; 10 outliers scored separately. Sanity check: scripts/ttcc_sanity.py. Edge cases: scripts/edge_cases.py.
- Docs: docs/how_it_works.md (pitch: architecture, method, evidence, limits, future scope), docs/evaluation.md, docs/ttcc.md, docs/gemini.md, README (upload page section).
- Memory notes: ~/.claude/projects/-Users-bhoomikarhegde-Documents-hackathon-hookline/memory/ (hookline-hackathon-state.md etc.).

## Evidence so far (be honest in the pitch)
- TTCC sanity (3 reels, 10 s): black frames move R(3) ~12x re-encode noise, different reels ~20x, shuffled chunks ~2x, told length (45/60 s) ~0 => reads visual look, not a template.
- Views eval (100 regular reels, run 20261010T192956Z): measured 0.598 [0.463,0.681]; ttcc_only 0.526 [0.409,0.627] p 0.35; measured+ttcc 0.617 [0.491,0.715] p 0.036 (no demonstrated gain
  over measured, diff -0.04..+0.10); duration 0.592; page_type (face/faceless folder = ACCOUNT metadata, not used in product) 0.643 [0.551,0.729] p 0.008; person-on-screen (SigLIP) 0.583 (exploratory).
  TTCC R3 vs overperformance rho +0.13 [-0.06,+0.32] p 0.09. Outliers (10 viral) rank median ~60/110. Conclusion: NO content-only model is shown to predict beating a page's usual views.
  Position Hookline as a hook/edit diagnostic with honest accuracy, not a viral predictor; breakout score is labelled experimental with its AUC/CI shown.
- Edge cases: 13 inputs; 9 full analyses; 4 stop clearly (empty, corrupt, audio-only, 0.4 s).
- Earlier "pooled OOF AUC" numbers (run 20261010T172437Z) were biased and withdrawn.

## Infra / costs (HF org wlddhq, private Spaces; creds in untracked .env: TTCC_ENDPOINT_URL, TRIBE_ENDPOINT_URL, HF_TOKEN read-only, GEMINI_API_KEY)
- TTCC Space `wlddhq/hooklinewldd-ttcc` on 1xL4 ($0.80/h), sleep after 30 min idle (wake ~4 min). All 110 reels have hook predictions in artifacts/hook/features.
- TRIBE Space `wlddhq/hooklinewldd` on L4, sleep 30 min. Running now: `scripts/run_tribe_batch.py` on a balanced 40-reel subset (FC001..FC020 + FL001..FL020 interleaved), 10 s clips, ~2-3 min each,
  18/40 done at 01:17 (log: artifacts/tribe_run.log; outputs artifacts/tribe/features). When done: rerun eval_views.py with --ttcc-artifacts artifacts/hook --tribe-artifacts artifacts/tribe
  (needs >=30 reels with TRIBE; underpowered, expect "not shown"). TRIBE licence CC BY-NC: confirm hackathon use with organisers.
- Write token (handler-token) was used then should be logged out: `.venv-models/bin/hf auth logout` (cached login currently still works for restart/hardware calls; log out when done).
- Gemini key verified working (gemini-3.8-flash, text + video). User must confirm it is the NEW key (old leaked one must stay revoked). Never print secrets.
- Approvals given by the user: TTCC on all 110 company reels (own private Space), TRIBE on 40 reels, sanity on FC002/FL031/FL002. No other company-reel uploads (e.g. to Gemini) without fresh approval.

## Remaining / next steps
1. Wait for TRIBE batch; run eval_views.py again; update docs/evaluation.md + docs/how_it_works.md with the TRIBE numbers; pause/let TRIBE Space sleep.
2. Demo prep: warm the TTCC Space (restart if asleep), run one real non-company video through `scripts/serve_app.py` (~4-5 min per minute of video, local speech/OCR ~3 min); rehearse 2-3 outlier reels; ask whether any
   public reel may be sent to Gemini.
3. Optional: speed up local steps (skip SigLIP/Whisper medium), face detector for page type is NOT used in product (account data).
4. Pitch slides: architecture, why TTCC/Gemini/hazard merge, evidence table above, edge cases table (01-Hookline/evaluation/edge_cases/*/edge_cases.md), limits, future scope (collect reels with known outcomes via the provided scrapers, fine-tune on retention, audio-aware retention model, calibrate hook rules, A/B test edits).
5. Before finishing: pause Spaces (they auto-sleep), `hf auth logout`, remind user to revoke old Gemini key, no commits unless asked.
