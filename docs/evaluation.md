# Update 2026-10-11 (read this first)

Full TRIBE coverage (run `20261011T031947Z`, 99 regular reels with TRIBE output; 2 reels failed on TRIBE's transcript alignment: FL004, OL009):

| model | n | AUC | 95% CI | perm p |
|---|---|---|---|---|
| tribe_only | 99 | 0.458 | [0.344, 0.522] | 0.74 |
| ttcc_plus_tribe | 99 | 0.493 | [0.372, 0.574] | 0.57 |
| measured_plus_ttcc_plus_tribe | 99 | 0.581 | [0.474, 0.692] | 0.11 |
| measured_plus_ttcc (100 reels) | 100 | 0.617 | [0.491, 0.715] | 0.036 |

TRIBE shows no signal for beating a page's usual views and adds nothing to TTCC or the measured features. It is kept only as a separate, labelled
brain-response view (never combined with TTCC); the page describes the verdict without leading with the AUC. An earlier 39-reel run (0.54) was too small to mean anything.

Other checks, same 100 regular reels:
- Predicted hold (mean of R3, R5, R10) in quartiles: bottom quarter beat their page median 40%, middle half 58%, top quarter 44% (all reels 50%). Not monotone.
- Measured features vs over-performance (Spearman): cuts per second -0.01, motion +0.23 (p=0.02), loudness +0.12, first word -0.05.
- Demucs, first 5 s: voice loudness +0.26 (p=0.01), music loudness -0.16 (p=0.12), music minus voice -0.22 (p=0.03). Four related tests; exploratory, not used.
- |motion percentile - loudness percentile| (a crude audio-visual mismatch): +0.01 (p=0.93).
- TTCC curves across reels: 99% of the between-reel difference is one scale factor; 3 s hold sd 2.8 points (median 19%).

# TTCC vs TRIBE v2: what is verified, what is not, and how to decide

Status date: 2026-10-10. This file separates **measured** facts from **untested** claims.

## What each model is

| | TTCC (`liangyuch/ttcc-sft-qwen25omni-3b-v8`) | TRIBE v2 (`facebook/tribev2`) |
|---|---|---|
| Output | predicted retention curve R(t), 1 s steps, up to 60 s | predicted fMRI brain response, 20,484 cortical vertices per 1 s segment |
| Trained on | real retention curves of TikTok ads | fMRI recordings of people watching/listening |
| Is it a retention model? | **yes, by design** (domain: ads; not validated on these reels) | **no.** Nothing links its output to viewer retention |
| Role in Hookline | step 2 of the pipeline | optional side output; at most a feature source for a model fitted on retention labels |

## Verified on FC002 (the only video with both outputs)

- Both models run end to end on your Hugging Face Spaces (TTCC: L40S, TRIBE: L4). Outputs are valid and reproducible
  (TTCC curve identical across two runs, max difference 0.0).
- **Timestamps align:** TTCC points at t = 0..25 s and TRIBE segments starting at 0..25 s, both on a 1 s grid, 26 each
  (`python scripts/eval_retention.py --check-alignment FC002 --artifacts artifacts/tribe_test`). TRIBE's last segment ends at 26.0 s
  for a 25.005 s video (upstream pads the last segment).
- **Not verified:** TRIBE trains with a 5 s hemodynamic offset (`data.neuro.offset = 5.0`). Whether the row stamped *t* describes
  the stimulus at *t* or earlier is unknown; Hookline does not shift it. Any model that pairs TRIBE rows with TTCC seconds depends on this.
- On this one video the two per-second signals are unrelated beyond chance (after removing the shared time trend, |rho| = 0.52 against
  a chance level of 0.78 for such series). One video is not evidence of anything.

## Labels that exist today

| Source | What it is | Usable now? |
|---|---|---|
| 110 company reels | `views`, `follower_band` only; not retention; excluded from evaluation by your own rule | **no** |
| SnapUGC subset on disk | 12 videos with ECR (share watching at least 5 s, rank-normalised, one number per video) | **no** (12 < 30 minimum, all ECR 0 to 0.2) |
| SnapUGC full | about 9.7k downloadable labelled videos (feasibility report), skewed to low ECR; ECR is comparable to TTCC's R(5), by rank only | needs download and many GPU runs |
| TTCC's own dataset `liangyuch/ttcc-v0_2_0` | per-second retention curves for 61,789 ads, videos embedded (about 935 GB) | would give true curves; split sizes, licence and that the checkpoint never saw these ads are **unverified** |

**Conclusion: there are no retention labels in this project today, so no accuracy comparison has been run.** `scripts/eval_retention.py`
refuses to compare fewer than 30 labelled videos that have all outputs.

## The comparison (built, tested on synthetic data only)

`src/hookline/evaluation.py`, `scripts/eval_retention.py`. Out-of-fold ridge (repeated 5-fold, standardisation fitted inside each fold),
rank metrics (Spearman, Kendall tau-b), paired bootstrap intervals over videos. Configurations:
`constant`, `duration` (confound check), `ttcc_raw`, `ttcc_ridge`, `tribe` (8 summary features), `combined`.
"Does TRIBE add value beyond TTCC" = the interval of `combined - ttcc_ridge`. If it contains 0 the answer is "not shown", never "no effect".
The tests show the harness separates "TRIBE helps" from "TRIBE is noise" on synthetic data, and that a shuffled label scores near 0
(no leakage). That says nothing about the real models.

## Costs and limits (measured unless marked)

| | TTCC | TRIBE v2 |
|---|---|---|
| Time per 25 s video | about 19 s | about 522 s (predict 343 s; first run on that Space, so it may include one-time downloads) |
| GPU price while it runs | L40S $1.80/h, so about $0.01 per video | L4 $0.80/h, so about $0.12 per video (12x TTCC) |
| Smallest GPU that worked | **L4 failed** (out of memory at 14k tokens with sdpa); L40S works | L4 works |
| Duration limit | 5 to 60 s (longer: first 60 s only). Attention memory grows with length squared: a ~60 s video (about 36k tokens) probably exceeds even 48 GB with sdpa (**my estimate, untested**); FlashAttention would be needed | not tested beyond 25 s; time grows with length |
| Disk on restart | 9.4 GB checkpoint re-downloaded on every start | about 20 GB of weights re-downloaded on every start |
| Licence / access | checkpoint licence "other" (commercial terms not checked) | **CC BY-NC 4.0, non-commercial**; needs gated Llama 3.2 access |

Pilot estimate (untested): 150 labelled videos need about 22 GPU-hours of TRIBE (about $18) and about 50 minutes of TTCC (about $2),
plus idle and download time. With n = 150 a Spearman interval is roughly plus or minus 0.15, so only a TRIBE gain larger than about 0.1 to 0.15
could be detected; smaller gains would be invisible.

## Recommendation (based on the evidence above)

1. **Default architecture: TTCC alone for the retention curve, plus Hookline's measured features for the explanations.** It is the only
   component whose output is a retention prediction, it runs in seconds for cents, and it is the one in the end-to-end pipeline.
2. **Keep TRIBE out of the main pipeline.** It is 12x costlier and about 25x slower per video, carries a non-commercial licence, and has no
   evidence of adding retention information. Leave it as an optional research output.
3. **Promote TRIBE only if** the harness shows `combined - ttcc_ridge` with a 95% interval above 0 on at least about 150 held-out labelled
   videos, with the duration baseline controlled, and the timestamp-offset question resolved. Until then, "TRIBE adds value" is a hypothesis.

## Still untested

- Accuracy of TTCC on any of your reels, or of either model against measured retention.
- Whether TRIBE-derived features add anything beyond TTCC (needs the labelled run above).
- TRIBE's timestamp lag, TRIBE on videos beyond 25 s, TTCC on videos beyond about 35 s, flash-attn on the Space.
- Whether TTCC (trained on TikTok ads) transfers to Instagram reels or to Snapchat UGC.
- The licensing position of using either model commercially.


## Views: does a reel beat its page's usual views? (`scripts/eval_views.py`)

Labels come from `01-Hookline/*/metadata.csv` (real Instagram views). **Views are not retention labels and are never used as such.**

- `overperformance = log(views) - log(median views of the follower band)`; band medians over the 100 regular reels (`face/`, `faceless/`).
- Primary binary label: `above_band_median` (overperformance > 0); 50 of 100 regular reels are positive.
- The 10 `outliers/` reels were selected for unusually high views: they never set a median and are never trained on. Each is scored by
  models fitted on the regular reels of *other* creators (3 outlier creators also have a regular reel).

Models (ridge on the continuous label, alpha = 10 fixed, in-fold median imputation and standardisation; out-of-fold scores from 5-fold
cross-validation grouped by creator, repeated 20 times): `constant`, `follower_band` (one-hot), `duration` (log), `measured` (8 opening
features from the first 5 s, fixed in advance: cuts per second, first cut, motion, brightness, loudness, first word, words per second, first
on-screen text), `measured_plus_ttcc` (+ TTCC R(3), R(5), R(10), steepest early drop) and `ttcc_only`. TTCC models run only when at least 30
regular reels have a TTCC hook prediction, and are compared with `measured` on the same reels.
Metrics: AUC (binary), Spearman (continuous), 95% creator-level bootstrap intervals (2,000), paired differences, permutation tests (1,000
shuffles, whole cross-validation re-run each time). Outputs: `01-Hookline/evaluation/<timestamp>/` (see the README there).

### Metric correction (read this first)

The first run (`20261010T172437Z`) pooled all held-out scores before computing AUC. That biases weak and category-only models **below** their true
value, because each fold's model is trained on the other folds (a fold whose reels do well is trained on reels that do worse). A face/faceless
feature whose raw AUC is 0.64 scored 0.47 that way. The metrics are now computed **inside each held-out fold** and averaged over folds and repeats
(`views_eval.foldwise_metrics`), with a regression test. Numbers from `20261010T172437Z` should not be quoted; use the run below or later.

### Second run (2026-10-10, `01-Hookline/evaluation/20261010T180015Z/`), no TTCC or TRIBE predictions yet

| model | AUC | 95% CI | Spearman | permutation p (AUC) |
|---|---|---|---|---|
| constant | 0.500 | – | – | – |
| follower_band | 0.413 | 0.346–0.498 | −0.134 | 0.92 |
| duration | 0.592 | 0.481–0.699 | 0.146 | 0.077 |
| **page_type** (face / faceless, dataset metadata) | **0.643** | **0.551–0.729** | 0.235 | **0.008** |
| measured (8 opening features) | 0.598 | 0.463–0.681 | 0.149 | 0.065 |
| measured + page_type | 0.612 | 0.486–0.706 | 0.163 | 0.048 |
| ttcc_only, measured + ttcc, tribe_only, ttcc + tribe, measured + ttcc + tribe | not run: no TTCC or TRIBE outputs for the reference reels yet | | | |

Paired AUC differences (95% bootstrap interval): `page_type − constant` +0.05 to +0.23 (above 0); `measured − constant` −0.04 to +0.18;
`measured − duration` −0.16 to +0.13; `measured + page_type − measured` −0.02 to +0.07.

Reading it honestly:
- **One result is distinguishable from chance: page type.** Face reels beat their band's median 64 % of the time, faceless 36 % (AUC 0.64, interval above 0.5,
  permutation p = 0.008). That is the dataset's folder label, not something computed from the video; to use it on a new upload it must be measured (a face detector
  on the first seconds) or supplied. It may also reflect how the organisers sampled reels, not something that generalises.
- **The measured opening features are not shown to beat chance** (interval 0.46–0.68, p = 0.065), and not shown to beat duration alone. Adding page type to them
  gives no clear gain either (difference −0.02 to +0.07).
- `follower_band` stays below 0.5 by construction (the label is centred on each band's median, so band cannot predict it and the held-out fold bias still shows); its
  p = 0.92 confirms no signal. It is a leakage check.
- Outliers (scored by models that never saw their creator): `measured` ranks them low (median rank 67 of 110, AUC 0.375 against regular reels); `page_type` and
  `duration` put them near the middle (median ranks 32 and 34). Ten reels: a case study, not a result.
- 100 reels detect only a fairly strong effect.

### Exploratory: a person on screen in the first 3 s (video-computed stand-in for face / faceless)

A new upload does not come with the dataset's page type, so a video-computed proxy was tried: SigLIP's existing per-frame zero-shot scores for "a person in close-up" and
"a person speaking to the camera" over the first 3 s (`person_only`, `measured_plus_person`; run `20261010T182305Z`). It matches the folder label with AUC 0.75 (72 % accuracy
at the best threshold, in-sample), and as a predictor of beating the band median it reaches AUC 0.58 (interval 0.51–0.70, permutation p = 0.11): suggestive, not shown, and weaker
than the folder label itself (0.64). This group was added *after* the first two runs, so treat it as a hypothesis to confirm on new reels, not as a result.

### Third run: with TTCC hook predictions for all 110 reels (2026-10-11, `01-Hookline/evaluation/20261010T192956Z/`)

TTCC hook predictions (10 s opening, native single pass on the L4 Space) exist for all 110 reels (4 VP9 reels needed a one-time H.264 conversion; the runner's video reader failed on VP9).

| model | AUC | 95% CI | Spearman | permutation p (AUC) |
|---|---|---|---|---|
| measured (8 opening features) | 0.598 | 0.463–0.681 | 0.149 | 0.065 |
| ttcc_only (R3, R5, R10, steepest early drop) | 0.526 | 0.409–0.627 | 0.148 | 0.35 |
| measured + ttcc | 0.617 | 0.491–0.715 | 0.215 | 0.036 |

Paired: `measured + ttcc − measured` AUC −0.04 to +0.10, Spearman −0.01 to +0.20 (no demonstrated gain). Pre-specified directional check (predicted hold should relate positively to
overperformance; Spearman over 100 regular reels, 95% creator bootstrap, one-sided permutation p): R(3) rho +0.13 [−0.06, +0.32] p = 0.09; R(5) +0.13 p = 0.10; R(10) +0.10 p = 0.17.
TTCC's R(3) varies little across these reels (0.12 to 0.26, median 0.19).

Honest reading: TTCC's predicted early hold has the sign we would hope for but is **not shown** to relate to how far a reel beats its page's usual views, and adding it to the measured features does
not demonstrably improve prediction. This is the expected outcome (retention is one ingredient of reach; views also depend on distribution) and is why Hookline presents TTCC as a hook / drop
diagnostic and does not claim to predict virality. Outliers: `measured + ttcc` ranks the 10 extreme viral reels at median 60.5 of 110 (AUC 0.46 vs regular reels), `ttcc_only` 61.5 (0.52): a content model
does not find breakout reels.

### Scores for a video (`scripts/score_video.py`)

`hook_scores.json` per video: a **hook score** (percentile of TTCC's predicted early hold among the reference reels' predictions; needs TTCC on the reference set),
a **breakout score** (percentile of the predicted overperformance among the reference reels' held-out predictions) and the predicted retention at 1/3/5/10 s. Each carries
the measured accuracy of the model behind it (AUC, interval, permutation p, and a plain verdict). The model follows a fixed rule: `measured_plus_ttcc` when TTCC exists
for the video and for 30 or more reference reels, else `measured_plus_page_type` when a page type is given, else `measured`. These are positions, not verdicts; the page labels
the breakout score "experimental" while its interval includes chance.

### TRIBE (not yet run)

`eval_views.py --tribe-artifacts artifacts/tribe` adds eight summaries of TRIBE's predicted brain response over the first 10 s as a feature group and runs `tribe_only`,
`ttcc_plus_tribe` and `measured_plus_ttcc_plus_tribe`, each compared with its smaller sibling on the same reels (e.g. `ttcc_plus_tribe − ttcc_only` on the reels that have both).
TRIBE predicts fMRI-like responses, not retention; whether it helps is what this tests. Not run: only 1 reel has TRIBE output.
