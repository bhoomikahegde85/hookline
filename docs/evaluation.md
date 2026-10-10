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

### First run (2026-10-10, `01-Hookline/evaluation/20261010T172437Z/`), no TTCC predictions yet

| model | AUC | 95% CI | Spearman | 95% CI | permutation p (AUC) |
|---|---|---|---|---|---|
| constant | 0.500 | – | – | – | – |
| follower_band | 0.259 | 0.163–0.366 | −0.432 | −0.610 – −0.233 | 0.93 |
| duration | 0.563 | 0.447–0.675 | 0.095 | −0.105 – 0.280 | 0.046 |
| measured | 0.578 | 0.461–0.691 | 0.118 | −0.089 – 0.308 | 0.071 |
| measured_plus_ttcc / ttcc_only | not run: no TTCC hook predictions (the Space has no credits) | | | | |

Reading it honestly:
- **No model is shown to beat chance.** `measured` has AUC 0.578, but its interval includes 0.5 and p = 0.07; `measured − duration` is
  −0.13 to +0.17. `duration` has p = 0.046 on the permutation test while its bootstrap interval includes 0.5: borderline at best.
- `follower_band` is far *below* 0.5 by construction, not by information: the label is centred on each band's median, so a band-mean model
  fitted without the held-out reel is pushed the wrong way (leaving a positive out lowers its band's training mean). Its permutation
  p = 0.93 confirms there is no signal. It is a leakage check: band alone cannot predict a band-relative label.
- Outliers: `measured` ranks them low (median rank 67 of 110, AUC 0.375 against regular reels); `duration` puts them higher (median rank 34,
  AUC 0.63). Ten reels: a case study, not a result.
- 100 reels can detect only a fairly strong effect; an AUC near 0.6 would need more data to confirm.
