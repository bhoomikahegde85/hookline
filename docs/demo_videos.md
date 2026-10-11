# Demo videos: what to show, and what it proves

## What the graph can and cannot prove

- **Can show:** TTCC reads the picture. All-black video gets one fixed curve (3 s hold 17.0%) whatever the reel; real reels move away from it, and
  the same reel re-encoded barely moves (noise about 0.2 to 0.6 points of R(3)). FL031: 25.9% real vs 17.0% black vs 25.7% re-encoded.
- **Cannot show:** that people will watch a video fully, or that a video will go viral. No retention labels exist here, and on 100 reels with real
  views the predicted hold did not track views (bottom quarter beat their page's median 40% of the time, middle half 58%, top quarter 44%).
- **Be upfront:** TTCC gives most reels a similar decay (3 s hold: median 19%, 10th to 90th percentile 15% to 23%). The page therefore plots each curve
  against the reference band and the all-black baseline.

## Live demo A: "the model reads content" (about 10 minutes, controlled)

Use the two 10 s clips in `01-Hookline/evaluation/sanity_checks/20261010T183353Z/variants/` (the opening of FL031; short, so each run is quick):

1. Upload `FL031__reencoded.mp4` (the real picture). Predicted 3 s hold about 26%, above the grey band.
2. Upload `FL031__black.mp4` (same 10 s, every frame black). The curve falls onto the red dotted baseline (about 17% at 3 s). Only the picture changed.
3. Optional: `FL031__shuffled.mp4` (the same 10 s in a different chunk order) lands near the real one, because the picture content is the same.

These are company-reel files: only upload them if the organisers allow it, and note the upload page sends them to Google too.

Say: "Only the picture changed and the prediction fell to the no-content baseline, so the curve is responding to content, not to a template."

## Live demo B: "edits change the prediction" (what-if)

Upload any reel, press **Accept all suggested trims**, then **Predict the edited version**. The page re-predicts the edited cut with the same model.
Say: "this is the model's prediction of the edited cut, not proof it retains better."

## Reels that tell an honest story (from the 105 reels with TTCC predictions and real views)

Predicted hold = mean of R(3), R(5), R(10). Over-performance = log(views) minus log(median views of that follower band), over 100 regular reels.

| Reel | Predicted hold | Views | vs page's usual | Reading |
|---|---|---|---|---|
| FL031 | highest of all (18.4%) | 321,812 | +0.84 (2.3x) | prediction and views agree |
| FL025 | 3rd (17.4%) | 554,417 | +1.38 | agree |
| FL030 | 8th (16.2%) | 4,596,228 | +3.50 | agree |
| OL005 (outlier) | 6th (16.3%) | 10,300,207 | +5.56 | agree |
| FC028 | near bottom (10.2%) | 52,603 | -1.16 | agree |
| FC037 | near bottom (8.8%) | 8,708 | -1.52 | agree |
| FL041 | 2nd (18.1%) | 65,361 | -0.76 | disagree: high predicted hold, low views |
| FC034 | near bottom (9.4%) | 1,157,278 | +1.94 | disagree: low predicted hold, high views |
| FC043 | 14th percentile | 591,832 | +2.70 (15x) | disagree: shown as a weak hook, did very well |

Show the agreeing ones if you like, and say plainly that the disagreeing ones exist and that overall the relation is weak: choosing only agreeing
examples would mislead. The 10 viral outliers rank 6th to 89th of 105 by predicted hold, median about 56th, i.e. in the middle.

## What would make the claim testable

Real per-second retention curves from creators' analytics for a few hundred videos: calibrate or fine-tune TTCC on them, then compare predicted and real
curves on held-out videos. This is the next step; it is not done here.
