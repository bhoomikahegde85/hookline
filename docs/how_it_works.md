# Hookline: how a video moves through the system, and how we know it works

Hookline takes one video and shows a creator, before posting, how attention is predicted to behave second by second: how strong the hook is and why,
where attention holds or drops, the strongest and weakest moments, and what to change and where. It judges the **video**, not the account: no
follower count, page or past views of the video feeds any score. Everything below is a **model prediction or an unvalidated rule of thumb, not audience
data**, and nothing here says a video will or will not go viral.

## Architecture (input to output)

```
video (upload page, http://127.0.0.1:8770, or scripts/run_e2e.py)
  |
  |-- 1. Measured features (local, OpenCV / FFmpeg / librosa): cuts, motion, brightness, loudness, quiet stretches, speech timing (Whisper),
  |      on-screen text and its timing (PaddleOCR), SigLIP 2 scene semantics (person on screen, product demo, text screen ...)
  |-- 2. TTCC (pretrained retention model, Qwen2.5-Omni-3B + hazard head, on our Hugging Face Space)
  |      native 10 s windows every 5 s over the whole video, merged in hazard space into one continuous predicted retention curve R(t);
  |      the first window, a native prediction for viewers who start at 0, is the "hook" (R at 1, 3, 5, 10 s)
  |-- 3. Gemini (required): reads the flagged clips together with the transcript, OCR, audio and TTCC evidence and returns timestamped, evidence-cited
  |      observations and edit suggestions, validated strictly (invented ids, out-of-range timestamps and unsupported claims are dropped)
  |-- 4. Music vs voice (Demucs htdemucs, local): music and voice levels over the first 5 s; descriptive only
  |-- 5. TRIBE v2 (separate model, on our Space): predicted brain response over the first 10 s, used ONLY for a separate experimental popularity score;
  |      never combined with TTCC
  |-- 6. Reference set: 100 real reels with real views; the video is placed among them (percentiles), never scored on account data
  |-- Segments: the predicted span is split evenly by duration (about 4 s each, at most 12 segments), each typed from its own predicted fall
  v
Outputs (one review page): hook verdict + why | scores 0-100 with the accuracy of their model | predicted retention curve and drop segments |
strongest / weakest moments | per-segment evidence, rule and Gemini suggestions with trims you can preview and accept | "what if I make these edits?"
(re-predicts the edited cut) | exports (JSON, CSV)
```

## What is analysed, and why
- **Seen**: scene cuts and pacing, motion, brightness, whether a person is on screen in the first 3 s (SigLIP), the frames themselves (TTCC and Gemini both see
  frames at 1 fps).
- **Heard**: loudness, quiet stretches (librosa), and speech timing and words (Whisper, including non-English).
- **Said / shown as text**: transcript, on-screen text and when it first appears (PaddleOCR).
- **Over time**: every signal is on one timeline, so a drop in the predicted curve is shown next to what was on screen, said and heard at that moment.
- **Why these**: the brief says impact comes from what is seen, heard and said together; each is measured by a dedicated tool, and TTCC / Gemini give a
  model-level reading of the same frames. Using separate measured evidence keeps every suggestion citable.

## The method, and the alternatives we rejected
- **TTCC** is the only available pretrained model that predicts a retention curve (trained on TikTok ads). We call it as published (native protocol), and we
  use it only for what it supports: windows of 5 to 60 s, here 10 s. Longer videos are stitched from native windows and labelled approximate.
- **Hazard-space merging** (not chaining conditional survival) because each window is a standalone prediction that would otherwise repeat the "opening drop"
  at every boundary. Tested: windows that agree reproduce the underlying curve exactly; a start-of-window artifact is suppressed.
- **Hook strength** compares the video's own measured opening and TTCC prediction with 100 reference reels (percentiles) and says why with hedged rules of thumb.
- **Moments** are ranked by predicted hazard in 3 s windows after the opening (a relative measure within the video).
- **Gemini explains, TTCC predicts**: Gemini never predicts retention; its output is validated against the evidence pack.
- **Rejected**: a viral verdict (the data cannot support it, see below), mixing page size into any score, and presenting TTCC's curve as audience data.

## How we know it works (and where it does not)
Evidence, all reproducible under `01-Hookline/evaluation/` (see its README):
1. **TTCC is not a template.** On real reels: black frames moved R(3) about 12x the re-encoding noise, different reels about 20x, shuffled 1-second chunks about 2x, and telling
   the model the ad is 45 or 60 s long instead of 10 s changed nothing meaningful. It responds to the visual content, mostly the overall look rather than the order of events.
   Caveat: curves are close to one shared decay (99% of the between-reel difference is one scale factor; 3 s hold median 19%, 10th to 90th percentile 15% to 23%; an all-black video gets 17%),
   so the page plots each curve against the reference band and the all-black baseline.
2. **Views-based evaluation** on 100 real reels (label: log views relative to the follower band's median; grouped cross-validation by creator; AUC, Spearman, bootstrap
   intervals, permutation tests). Result (details in `docs/evaluation.md`): measured opening features AUC 0.60 [0.46, 0.68]; TTCC hook values alone 0.53 [0.41, 0.63]; both together 0.62 [0.49, 0.72] (permutation p = 0.036 but no demonstrated gain over measured alone);
   the predicted early hold has the expected sign (rho +0.13) but is not significant. So **no content-only model is shown to predict beating a page's usual views**, and the page says so next to the score. The 10 extreme viral reels are scored separately by models that never saw their creator.
   **TRIBE v2 popularity score (separate, experimental), 99 reels with TRIBE output:** AUC 0.46 [0.34, 0.52]: no signal found. TTCC plus TRIBE 0.49; measured plus TTCC plus TRIBE 0.58 (no gain over 0.62 without TRIBE).
   **Hook verdict vs views:** reels in the bottom quarter of predicted hold beat their page's median views 40% of the time, the middle half 58%, the top quarter 44%: the verdict describes the predicted curve and is not a grade.
   **Music and voice (Demucs, exploratory):** voice loudness vs over-performance rho +0.26 (p=0.01, about 0.04 after correcting for four tests), music loudness -0.16 (p=0.12); not used in any score.
3. **No retention or watch-time labels exist**, so we never use views as retention. Views are used only to ask "does the video's own content predict beating its page's usual reach?".
4. **Recommendations** are timestamped and tied to cited evidence; accepted trims can be previewed, and the "what if" feature re-predicts the edited cut
   with the same model and states whether the change is beyond the noise floor (about 0.003 in R(3)).

## Cases considered
See `01-Hookline/evaluation/edge_cases/` (generated by `scripts/edge_cases.py`): no audio stream, silent audio, text-only slideshows, static image, dark low-quality
footage, 0.4 s clip, audio-only file, truncated and empty files, landscape, a 75 s clip (many windows), a video-only clip. Also: non-English speech, videos under the model's 5 s minimum,
GPU out-of-memory (the runner shrinks the window), Space not reachable (clear error, earlier results kept), company reels (explicit approval before any upload).

## Limits (said plainly)
- TTCC was trained on TikTok ads with audio off and is not validated on these reels; stitched curves beyond 10 s are approximate.
- 100 reels can detect only a fairly strong effect; the content-based breakout score is not shown to beat chance and is labelled experimental.
- Views reflect distribution, timing and trends that are not in the video.
- Hook "reasons" are rules of thumb, not fitted or validated.

## Where we would take it with more time and data
- Collect reels with known outcomes (the brief says this is part of the task) using the provided Instagram scrapers: watch time and retention curves would let us fit and validate
  the hook and drop predictions directly, and fine-tune TTCC on this domain.
- Feed audio to the retention model (TTCC was trained with audio off). TRIBE v2 was evaluated on 99 reels and showed no signal for views; real retention curves are the missing label.
- Trending-sound detection needs platform data (trend lists, sound usage counts) that we do not have; Gemini only describes the audio and whether it fits the picture, as an unvalidated hypothesis.
- Calibrate the hook rules from outcome data instead of rules of thumb; A/B-test suggested edits.
- Per-niche reference sets (meme, quote, creator) instead of one pool.
