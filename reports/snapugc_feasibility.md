# SnapUGC feasibility assessment

Checked 2026-10-09 against the live sources. Every number below was observed
directly; inferences are marked as such.

Sources:
- Code: https://github.com/dasongli1/SnapUGC_Engagement (last commit 4e0ce31, 2025-05-29)
- Paper: Li et al., "Delving Deep into Engagement Prediction of Short Videos", ECCV 2024
- Dataset mirror: https://huggingface.co/datasets/tcm03/SnapUGC (last modified 2025-03-27)

## Bottom line

- **What the label means.** ECR is "the proportion of viewers who watched the
  video for at least 5 seconds" (paper, Sec. 3.3). It is **one number per video**.
  It is not a retention curve and not second-by-second data. It does, however,
  measure exactly what Hookline's opening-hook analysis is about.
- **The pretrained model is impractical for us.** Inference needs about 13 GB of
  checkpoints, it calls `.cuda()` throughout, and it pins a 2022-era CUDA 11.3
  stack. It cannot run on an Apple M2.
- **Labelled videos are skewed.** Only about 9.7k of the 106k labelled training
  videos are downloadable, and almost all of them have low ECR.
- **Recommendation.** Don't train an engagement model this weekend. Use the small
  subset to test the pipeline and as qualitative examples. Phrase Hookline's
  output as heuristic hook diagnostics, not predicted retention.

## 1. Can the repository be cloned?

Yes. A shallow clone is 52 MB. It contains:
- `ECR_inference/`: the inference script, model modules and a vendored copy of
  mPLUG-2 (Apache-2.0).
- `dataset/val_data.csv`: 6,000 rows with no labels.
- 10 sample validation videos (14 MB).
- `submission_baseline.csv`: predicted ECR for those 10 videos.

The checkpoints folder only contains a readme.

## 2. Environment and dependencies

`ECR_inference/environment.yaml` pins Python 3.8.5, PyTorch 1.11.0,
cudatoolkit 11.3, TensorFlow and tensorflow_hub (unpinned), diffusers 0.10.0,
transformers 4.25.1, timm 1.0.15, decord 0.6.0 and opencv-python 4.1.2.30.

`test_SnapUGC_baseline.py` loads all of the following at import time:
- `CompVis/stable-diffusion-v1-4`: the full pipeline, of which only the
  tokenizer and text encoder are used.
- `bert-large-uncased`.
- mPLUG-2 Large captioning.
- YAMNet, from TF Hub.
- EfficientNetV2-S, with ImageNet weights downloaded by timm.
- A UVQ-style distortion network.
- ResNet3D-18.

It calls `.cuda()` unconditionally, so there is no CPU or Apple MPS path.

## 3. Dataset files, schemas and labels

| File | Where | Rows | Columns | Label |
|---|---|---|---|---|
| `train_data.csv` (31.3 MB) | Google Drive (linked from repo) | 106,192 | Id, Title, Description, Download_link, ECR | ECR, continuous 0–1 |
| `val_data.csv` (1.7 MB) | Google Drive and repo | 6,000 | Id, Title, Description, Download_link | none (challenge validation set) |
| `snapugc_train.json` (87 MB) | HF mirror | 107,966 | video, label, conversations | 3 classes: 0 = not engaged, 1 = neutral, 2 = engaged |
| `snapugc_test.json` (11.9 MB) | HF mirror | 14,729 | same | 3 classes |
| `snapugc_{10..60}s_*`, `mini_*`, `*_short` | HF mirror | subsets | same | 3 classes |

Facts checked in `train_data.csv`:
- No duplicate IDs.
- Title is missing in 66,451 rows and Description in 60,914.
- ECR: min 0.000, p25 0.246, median 0.497, p75 0.749, max 1.000, mean 0.498.
- There is no NAWP column, even though the paper's second metric is NAWP.

Two inferences:

- **The released ECR looks rank-normalised.** Its quartiles are almost exactly
  0.25, 0.50 and 0.75, which is what a uniform distribution looks like. The paper
  describes raw ECR as bimodal with most values in [0, 0.8]. So the released
  value is probably a percentile rank, not the raw proportion. Rankings are
  preserved, but absolute values are not interpretable.
- **The HF 3-class labels were added by the mirror's uploader.** Their split
  (26.6k / 54.3k / 27.1k) is roughly 25/50/25 of ECR. They overlap heavily on
  ECR, though: class 0 has mean ECR 0.13, class 2 has mean 0.82, and every class
  spans 0 to 1. Prefer the continuous ECR from `train_data.csv`.

The paper says all videos are 10–60 s long. The downloaded subset includes a
6.4 s clip, so treat that range as approximate.

## 4. Are the videos downloadable?

- **Snapchat CDN links: no.** All 8 randomly sampled `Download_link` URLs from
  `train_data.csv` returned HTTP 404. Treat the CSV as metadata only.
- **HF mirror: partly.** It hosts 9,873 MP4s, all under `train/` (23.5 GB).
  - The `test/`, `train_new/` and `test_new/` paths referenced by the HF JSON
    files are not hosted.
  - 9,737 of the hosted videos are in `train_data.csv`.
  - Their ECR ranges from 0.000 to 0.522, and 95% are at or below 0.2. In the
    HF 3-class labels, all 9,872 labelled hosted videos are class 0.
  - So the downloadable labelled set is essentially the bottom quartile.
- **Repo:** the 10 validation videos are bundled, with no labels.

## 5. Pretrained checkpoints

Google Drive folder (sizes read from HTTP headers; nothing downloaded):

| File | Size |
|---|---|
| EVQA.pth | 0.77 GB |
| mPLUG2_MSRVTT_Caption.pth | 11.07 GB |
| ViT-L-14.tar | 0.93 GB |
| r3d18_K_200ep.pth | 0.27 GB |
| net_distort6_g_latest.pth | 0.12 GB |

That's about 13.2 GB, plus a Stable Diffusion v1.4 download (several GB),
bert-large, YAMNet and EfficientNetV2 weights. There's also a Baidu mirror.
Not practical within the hackathon's time or this disk (about 48 GB free).

## 6. Prediction targets and metrics

- **Targets.** The paper predicts NAWP (normalised average watch percentage) and
  ECR jointly. ECR is predicted from clips within the first 5 seconds only.
- **Released baseline output.** One ECR value per video, written as `Id,ECR`
  (the mean of the model output, clamped to [0, 1]).
- **Metrics.** SRCC, PLCC, RMSE, and RMSE on the top 10% of videos.
- **Paper results on ECR.** Their model gets SRCC 0.675 and PLCC 0.688. The best
  VQA-only baseline (DOVER) gets SRCC 0.619.
- **No temporal labels.** The labels are aggregated over 2,000+ viewers. There is
  no per-timestamp signal. Hookline must not call ECR-derived output a
  "retention curve".

## 7. Licensing and restrictions

- The GitHub repo has **no top-level licence**. Only the vendored `mPLUG_2/`
  carries Apache-2.0.
- The HF mirror is tagged `apache-2.0`, but by a third-party uploader (tcm03),
  not by the authors or Snap.
- The videos are Snapchat Spotlight user content. Copyright stays with the
  creators.
- Treat it all as **research/internal evaluation only**:
  - Don't commit or redistribute the videos.
  - Don't show them in a public demo.
  - Cite the paper if you report results.
- The baseline also pulls Stable Diffusion v1.4 (CreativeML OpenRAIL-M licence).
  That's another reason to avoid it.

## 8. Can inference run on this hardware?

No. The machine is an Apple M2 with 16 GiB RAM, no NVIDIA GPU, and about 48 GB
free disk. The blockers:
- `.cuda()` is hard-coded throughout.
- cudatoolkit 11.3 and PyTorch 1.11 have no Apple-silicon CUDA path.
- The 11 GB mPLUG-2 checkpoint alone is close to RAM limits once loaded.

Porting it to CPU/MPS would mean patching every `.cuda()` call and resolving
2022-era dependency pins against TensorFlow on arm64. That's several hours with
uncertain success, for a model whose output is a single score.

Fallback if it's ever needed: rent a single CUDA GPU (24 GB or more, e.g. an
A10/L4/A100 cloud instance) and run:

```bash
conda env create -f environment.yaml
# then download checkpoints into ECR_inference/checkpoints/
python3 test_SnapUGC_baseline.py --videos_dir dataset/test_videos/ --csv_file dataset/val_data_sample.csv
```

That's about 15 GB of downloads. This was not run here.

## 9. Smallest valid subset (prepared locally)

`scripts/fetch_snapugc_subset.py --yes --n 12` downloaded:
- **12 videos (27.9 MB)** from the HF `train/` folder, stratified across the
  hosted ECR range (0.000 to 0.205).
- The 10 unlabelled repo validation clips.

All 22 files probed as readable: vertical 480×852–868, about 30 fps, all with
audio, 5.8–60.1 s long.

What it's good for:
- Real-world inputs for the preprocessing and feature pipeline in Task 2.
- A sanity check that features vary sensibly within the low-ECR range.

What it can't do: train or validate an engagement model. The ECR range is
truncated, so any correlation would be uninformative.

To build a useful labelled evaluation set, you'd need videos spanning the full
ECR range. Those aren't publicly hosted, so you'd have to ask the authors or
the challenge organisers (VQualA 2025 EVQA-SnapUGC, ICCV 2025) for access.

## Manual fallback (if automated access breaks)

1. **Labels:** open the repo's `dataset/readme.md` and download "train set"
   (`train_data.csv`) and "val set" (`val_data.csv`) from Google Drive in a
   browser. Save them to `data/snapugc/metadata/`.
2. **Videos:** run `python scripts/fetch_snapugc_subset.py` without `--yes`. It
   prints the planned video IDs and makes no video downloads. Download each one
   from `https://huggingface.co/datasets/tcm03/SnapUGC/resolve/main/train/<Id>.mp4`
   in a browser and save it as `data/snapugc/videos/<Id>.mp4`.
3. Re-run the same command (same `--n` and `--seed`). It picks up the files
   already on disk and writes `data/snapugc/manifest.csv`. Then run
   `python scripts/inspect_dataset.py`.
