# data/snapugc: tiny SnapUGC research subset (git-ignored)

Populated by `scripts/fetch_snapugc_subset.py`. Nothing here is committed.

```
data/snapugc/
├── metadata/
│   ├── train_data.csv        # 106,192 rows: Id, Title, Description, Download_link, ECR
│   ├── val_data.csv          # 6,000 rows: same columns minus ECR (no public labels)
│   └── hf_hosted_index.csv   # videos actually hosted on HF, joined with their ECR
├── videos/                   # small ECR-stratified subset from HF tcm03/SnapUGC train/
├── repo_val_samples/         # optional: 10 unlabeled val clips bundled in the GitHub repo
└── manifest.csv              # Hookline manifest with label_name=ECR where known
```

Usage restrictions: research/internal testing only. These are Snapchat users'
videos; the original repository ships no licence, and the Hugging Face mirror's
`apache-2.0` tag was set by a third-party uploader. Do not redistribute them or
show them in a public demo. See `reports/snapugc_feasibility.md`.
