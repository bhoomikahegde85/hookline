# data/raw: your own demo videos

Put short-form videos you have permission to use here, then list each one in
`manifest.csv`. Video files are git-ignored; only this README and the manifest
are tracked.

Do not add scraped Instagram/TikTok/YouTube videos or anything you do not have
rights to. Good sources: your own phone recordings, clips friends have given you
written permission to use, or openly licensed footage (e.g. CC0 / CC-BY, with the
licence and attribution recorded in the manifest).

## Adding a video

1. Copy the file, e.g. `data/raw/talking_head_01.mp4`.
2. Add a row to `manifest.csv`. `file_path` is relative to this folder.
3. Run `python scripts/inspect_dataset.py`. It probes the file and flags any
   mismatch between the manifest and the file (duration, audio, video).

Example row:

```csv
video_id,file_path,duration_seconds,has_audio,has_video,source,license_or_permission,dataset_split,notes
talking_head_01,talking_head_01.mp4,23.4,true,true,recorded by me on iPhone,own content,demo,captions burned in
```

## Columns

| Column | Meaning |
|---|---|
| `video_id` | Unique ID (no spaces). |
| `file_path` | Path relative to this folder (absolute paths also work). |
| `duration_seconds` | Leave blank if unknown; the inspector reports the real value. |
| `has_audio`, `has_video` | `true`, `false` or `unknown`/blank. Unknown is not the same as false. |
| `source` | Where the clip came from. |
| `license_or_permission` | Required: why we may use it (e.g. `own content`, `CC-BY 4.0, author X`). |
| `dataset_split` | `demo`, `train`, `val`, `test`, `fixture` or `unassigned`. |
| `notes` | Format (talking head, meme, slideshow...), captions, language, anything unusual. |

Optional label columns `label_name`, `label_value` and `label_source` may only be
filled when the label's definition and source are known (e.g. real retention
data exported from your own analytics). The validator rejects a value without a
name and source.
