"""Custom Hugging Face Inference Endpoint handler for TRIBE v2 (facebook/tribev2).

TRIBE v2 has no transformers pipeline (no ``pipeline_tag``/``library_name`` on the Hub), so the
default Inference Endpoints toolkit cannot serve it: this handler is required. It runs the
upstream inference path from https://github.com/facebookresearch/tribev2 (``tribev2/demo_utils.py``):

    TribeModel.from_pretrained(<facebook/tribev2 at a pinned revision>)
    events = model.get_events_dataframe(video_path=...)   # audio extraction, whisperX words, chunking
    preds, segments = model.predict(events)                # (n_segments, n_vertices) on fsaverage5

Request (JSON):  {"inputs": {"video_b64": "<base64 of the file>", "filename": "clip.mp4"},
                  "parameters": {"language": "english"}}
Response (JSON): {"handler_version", "model_repo", "model_revision", "tr_s", "mesh", "shape", "dtype",
                  "preds_b64" (little-endian float32, row-major), "segments": [{"start_s", "duration_s"}],
                  "events": {type: count}, "language", "timing", "versions"}  or  {"error": "..."}.

Outputs are TRIBE v2's predicted fMRI responses of an "average subject" on the fsaverage5 cortical
mesh, one row per kept 1-TR segment (segments without any stimulus event are dropped upstream).
They are not retention, engagement or virality predictions.

Licence: TRIBE v2 weights and code are CC BY-NC 4.0 (non-commercial). Its text features use
meta-llama/Llama-3.2-3B, a gated model: the endpoint's HF_TOKEN must belong to an account that has
accepted the Llama 3.2 licence.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any

HANDLER_VERSION = "1"
MODEL_REPO = "facebook/tribev2"
MODEL_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}  # tribev2.demo_utils.VALID_SUFFIXES["video_path"]
LANGUAGES = ("english", "french", "spanish", "dutch", "chinese")  # tribev2 ExtractWordsFromAudio
CACHE_DIR = os.environ.get("TRIBE_CACHE_DIR", "/tmp/tribe_cache")


def _versions() -> dict[str, str | None]:
    from importlib import metadata
    out = {}
    for dist in ("tribev2", "neuralset", "torch", "transformers", "numpy"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def video_events(model: Any, video_path: str, language: str):
    """Events for one video. English is exactly ``model.get_events_dataframe``; other languages run the
    same transform list as ``tribev2.demo_utils.get_audio_and_text_events`` with that whisperX language."""
    if language == "english":
        return model.get_events_dataframe(video_path=video_path)
    import pandas as pd
    from neuralset.events.transforms import (AddContextToWords, AddSentenceToWords, AddText, ChunkEvents,
                                             ExtractAudioFromVideo, RemoveMissing)
    from neuralset.events.utils import standardize_events
    from tribev2.eventstransforms import ExtractWordsFromAudio

    events = standardize_events(pd.DataFrame([{"type": "Video", "filepath": video_path, "start": 0,
                                               "timeline": "default", "subject": "default"}]))
    for transform in (ExtractAudioFromVideo(),
                      ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
                      ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30),
                      ExtractWordsFromAudio(language=language), AddText(),
                      AddSentenceToWords(max_unmatched_ratio=0.05),
                      AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
                      RemoveMissing()):
        events = transform(events)
    return standardize_events(events)


class EndpointHandler:
    def __init__(self, path: str = ""):
        from huggingface_hub import snapshot_download
        from tribev2 import TribeModel

        t0 = time.time()
        ckpt_dir = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=["config.yaml", "best.ckpt"])
        self.model = TribeModel.from_pretrained(ckpt_dir, cache_folder=CACHE_DIR)
        self.load_seconds = round(time.time() - t0, 1)
        self.versions = _versions()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        inputs = data.get("inputs") if isinstance(data, dict) else None
        params = (data.get("parameters") or {}) if isinstance(data, dict) else {}
        if not isinstance(inputs, dict) or not isinstance(inputs.get("video_b64"), str):
            return {"error": "expected JSON {'inputs': {'video_b64': <base64>, 'filename': <name>}}"}
        suffix = Path(str(inputs.get("filename") or "video.mp4")).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            return {"error": f"unsupported video suffix {suffix!r}; expected one of {sorted(VIDEO_SUFFIXES)}"}
        language = params.get("language", "english")
        if language not in LANGUAGES:
            return {"error": f"language must be one of {LANGUAGES}"}
        try:
            video = base64.b64decode(inputs["video_b64"], validate=True)
        except (ValueError, TypeError) as exc:
            return {"error": f"video_b64 is not valid base64: {exc}"}
        if not video:
            return {"error": "empty video"}

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"input{suffix}"
            path.write_bytes(video)
            try:
                events = video_events(self.model, str(path), language)
                t1 = time.time()
                preds, segments = self.model.predict(events=events, verbose=False)
            except Exception as exc:  # returned to the client as an error, never as a prediction
                return {"error": f"{type(exc).__name__}: {str(exc)[:800]}"}
        preds = np.ascontiguousarray(preds, dtype="<f4")
        return {
            "handler_version": HANDLER_VERSION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "tr_s": float(self.model.data.TR),
            "mesh": "fsaverage5",
            "shape": list(preds.shape),
            "dtype": "float32",
            "preds_b64": base64.b64encode(preds.tobytes()).decode("ascii"),
            "segments": [{"start_s": float(s.start), "duration_s": float(s.duration)} for s in segments],
            "events": {str(k): int(v) for k, v in events["type"].value_counts().items()},
            "language": language,
            "timing": {"events_seconds": round(t1 - t0, 2), "predict_seconds": round(time.time() - t1, 2),
                       "model_load_seconds": self.load_seconds},
            "versions": self.versions,
        }
