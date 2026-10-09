"""SigLIP 2 visual-semantic signals on sampled proxy frames (opt-in, ``--siglip``).

Model: ``google/siglip2-base-patch16-224`` (pinned revision, see ``SiglipConfig``), used
as-is; nothing is trained. Frames come from the cached analysis proxy that Task 2 already
decoded: proxy frame ``i`` is row ``i`` of ``frames.csv`` and shows the source frame on
screen at ``frames.csv.time_s`` on the original playback timeline. The source video is
never read here. Frames are resized to 224 x 224 by the model's processor (no crop, so
portrait frames are squashed).

Per sampled frame:
- an L2-normalised image embedding (``siglip_embeddings.npy``, float16, in frame order)
- ``similarity_to_previous``: cosine similarity with the previous sampled frame, and
  ``change`` = 1 - similarity; ``crosses_cut`` if a detected scene cut lies between the two
- ``scores``: one sigmoid score per description, ``sigmoid(cos * exp(logit_scale) + logit_bias)``
  with the model's own scale and bias, as in SigLIP's zero-shot classification. Each score is
  independent (they do not sum to 1). They are similarity scores, not calibrated
  probabilities, and not ground truth about the frame.

Per segment: description scores averaged over the segment's sampled frames, the largest
within-shot change (pairs not crossing a detected cut: content change the cut detector does
not already mark) and, separately, the change across a cut (how different the new shot
is). A large change is not "more engaging" and a small one is not "boring".
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .config import SCHEMA_VERSION, SIGLIP_VERSION, SiglipConfig, stable_hash
from .context import BackendUnavailable, LazyBackend, MediaContext, peak_rss_mb
from .status import ABSENT, AVAILABLE, FAILED, MEASURED_STATUSES, UNCERTAIN

# Descriptions scored on every sampled frame. Keys are column suffixes; keep them stable.
# A single frame cannot show motion or editing rhythm, so "montage" and "static shot" score
# how much one frame looks like such imagery (e.g. a collage, a plain locked-off frame).
SIGLIP_LABELS: tuple[tuple[str, str], ...] = (
    ("talking_to_camera", "a person speaking directly to the camera"),
    ("product_demo", "a product demonstration"),
    ("person_closeup", "a close-up of a person"),
    ("montage", "a montage of rapidly changing scenes"),
    ("text_screen", "a screen full of text"),
    ("static_shot", "a static shot"),
    ("outdoor_scene", "an outdoor scene"),
)
SIGLIP_COLUMNS = [
    "siglip_status", "siglip_frames", "siglip_frame_ids", "siglip_within_shot_change", "siglip_cut_change",
    "siglip_top_label",
] + [f"siglip_score_{k}" for k, _ in SIGLIP_LABELS]
EMBEDDINGS_FILE = "siglip_embeddings.npy"
NORM_TOLERANCE = 1e-3


class Embedder(Protocol):
    logit_scale: float  # already exponentiated
    logit_bias: float

    def info(self) -> dict: ...

    def encode_images(self, images_rgb: list[np.ndarray]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


def library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"numpy": np.__version__}
    for name in ("torch", "transformers"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def select_device(preference: str, torch_mod: Any) -> str:
    """``mps`` when requested/auto and PyTorch reports it usable on this machine, else ``cpu``."""
    if preference == "cpu":
        return "cpu"
    mps = getattr(getattr(torch_mod, "backends", None), "mps", None)
    usable = bool(mps and mps.is_available() and mps.is_built())
    if preference == "mps" and not usable:
        raise BackendUnavailable("MPS requested but not available in this PyTorch build/machine")
    return "mps" if usable else "cpu"


class HfSiglipEmbedder:
    """``transformers`` SigLIP model + processor, loaded once per run, eval mode, no gradients.

    Weights are only read from the local Hugging Face cache unless ``allow_download``; the
    model is public, so no token is needed. On MPS, an unsupported operation moves the model
    to the CPU once and the run continues there (recorded in ``info()``)."""

    def __init__(self, cfg: SiglipConfig, allow_download: bool = False):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise BackendUnavailable("torch/transformers are not installed (pip install -r requirements-models.txt)") from exc
        self._torch = torch
        kw = {"revision": cfg.model_revision, "local_files_only": not allow_download}
        t0 = time.perf_counter()
        try:
            self._processor = AutoProcessor.from_pretrained(cfg.model_id, **kw)
            self._model = AutoModel.from_pretrained(cfg.model_id, dtype=torch.float32, **kw).eval()
        except OSError as exc:
            hint = "" if allow_download else " (not in the local Hugging Face cache; re-run with --download-models, ~1.5 GB)"
            raise BackendUnavailable(f"could not load {cfg.model_id}@{(cfg.model_revision or 'latest')[:12]}{hint}: {exc}") from exc
        self.device = select_device(cfg.device, torch)
        self._model.to(self.device)
        self.mps_fallback: str | None = None
        self.load_seconds = round(time.perf_counter() - t0, 3)
        self.logit_scale = float(self._model.logit_scale.detach().exp().cpu())
        self.logit_bias = float(self._model.logit_bias.detach().cpu())
        self._cfg = cfg

    def info(self) -> dict:
        return {"backend": "transformers", "model_id": self._cfg.model_id, "model_revision": self._cfg.model_revision,
                "device": self.device, "mps_fallback": self.mps_fallback, "load_seconds": self.load_seconds,
                "model_class": type(self._model).__name__}

    def _run(self, fn: Callable[[], Any]) -> np.ndarray:
        torch = self._torch
        try:
            with torch.inference_mode():
                return fn()
        except (RuntimeError, NotImplementedError) as exc:
            if self.device != "mps":
                raise
            self.mps_fallback = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.device = "cpu"
            self._model.to("cpu")
            with torch.inference_mode():
                return fn()

    @staticmethod
    def _pooled(out: Any) -> Any:
        # transformers >= 5 returns BaseModelOutputWithPooling; 4.x returned the tensor.
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def encode_images(self, images_rgb: list[np.ndarray]) -> np.ndarray:
        def fn():
            inputs = self._processor(images=list(images_rgb), return_tensors="pt").to(self.device)
            emb = self._pooled(self._model.get_image_features(**inputs))
            return (emb / emb.norm(p=2, dim=-1, keepdim=True)).float().cpu().numpy()
        return self._run(fn)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        def fn():
            # SigLIP was trained on text padded to 64 tokens; other padding changes the scores.
            inputs = self._processor(text=list(texts), padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            emb = self._pooled(self._model.get_text_features(**inputs))
            return (emb / emb.norm(p=2, dim=-1, keepdim=True)).float().cpu().numpy()
        return self._run(fn)


def frame_selection(frame_times: list[float], segments: list[dict], proxy_fps: float, cfg: SiglipConfig) -> list[int]:
    """Proxy frame indices to embed: a fixed ``sample_fps`` grid over the proxy, plus (if
    ``cover_every_segment``) the frame nearest the midpoint of each segment the grid misses."""
    n = len(frame_times)
    if not n:
        return []
    step = max(1, int(round(proxy_fps / cfg.sample_fps)))
    chosen = set(range(0, n, step))
    if cfg.cover_every_segment:
        times = np.asarray(frame_times)
        last = len(segments) - 1
        for i, s in enumerate(segments):
            inside = np.nonzero((times >= s["start_s"]) & ((times < s["end_s"]) | ((i == last) & (times <= s["end_s"]))))[0]
            if inside.size and not chosen.intersection(inside.tolist()):
                mid = (s["start_s"] + s["end_s"]) / 2
                chosen.add(int(inside[np.argmin(np.abs(times[inside] - mid))]))
    return sorted(chosen)


def read_frame_times(frames_csv: Path) -> list[float]:
    import csv

    with frames_csv.open() as f:
        return [float(r["time_s"]) for r in csv.DictReader(f)]


def siglip_key(ctx: MediaContext, cfg: SiglipConfig, selected_times: list[float]) -> str:
    return stable_hash("siglip", ctx.source_sha256 or ctx.source_path, ctx.prep_key, cfg.cache_dict(),
                       [round(t, 6) for t in selected_times], library_versions())


def _base(ctx: MediaContext, cfg: SiglipConfig, key: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "siglip_version": SIGLIP_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": ctx.video_id,
        "source_sha256": ctx.source_sha256,
        "cache_key": key,
        "status": FAILED,
        "measured": False,
        "reason": None,
        "model": {"model_id": cfg.model_id, "model_revision": cfg.model_revision},
        "versions": library_versions(),
        "config": cfg.to_dict(),
        "timeline": {"convention": "seconds from start of original playback; frame_index = row of frames.csv (proxy frame)",
                     "proxy_fps": ctx.proxy_fps, "duration_s": ctx.duration},
        "score_note": ("Per-description sigmoid(cos * exp(logit_scale) + logit_bias) with the model's own scale/bias. "
                       "Independent similarity scores, not calibrated probabilities or ground truth."),
        "labels": [{"key": k, "description": d, "prompt": cfg.prompt_template.format(d)} for k, d in SIGLIP_LABELS],
        "embedding": None,
        "frames": [],
        "segments": [],
        "timing": None,
        "warnings": [],
    }


def _check_embeddings(x: Any, n: int, what: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] != n or a.shape[1] < 1:
        raise ValueError(f"{what} embeddings have shape {a.shape}, expected ({n}, d)")
    if not np.isfinite(a).all():
        raise ValueError(f"{what} embeddings contain NaN/inf")
    norms = np.linalg.norm(a, axis=1)
    if (np.abs(norms - 1) > NORM_TOLERANCE).any():
        raise ValueError(f"{what} embeddings are not L2-normalised (norms {norms.min():.4f}-{norms.max():.4f})")
    return a


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _decode(ctx: MediaContext, indices: list[int], reader_factory: Callable[..., Any]) -> tuple[dict[int, np.ndarray], list[str]]:
    wanted, frames = set(indices), {}
    w, h = ctx.proxy_size
    reader = reader_factory(Path(ctx.proxy_path), int(w), int(h))
    for i, bgr in enumerate(reader):
        if i in wanted:
            frames[i] = np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB
        if i >= max(wanted):
            break
    return frames, list(getattr(reader, "errors", []))


def segment_rows(frames: list[dict], segments: list[dict]) -> list[dict]:
    rows = []
    last = len(segments) - 1
    for i, s in enumerate(segments):
        inside = [f for f in frames if s["start_s"] <= f["t_s"] < s["end_s"] or (i == last and f["t_s"] == s["end_s"])]
        row: dict[str, Any] = {"segment_index": s["segment_index"], "siglip_frames": len(inside),
                               "siglip_frame_ids": [f["frame_index"] for f in inside] or None, "notes": []}
        within = [f["change"] for f in inside if f["change"] is not None and not f["crosses_cut"]]
        across = [f["change"] for f in inside if f["change"] is not None and f["crosses_cut"]]
        row["siglip_within_shot_change"] = round(max(within), 4) if within else None
        row["siglip_cut_change"] = round(max(across), 4) if across else None
        means = {k: round(float(np.mean([f["scores"][k] for f in inside])), 4) if inside else None for k, _ in SIGLIP_LABELS}
        for k, v in means.items():
            row[f"siglip_score_{k}"] = v
        row["siglip_top_label"] = max(means, key=lambda k: means[k]) if inside else None
        if not inside:
            row["notes"].append("siglip=not_applicable: no sampled frame in segment")
        rows.append(row)
    return rows


def extract_siglip(
    ctx: MediaContext,
    cfg: SiglipConfig,
    backend: LazyBackend | None,
    frame_times: list[float] | None,
    reader_factory: Callable[..., Any] | None = None,
) -> tuple[dict, np.ndarray | None]:
    """(artifact, embeddings or None). Never raises; the model is loaded only if frames need it."""
    indices = frame_selection(frame_times or [], ctx.segments, ctx.proxy_fps or 10.0, cfg)
    art = _base(ctx, cfg, siglip_key(ctx, cfg, [frame_times[i] for i in indices] if frame_times else []))
    if ctx.visual_stream != AVAILABLE:
        art["status"] = ctx.visual_stream
        art["reason"] = {ABSENT: "no video stream", UNCERTAIN: "video presence unknown (inspection failed)",
                         FAILED: "video stream could not be decoded"}.get(ctx.visual_stream, f"video status {ctx.visual_stream}")
        return art, None
    if not ctx.proxy_path or not Path(ctx.proxy_path).is_file() or not ctx.proxy_size:
        art["reason"] = "cached proxy missing; re-run scripts/run_pipeline.py"
        return art, None
    if not indices:
        art["reason"] = "no proxy frames listed in frames.csv"
        return art, None
    if backend is None:
        art["reason"] = "no SigLIP backend configured"
        return art, None
    t0 = time.perf_counter()
    if reader_factory is None:
        from .visual import FrameReader
        reader_factory = FrameReader
    try:
        images, decode_errors = _decode(ctx, indices, reader_factory)
    except Exception as exc:
        art["reason"] = f"proxy decoding failed: {type(exc).__name__}: {exc}"
        return art, None
    art["warnings"] += decode_errors
    missing = [i for i in indices if i not in images]
    if missing:
        art["warnings"].append(f"{len(missing)} selected proxy frames could not be decoded: {missing[:10]}")
    kept = [i for i in indices if i in images]
    if not kept:
        art["reason"] = "no selected proxy frame could be decoded"
        return art, None
    try:
        model: Embedder = backend.get()
    except BackendUnavailable as exc:
        art["reason"] = f"model unavailable: {exc}"
        return art, None
    t1 = time.perf_counter()
    try:
        texts = [cfg.prompt_template.format(d) for _, d in SIGLIP_LABELS]
        text_emb = _check_embeddings(model.encode_texts(texts), len(texts), "text")
        batches = [_check_embeddings(model.encode_images([images[i] for i in kept[b:b + cfg.batch_size]]),
                                     len(kept[b:b + cfg.batch_size]), "image")
                   for b in range(0, len(kept), cfg.batch_size)]
        emb = np.concatenate(batches)
        if emb.shape[1] != text_emb.shape[1]:
            raise ValueError(f"image ({emb.shape[1]}) and text ({text_emb.shape[1]}) embedding sizes differ")
        scale, bias = float(model.logit_scale), float(model.logit_bias)
        if not (math.isfinite(scale) and math.isfinite(bias)):
            raise ValueError("model logit scale/bias are not finite")
    except Exception as exc:
        art["reason"] = f"inference failed: {type(exc).__name__}: {exc}"
        return art, None
    infer_s = time.perf_counter() - t1
    scores = _sigmoid(emb @ text_emb.T * scale + bias)
    cuts = sorted(ctx.scene_cut_times or [])
    frames = []
    for j, i in enumerate(kept):
        t = frame_times[i]
        sim = change = None
        crosses = False
        if j:
            sim = float(np.clip(emb[j] @ emb[j - 1], -1.0, 1.0))
            change = round(1.0 - sim, 4)
            t_prev = frame_times[kept[j - 1]]
            crosses = any(t_prev < c <= t + 1e-9 for c in cuts)
        frames.append({"frame_index": i, "t_s": round(t, 6), "similarity_to_previous": None if sim is None else round(sim, 4),
                       "change": change, "crosses_cut": crosses,
                       "scores": {k: round(float(scores[j, n]), 5) for n, (k, _) in enumerate(SIGLIP_LABELS)}})
    info = model.info() if hasattr(model, "info") else {}
    art.update(
        status=AVAILABLE,
        measured=True,
        reason=None,
        model=art["model"] | {k: v for k, v in info.items() if k not in ("model_id",)},
        embedding={"dim": int(emb.shape[1]), "count": int(emb.shape[0]), "normalised": "L2",
                   "file": EMBEDDINGS_FILE if cfg.store_embeddings else None, "dtype": "float16", "sha256": None,
                   "logit_scale": round(scale, 5), "logit_bias": round(bias, 5)},
        frames=frames,
        segments=segment_rows(frames, ctx.segments),
        timing={"frames": len(kept), "inference_seconds": round(infer_s, 3),
                "seconds_per_frame": round(infer_s / len(kept), 4), "total_seconds": round(time.perf_counter() - t0, 3),
                "device": info.get("device"), "process_peak_rss_mb": peak_rss_mb()},
    )
    if info.get("mps_fallback"):
        art["warnings"].append(f"MPS operation unsupported, ran on CPU: {info['mps_fallback']}")
    return art, emb


def embeddings_bytes(emb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, emb.astype(np.float16), allow_pickle=False)
    return buf.getvalue()


def write_siglip(out_dir: Path, art: dict, emb: np.ndarray | None, write_json: Callable[[Path, dict], None]) -> bool:
    """Write ``siglip.json`` (+ embeddings). A failure never replaces a usable earlier artifact."""
    path = out_dir / "siglip.json"
    if not art.get("measured"):
        try:
            import json
            old = json.loads(path.read_text())
        except (OSError, ValueError):
            old = None
        if old and old.get("measured"):
            return False
        write_json(path, art)
        return True
    if emb is not None and art["embedding"]["file"]:
        data = embeddings_bytes(emb)
        art["embedding"]["sha256"] = hashlib.sha256(data).hexdigest()
        tmp = out_dir / (EMBEDDINGS_FILE + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(out_dir / EMBEDDINGS_FILE)
    write_json(path, art)
    return True


def siglip_window(art: dict | None, seg: dict) -> tuple[dict, list[str]]:
    out: dict[str, Any] = {k: None for k in SIGLIP_COLUMNS}
    if art is None:
        out["siglip_status"] = UNCERTAIN
        return out, ["siglip=not run (disabled)"]
    if not art.get("measured"):
        out["siglip_status"] = art["status"]
        return out, []
    status = seg.get("visual_status") or UNCERTAIN
    out["siglip_status"] = status
    if status not in MEASURED_STATUSES:
        return out, []
    row = next((r for r in art["segments"] if r["segment_index"] == seg["segment_index"]), None)
    if row is None:
        out["siglip_status"] = FAILED
        return out, ["siglip=segment missing from siglip.json"]
    out.update({k: row.get(k) for k in SIGLIP_COLUMNS if k != "siglip_status"})
    if not row["siglip_frames"]:
        out["siglip_status"] = UNCERTAIN
    return out, list(row.get("notes") or [])
