"""Versioned pipeline configuration.

Changing a preprocessing field (or PREPROCESS_VERSION) invalidates cached
proxies/WAVs; changing any field (or FEATURE_VERSION) invalidates features.
Speech, OCR, librosa and Gemini have their own configs and versions, so changing
them never invalidates the visual/audio features (and vice versa).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

PREPROCESS_VERSION = "1"
FEATURE_VERSION = "2"
SCHEMA_VERSION = "1"
SPEECH_VERSION = "1"
OCR_VERSION = "4"
LIBROSA_FEATURE_VERSION = "1"
GEMINI_PROMPT_VERSION = "1"
GEMINI_SCHEMA_VERSION = "1"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
SIGLIP_VERSION = "1"
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"  # Hugging Face commit, 2025-02-21

PREPROCESS_FIELDS = ("proxy_fps", "proxy_short_side", "audio_sample_rate")


@dataclass(frozen=True)
class PipelineConfig:
    # Preprocessing
    proxy_fps: int = 10
    proxy_short_side: int = 360  # never upscaled
    audio_sample_rate: int = 16000

    # Segmentation
    window_seconds: float = 0.5
    min_segment_seconds: float = 0.2  # shorter trailing pieces merge into a neighbour
    min_coverage: float = 0.5  # below this, a measured window is "uncertain"

    # Visual
    scene_cut_threshold: float = 0.4  # Bhattacharyya distance between HSV histograms
    struct_thumb_width: int = 64  # grayscale thumbnail width for structural change
    struct_cut_threshold: float = 0.2  # 1 - max(NCC, 0) between consecutive thumbnails
    struct_cut_margin: float = 0.15  # must exceed the local median by this much (rejects pans/noise)
    cut_context_frames: int = 5  # frames each side used for the local median
    motion_blur_kernel: int = 5  # Gaussian blur before frame differencing, reduces noise
    dark_brightness: float = 40.0
    bright_brightness: float = 220.0
    low_contrast_std: float = 12.0
    low_sharpness_lapvar: float = 20.0  # Laplacian variance on the proxy
    noise_sigma_warn: float = 3.0  # Immerkaer noise estimate on the proxy; noise also inflates sharpness
    low_resolution_short_side: int = 480
    low_fps: float = 15.0

    # Audio
    audio_frame_seconds: float = 0.02
    stft_n_fft: int = 512
    stft_hop: int = 160
    silence_dbfs: float = -50.0
    db_floor: float = -100.0
    clip_level: float = 0.999
    clipping_warn_ratio: float = 0.001
    spectrum_floor_db: float = -80.0  # dBFS floor per STFT bin, so hiss and silence do not create onsets
    onset_delta_db: float = 1.0  # peak must exceed the local mean flux by this much
    onset_max_window: float = 0.03  # peak must be the local maximum within +/- this many seconds
    onset_avg_window: float = 0.1  # +/- seconds for the local mean
    onset_wait_seconds: float = 0.1  # minimum gap between onsets
    onset_energy_lag_frames: int = 2  # an onset needs frame energy not to fall vs this many hops earlier
    onset_energy_drop_db: float = 1.0  # (rejects broadband splatter when a sound stops abruptly)

    # Summaries
    opening_seconds: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)

    def preprocess_dict(self) -> dict:
        return {k: getattr(self, k) for k in PREPROCESS_FIELDS} | {"preprocess_version": PREPROCESS_VERSION}

    def feature_dict(self) -> dict:
        return self.to_dict() | {"preprocess_version": PREPROCESS_VERSION, "feature_version": FEATURE_VERSION}


@dataclass(frozen=True)
class SpeechConfig:
    backend: str = "faster-whisper"
    model: str = "small"  # multilingual Whisper size (tiny/base/small/medium/large-v3); avoid *.en models
    model_revision: str | None = None  # Hugging Face revision of the converted model; None = latest
    compute_type: str = "int8"  # CPU: int8; CUDA: float16
    device: str = "cpu"  # "cpu", "cuda" or "auto"; not part of the cache key
    language: str | None = None  # None = auto-detect per video; never translated
    beam_size: int = 5
    vad_filter: bool = True  # Silero VAD (bundled with faster-whisper) skips non-speech audio
    word_timestamps: bool = True
    condition_on_previous_text: bool = False  # reduces repetition loops on music/noise
    skip_silent_audio: bool = True  # no model call when every 20 ms frame is below silence_dbfs
    silence_dbfs: float = -50.0
    low_logprob: float = -1.0  # segment flags (possible hallucination), not removals
    high_no_speech_prob: float = 0.6
    high_compression_ratio: float = 2.4

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_dict(self) -> dict:
        d = self.to_dict()
        d.pop("device")
        return d | {"hookline_speech_version": SPEECH_VERSION}


@dataclass(frozen=True)
class OcrConfig:
    backend: str = "paddleocr"
    sample_fps: float = 2.0  # each sample stands for [t, t + 1/sample_fps)
    frame_short_side: int = 720  # never upscaled
    max_seconds: float | None = None  # OCR only the first N seconds (None = whole video)
    det_model: str | None = "PP-OCRv5_mobile_det"  # set both model names, or neither and use lang/ocr_version
    rec_model: str | None = "en_PP-OCRv5_mobile_rec"
    lang: str | None = None  # e.g. "hi" for Devanagari; used only when model names are None
    ocr_version: str | None = None
    device: str = "cpu"  # "cpu" or "gpu:0"; not part of the cache key
    min_score: float = 0.5  # recognition confidence below this is dropped
    track_text_similarity: float = 0.8  # same text across frames (tolerates OCR jitter)
    track_iou: float = 0.3  # same place across frames
    track_max_gap_samples: int = 1  # missed samples bridged before text counts as reappearing

    def __post_init__(self) -> None:
        if (self.det_model is None) != (self.rec_model is None):
            raise ValueError("set both det_model and rec_model, or neither (then lang/ocr_version choose models)")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_dict(self) -> dict:
        d = self.to_dict()
        d.pop("device")
        return d | {"hookline_ocr_version": OCR_VERSION}


@dataclass(frozen=True)
class LibrosaConfig:
    """Frame-level librosa features on the cached mono WAV (same STFT grid as Task 2 audio)."""

    n_fft: int = 512  # 32 ms at 16 kHz
    hop: int = 160  # 10 ms
    n_mels: int = 64  # onset-strength mel bands (64 avoids empty filters at n_fft=512)
    rolloff_percent: float = 0.85
    silence_dbfs: float = -50.0  # frames below are "low energy"; spectral shape is not reported for them
    db_floor: float = -100.0
    low_energy_min_seconds: float = 0.3  # shortest low-energy interval listed
    tempo_min_seconds: float = 6.0  # no tempo estimate on shorter audio
    tempo_min_beats: int = 8
    tempo_min_beat_onset_agreement: float = 0.8  # share of tracked beats within 50 ms of a detected onset

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_dict(self) -> dict:
        return self.to_dict() | {"hookline_librosa_feature_version": LIBROSA_FEATURE_VERSION}


@dataclass(frozen=True)
class GeminiConfig:
    """Gemini video interpretation. ``mode``: off (default), offline (no network) or remote."""

    mode: str = "off"
    model: str = DEFAULT_GEMINI_MODEL
    allow_remote: bool = False  # must be True for any upload, in addition to mode="remote"
    approved_ids: tuple[str, ...] = ()  # video_ids under protected roots that may be uploaded
    protected_roots: tuple[str, ...] = ("01-Hookline",)  # company data: never uploaded unless approved
    fps: float = 1.0  # frames per second Gemini samples (API default 1)
    media_resolution: str = "low"  # low / medium / high
    temperature: float = 0.2
    timeout_s: float = 180.0  # per API request; not part of the cache key
    max_retries: int = 2  # on 429 / 5xx / timeouts; not part of the cache key
    upload_wait_s: float = 300.0  # max wait for the uploaded file to become ACTIVE
    offline_response: str | None = None  # offline mode: JSON file to use instead of an API response

    def __post_init__(self) -> None:
        if self.mode not in ("off", "offline", "remote"):
            raise ValueError("gemini mode must be off, offline or remote")
        if self.media_resolution not in ("low", "medium", "high"):
            raise ValueError("media_resolution must be low, medium or high")
        if self.fps <= 0 or self.timeout_s <= 0 or self.max_retries < 0:
            raise ValueError("fps and timeout must be positive, max_retries >= 0")

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("mode", "model", "fps", "media_resolution", "temperature", "offline_response")}
        return d | {"prompt_version": GEMINI_PROMPT_VERSION, "schema_version": GEMINI_SCHEMA_VERSION}


@dataclass(frozen=True)
class SiglipConfig:
    """SigLIP 2 image embeddings and description scores on sampled proxy frames."""

    model_id: str = SIGLIP_MODEL_ID
    model_revision: str | None = SIGLIP_MODEL_REVISION  # pinned commit; None = latest (not reproducible)
    sample_fps: float = 2.0  # proxy frames per second sent to the model (proxy is 10 fps)
    cover_every_segment: bool = True  # add the proxy frame nearest the midpoint of segments the grid misses
    prompt_template: str = "This is a photo of {}."  # the template SigLIP's zero-shot pipeline uses
    batch_size: int = 16
    device: str = "auto"  # auto (MPS if available, else CPU), mps or cpu; not part of the cache key
    store_embeddings: bool = True  # siglip_embeddings.npy (float16, L2-normalised)

    def __post_init__(self) -> None:
        if self.sample_fps <= 0 or self.batch_size < 1:
            raise ValueError("sample_fps must be positive and batch_size >= 1")
        if self.device not in ("auto", "mps", "cpu"):
            raise ValueError("device must be auto, mps or cpu")
        if "{}" not in self.prompt_template:
            raise ValueError("prompt_template needs a {} placeholder")

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_dict(self) -> dict:
        d = self.to_dict()
        d.pop("device")
        d.pop("batch_size")
        return d | {"hookline_siglip_version": SIGLIP_VERSION}


def stable_hash(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\x00")
    return h.hexdigest()


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
