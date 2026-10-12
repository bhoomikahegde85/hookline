"""Versioned pipeline configuration.

Changing a preprocessing field (or PREPROCESS_VERSION) invalidates cached
proxies/WAVs; changing any field (or FEATURE_VERSION) invalidates features.
Speech, OCR, librosa, SigLIP and TRIBE have their own configs and versions, so changing
them never invalidates the visual/audio features (and vice versa).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

PREPROCESS_VERSION = "1"
FEATURE_VERSION = "2"
SCHEMA_VERSION = "1"
SPEECH_VERSION = "2"  # 2: no-speech / word-probability / language-confidence flags, speech_result
OCR_VERSION = "4"
LIBROSA_FEATURE_VERSION = "1"
SIGLIP_VERSION = "1"
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"  # Hugging Face commit, 2025-02-21

# TTCC pretrained retention model (scripts/run_ttcc.py). Identity verified against the Hugging Face
# repo on 2026-10-10; see docs/ttcc.md. Files are checked before every load: no fallback to another model.
TTCC_VERSION = "1"  # Hookline-side normalisation / artifact schema
TTCC_CHECKPOINT_REPO = "liangyuch/ttcc-sft-qwen25omni-3b-v8"
TTCC_CHECKPOINT_REVISION = "1686d5f00408bfd8f6a32be12e71248b6f8ff447"
TTCC_CHECKPOINT_FILES = {  # file -> (size in bytes, sha256); shard hashes are the Hugging Face LFS ids
    "config.json": (17092, "02bf6bfb9891c343a2b0e8533e2fc866d0179f564fab4cbe19b5dafa2c167fbe"),
    "model.safetensors.index.json": (127855, "dd5d6c1ef23527dace74a30781c0bb9cb0f78c6659ce024bcceadcc439e79374"),
    "model-00001-of-00002.safetensors": (4994850880, "3506b349c33f3fec88beb9fadcb1c59a2212899dadbdbb70b55fdb6023e3eac0"),
    "model-00002-of-00002.safetensors": (4412497472, "07efff41ed474855185a3b73c732ad53868ba09c10d4f83afdbf323e71f845ee"),
    "spk_dict.pt": (259544, "6a05609b28f5d42b7b748f0f07592545c8f1f6885b9ae8fff64baf56e86b2a18"),
    "tokenizer.json": (11421866, "1ab7a851e5c63d5fafbfdac72e3b4a8d08613f6bbb06ee39036fdf870ef59c93"),
}
TTCC_UPSTREAM_REPO = "https://github.com/cliangyu/go_viral"
TTCC_UPSTREAM_BRANCH = "ttcc-rl"
TTCC_UPSTREAM_VERIFIED_COMMIT = "893da03bccee6a18d221e96c5de55c9b35a8e350"  # commit the adapter was written against
DEVICE_MAPS = ("detect", "cuda", "auto", "mps", "cpu")
ATTN_IMPLS = ("detect", "flash_attn", "sdpa", "eager")

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
    low_word_probability: float = 0.5  # segment flag when its words' mean probability is below this
    low_language_probability: float = 0.5  # file flag when auto-detected language confidence is below this

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


GEMINI_EDIT_VERSION = "4"  # prompt + schema + validation + analysed clip selection of the segment interpretation (gemini_edit.json)
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"  # what this project used before; override with GEMINI_MODEL if your key serves another Flash model
TRIBE_VERSION = "1"  # Hookline-side request/artifact schema for TRIBE v2
TRIBE_MODEL_REPO = "facebook/tribev2"
TRIBE_MODEL_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"  # Hugging Face commit checked 2026-10-10 (CC BY-NC 4.0)


@dataclass(frozen=True)
class GeminiConfig:
    """Gemini interpretation of the predicted-drop-off segments (opt-in). The VIDEO IS UPLOADED to Google, so this needs ``allow_remote``,
    and videos under ``protected_roots`` (company reels) also need their id in ``approved_ids``. The key comes from GEMINI_API_KEY
    (environment or the untracked ``.env``) and is never stored in artifacts."""

    mode: str = "off"  # "off" or "remote"
    model: str = DEFAULT_GEMINI_MODEL
    allow_remote: bool = False
    approved_ids: tuple[str, ...] = ()
    protected_roots: tuple[str, ...] = ("01-Hookline",)
    fps: float = 1.0  # frames per second Gemini samples from the video (more frames cost more tokens)
    media_resolution: str = "medium"  # low / medium / high
    temperature: float = 0.2
    max_suggestions_per_segment: int = 3
    timeout_s: float = 240.0  # per request; not part of the cache key
    max_retries: int = 5  # on 429 / 5xx / timeouts (a 503 "high demand" can last a minute or two); not part of the cache key
    fallback_models: tuple[str, ...] = ("gemini-3.7-flash", "gemini-3.5-flash", "gemini-2.5-flash")  # tried in order when the main model stays overloaded (503/429); not in the key
    upload_wait_s: float = 300.0  # max wait for the uploaded file to become ACTIVE; not part of the cache key
    max_upload_mb: float = 100.0  # refuse larger files before sending
    # Cost control for long videos: Gemini is given at most max_analysed_s seconds of frames. A video whose predicted span fits is sent
    # whole; a longer one is sent as clips: the opening (hook_s) plus every flagged segment with context_margin_s around it, largest
    # predicted fall first, until the budget is used. Segments outside the clips are not interpreted (recorded, never invented).
    max_analysed_s: float = 120.0
    context_margin_s: float = 3.0
    hook_s: float = 5.0

    def __post_init__(self) -> None:
        if not (0 < self.hook_s <= self.max_analysed_s) or self.context_margin_s < 0:
            raise ValueError("max_analysed_s must be positive, 0 < hook_s <= max_analysed_s and context_margin_s >= 0")
        if self.mode not in ("off", "remote"):
            raise ValueError("gemini mode must be off or remote")
        if self.media_resolution not in ("low", "medium", "high"):
            raise ValueError("media_resolution must be low, medium or high")
        if self.fps <= 0 or self.timeout_s <= 0 or self.max_retries < 0 or self.max_upload_mb <= 0 or self.max_suggestions_per_segment < 1:
            raise ValueError("fps, timeout_s, max_upload_mb must be positive, max_retries >= 0 and max_suggestions_per_segment >= 1")

    @property
    def enabled(self) -> bool:
        return self.mode == "remote"

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("mode", "model", "fps", "media_resolution", "temperature", "max_suggestions_per_segment",
                                                                  "max_analysed_s", "context_margin_s", "hook_s")} | {
            "gemini_edit_version": GEMINI_EDIT_VERSION}


@dataclass(frozen=True)
class TribeConfig:
    """TRIBE v2 brain-response predictions from a remote Hugging Face Inference Endpoint (opt-in).

    The video is uploaded to the endpoint, so this needs ``allow_remote``, and videos
    under ``protected_roots`` (company reels) also need their id in ``approved_ids``. The endpoint
    URL and token come from the environment (``TRIBE_ENDPOINT_URL``, ``HF_TOKEN``) or the untracked
    ``.env``; they are never stored in artifacts."""

    enabled: bool = False
    allow_remote: bool = False
    approved_ids: tuple[str, ...] = ()
    protected_roots: tuple[str, ...] = ("01-Hookline",)
    language: str = "english"  # TRIBE v2's whisperX transcription language (upstream default; also en/fr/es/nl/zh)
    api: str = "endpoint"  # "endpoint": one synchronous POST (Inference Endpoint); "job": POST /analyze then poll GET /report/{id} (Space)
    job_wait_s: float = 3600.0  # job API: give up waiting for a finished report after this long. Not in the key
    poll_s: float = 10.0  # job API: seconds between report polls. Not in the key
    model_repo: str = TRIBE_MODEL_REPO
    model_revision: str = TRIBE_MODEL_REVISION
    timeout_s: float = 900.0  # per request; feature extraction (V-JEPA2, Llama 3.2, whisperX) is slow. Not in the key
    max_retries: int = 6  # on 429 / 5xx (503 while a scaled-to-zero endpoint starts). Not in the key
    retry_wait_s: float = 30.0  # first wait; doubles, capped at 120 s. Not in the key
    max_upload_mb: float = 50.0  # refuse larger files before sending

    def __post_init__(self) -> None:
        if self.language not in ("english", "french", "spanish", "dutch", "chinese"):
            raise ValueError("language must be english, french, spanish, dutch or chinese (TRIBE v2's whisperX languages)")
        if self.api not in ("endpoint", "job"):
            raise ValueError("api must be endpoint (synchronous handler) or job (POST /analyze, GET /report/{id})")
        if self.job_wait_s <= 0 or self.poll_s <= 0:
            raise ValueError("job_wait_s and poll_s must be positive")
        if self.timeout_s <= 0 or self.max_retries < 0 or self.retry_wait_s < 0 or self.max_upload_mb <= 0:
            raise ValueError("timeout_s and max_upload_mb must be positive; max_retries and retry_wait_s >= 0")

    @classmethod
    def from_env(cls, **overrides) -> "TribeConfig":
        values = {"enabled": env_flag("ENABLE_TRIBE", False)}
        return cls(**(values | {k: v for k, v in overrides.items() if v is not None}))

    def to_dict(self) -> dict:
        return asdict(self)

    def cache_dict(self) -> dict:
        return {"model_repo": self.model_repo, "model_revision": self.model_revision, "language": self.language,
                "tribe_version": TRIBE_VERSION}


def env_flag(name: str, default: bool = False) -> bool:
    """Boolean environment variable: 1/true/yes/on (any case) is True, 0/false/no/off/empty is False."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name}={value!r} is not a boolean (use true or false)")


@dataclass(frozen=True)
class TtccConfig:
    """TTCC retention inference (opt-in: ``ENABLE_TTCC=true`` or ``--enable-ttcc``).

    Inference runs ``scripts/ttcc_infer.py`` with ``python`` (the separate TTCC environment) and
    the upstream code in ``repo_dir``. The inference settings reproduce the protocol behind the
    checkpoint's published evaluation (``rl/srcc_eval.py`` and the V8 launch environment in
    ``go_viral``): 1 frame/s, at most 60 frames, 8192 video tokens, audio off, ``<cot></cot>``
    assistant span, one forward pass. Changing any of them changes the cache key."""

    enabled: bool = False
    python: str | None = None  # TTCC environment interpreter; None = .venv-ttcc/bin/python
    repo_dir: str | None = None  # go_viral checkout (ttcc-rl); None = <project>/go_viral
    checkpoint_repo: str = TTCC_CHECKPOINT_REPO
    checkpoint_revision: str = TTCC_CHECKPOINT_REVISION
    checkpoint_dir: str | None = None  # local copy of the checkpoint; None = Hugging Face cache
    allow_download: bool = False  # the checkpoint is ~9.4 GB; never fetched unless asked
    # "detect": CUDA if visible, else Apple MPS, else CPU. Or cuda / auto (split over several GPUs) / mps / cpu.
    device_map: str = "detect"
    torch_dtype: str = "bfloat16"
    # "detect": flash_attn (upstream eval) on CUDA GPUs that support FlashAttention 2, else sdpa.
    attn_impl: str = "detect"
    max_length: int = 32768
    fps: float = 1.0
    fps_max_frames: int = 60
    video_max_pixels: int = 49152
    max_pixels: int = 49152
    video_max_token_num: int = 8192
    use_audio_in_video: bool = False  # the V8 checkpoint was trained with audio off
    head_type: str = "hazard"
    # Videos longer than window_s: "windows" predicts overlapping windows and merges them into one curve over the whole video;
    # "first_60s" predicts only their first 60 s (the rest unpredicted); "skip" makes no prediction.
    long_videos: str = "hook"
    # long_videos="hook" (default): ONE pass over the opening, min(duration, hook_s) seconds, prompted with that length. R(t) is then a
    # native start-anchored prediction, so R(3) / R(5) / R(10) answer "does the opening hold?". Nothing is stitched.
    hook_s: int = 30
    # Sensitivity checks only: tell the model a different length (seconds) than it is shown. None = the seconds shown (the normal case).
    hook_prompt_s: int | None = None
    # Windowing (long_videos="windows"). 30 s is a memory limit of the deployment (L40S, sdpa attention), not of the model: the
    # head emits 60 values and the backbone has a 32768-token context. window_s can be raised up to 60 on a bigger GPU / FlashAttention.
    # window_mode "auto": window_s is the largest window (and the single-pass limit); a longer video gets a shorter window by its length
    # (see ttcc_windows.AUTO_TIERS: 20 s up to 45 s of video, 15 s up to 90 s, 10 s beyond). "fixed": every window is window_s.
    window_mode: str = "auto"
    window_s: int = 30
    stride_s: int = 15  # window starts are stride_s apart (scaled with the window size); window - stride seconds overlap
    warmup_s: float = 5.0  # a later window's first seconds are down-weighted in the merge (they describe a fresh start)
    min_window_s: int = 10  # smallest window the memory fallback may shrink to
    auto_shrink_window: bool = True  # out of memory / too many tokens: replan with a 30 % shorter window
    # Remote mode: the model runs in a Hugging Face Space (endpoint/ttcc_space/) instead of a local environment.
    # The video is uploaded, so like TRIBE it needs allow_remote, and company reels need their id approved.
    remote: bool = False
    allow_remote: bool = False
    approved_ids: tuple[str, ...] = ()
    protected_roots: tuple[str, ...] = ("01-Hookline",)
    max_upload_mb: float = 50.0
    poll_s: float = 15.0  # remote: seconds between job polls; not part of the cache key
    timeout_s: float = 7200.0  # whole runner call (model load + all videos), or per job when remote; not in the cache key
    verify_weights_sha256: bool = False  # also hash the 9.4 GB of shards (sizes are always checked); not in the key

    def __post_init__(self) -> None:
        if self.head_type != "hazard":
            raise ValueError("the v8 checkpoint has a hazard head; head_type must be hazard")
        if self.device_map not in DEVICE_MAPS:
            raise ValueError(f"device_map must be one of {', '.join(DEVICE_MAPS)}")
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {', '.join(ATTN_IMPLS)}")
        if self.long_videos not in ("hook", "windows", "first_60s", "skip"):
            raise ValueError("long_videos must be hook, windows, first_60s or skip")
        if not (5 <= self.hook_s <= 60) or (self.hook_prompt_s is not None and not (5 <= self.hook_prompt_s <= 60)):
            raise ValueError("hook_s and hook_prompt_s must be 5..60 s")
        if self.window_mode not in ("auto", "fixed"):
            raise ValueError("window_mode must be auto or fixed")
        if not (5 <= self.min_window_s <= self.window_s <= 60):
            raise ValueError("window sizes must satisfy 5 <= min_window_s <= window_s <= 60")
        if not (1 <= self.stride_s <= self.window_s) or self.warmup_s < 0:
            raise ValueError("stride_s must be between 1 and window_s, and warmup_s >= 0")
        if self.torch_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("torch_dtype must be bfloat16, float16 or float32")
        if self.fps <= 0 or self.fps_max_frames < 2 or self.max_length < 1 or self.timeout_s <= 0:
            raise ValueError("fps, fps_max_frames, max_length and timeout_s must be positive")

    @classmethod
    def from_env(cls, **overrides) -> "TtccConfig":
        """Defaults, then ENABLE_TTCC / TTCC_PYTHON / TTCC_REPO / TTCC_CHECKPOINT_DIR, then ``overrides``."""
        values = {"enabled": env_flag("ENABLE_TTCC", False), "python": os.environ.get("TTCC_PYTHON") or None,
                  "repo_dir": os.environ.get("TTCC_REPO") or None,
                  "checkpoint_dir": os.environ.get("TTCC_CHECKPOINT_DIR") or None}
        return cls(**(values | {k: v for k, v in overrides.items() if v is not None}))

    def to_dict(self) -> dict:
        return asdict(self)

    def runner_env(self) -> dict[str, str]:
        """Environment variables the upstream code reads (ms-swift, qwen_omni_utils, register.py)."""
        return {
            "FPS": str(self.fps), "FPS_MAX_FRAMES": str(self.fps_max_frames),
            "VIDEO_MAX_PIXELS": str(self.video_max_pixels), "MAX_PIXELS": str(self.max_pixels),
            "VIDEO_MAX_TOKEN_NUM": str(self.video_max_token_num),
            "USE_AUDIO_IN_VIDEO": "true" if self.use_audio_in_video else "false",
            "ENABLE_AUDIO_OUTPUT": "False", "RETENTION_HEAD_TYPE": self.head_type,
        }

    def inference_settings(self, device_map: str | None = None, attn_impl: str | None = None) -> dict:
        """Everything that can change a prediction (part of the cache key). Pass the resolved
        ``device_map`` / ``attn_impl`` when the configured ones are "detect"."""
        return {"checkpoint_repo": self.checkpoint_repo, "checkpoint_revision": self.checkpoint_revision,
                "torch_dtype": self.torch_dtype, "attn_impl": attn_impl or self.attn_impl,
                "device_map": device_map or self.device_map, "max_length": self.max_length, "env": self.runner_env()}


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
