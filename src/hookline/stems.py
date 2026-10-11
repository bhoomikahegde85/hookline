"""Music and voice levels from Demucs stems (``stems.json``), measured on the opening seconds.

Demucs ``htdemucs`` splits the audio into drums, bass, other and vocals. "Music" here is drums + bass + other; "voice" is the
vocals stem, which also holds sung lyrics, so a song with singing counts as voice. Levels are RMS in dBFS per second.
Descriptive only: a louder music bed is not "more engaging". Nothing here is a retention or engagement measure.
"""

from __future__ import annotations

import numpy as np

STEMS_FILE = "stems.json"
FLOOR_DB = -80.0
MUSIC_STEMS = ("drums", "bass", "other")
VOICE_STEMS = ("vocals",)
PRESENT_DB = -45.0  # a stem quieter than this over the window is treated as absent


def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return FLOOR_DB
    return float(max(FLOOR_DB, 20.0 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)))


def per_second_db(wave: np.ndarray, sr: int, seconds: int) -> list[float]:
    """RMS dBFS of each whole second of a (channels, samples) or (samples,) array."""
    mono = wave.mean(axis=0) if wave.ndim == 2 else wave
    return [rms_db(mono[s * sr:(s + 1) * sr]) for s in range(seconds)]


def summarise(stems: dict[str, np.ndarray], sr: int, window_s: int = 5, max_s: int = 10) -> dict:
    """``stems``: name -> (channels, samples). Music/voice per-second levels plus aggregates over the first ``window_s`` seconds."""
    n = min(int(min(s.shape[-1] for s in stems.values()) // sr), max_s)
    if n < 1:
        return {"status": "unavailable", "reason": "less than one second of audio"}
    mix = lambda names: sum(stems[k].mean(axis=0) if stems[k].ndim == 2 else stems[k] for k in names if k in stems)  # noqa: E731
    music, voice = mix(MUSIC_STEMS), mix(VOICE_STEMS)
    m_t, v_t = per_second_db(music, sr, n), per_second_db(voice, sr, n)
    w = min(window_s, n)
    m_w, v_w = rms_db(music[:w * sr]), rms_db(voice[:w * sr])
    return {"status": "ok", "analysed_s": n, "window_s": w, "per_second": {"music_dbfs": [round(x, 1) for x in m_t], "voice_dbfs": [round(x, 1) for x in v_t]},
            f"music_rms_dbfs_0_{w}": round(m_w, 1), f"voice_rms_dbfs_0_{w}": round(v_w, 1),
            f"music_minus_voice_db_0_{w}": round(m_w - v_w, 1),
            "music_present": m_w > PRESENT_DB, "voice_present": v_w > PRESENT_DB,
            "note": "music = drums + bass + other; voice = the vocals stem (sung lyrics count as voice). Levels are descriptive, not engagement measures."}
