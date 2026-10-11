import numpy as np

from hookline.stems import FLOOR_DB, rms_db, summarise

SR = 8000


def tone(amp, seconds=6):
    return amp * np.sin(2 * np.pi * 220 * np.arange(SR * seconds) / SR)[None, :].repeat(2, 0)


def test_music_louder_than_voice_gives_a_positive_difference() -> None:
    stems = {"drums": tone(0.2), "bass": tone(0.2), "other": tone(0.2), "vocals": tone(0.02)}
    r = summarise(stems, SR)
    assert r["status"] == "ok" and r["analysed_s"] == 6 and r["window_s"] == 5
    assert r["music_minus_voice_db_0_5"] > 10 and r["music_present"] and r["voice_present"]
    assert len(r["per_second"]["music_dbfs"]) == 6


def test_absent_voice_is_reported_absent_not_zero() -> None:
    z = np.zeros((2, SR * 5))
    r = summarise({"drums": tone(0.1, 5), "bass": z, "other": z, "vocals": z}, SR)
    assert r["voice_present"] is False and r["voice_rms_dbfs_0_5"] <= -60 and r["music_present"]


def test_too_short_audio_is_unavailable_and_silence_hits_the_floor() -> None:
    assert summarise({"vocals": np.zeros((2, 100))}, SR)["status"] == "unavailable"
    assert rms_db(np.zeros(10)) == FLOOR_DB or rms_db(np.zeros(10)) < -79
