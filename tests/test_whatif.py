"""What-if edit support: trim arithmetic, ffmpeg application on tiny generated clips, and the predicted-curve comparison."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hookline import whatif
from hookline.whatif import apply_trims, compare_curves, edited_time, keep_segments, normalise_trims

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")


def make_clip(path: Path, seconds: int = 12, audio: bool = True) -> Path:
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def probe(path: Path) -> tuple[float, list[str]]:
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_entries", "format=duration:stream=codec_type", str(path)],
                       check=True, capture_output=True, text=True)
    info = json.loads(p.stdout)
    return float(info["format"]["duration"]), [s["codec_type"] for s in info["streams"]]


def tr(a, b):
    return {"start_s": a, "end_s": b}


# ---- normalise_trims ----

def test_normalise_sorts_clips_and_merges_overlaps():
    ranges, warns = normalise_trims([tr(8, 10), tr(1, 3), tr(2.5, 4), tr(-5, 0.5), tr(11, 30)], 12.0, min_keep_s=1.0)
    assert ranges == [(0.0, 0.5), (1.0, 4.0), (8.0, 10.0), (11.0, 12.0)]
    assert any("merged" in w for w in warns) and any("clipped" in w for w in warns)


def test_normalise_adjacent_ranges_are_merged():
    ranges, _ = normalise_trims([tr(1, 2), tr(2, 3), tr(3, 4)], 20.0)
    assert ranges == [(1.0, 4.0)]


def test_normalise_drops_zero_length_outside_and_malformed_with_warnings():
    ranges, warns = normalise_trims([tr(5, 5), tr(50, 60), {"start_s": "x", "end_s": 3}, "nope", tr(2, 4)], 20.0)
    assert ranges == [(2.0, 4.0)]
    assert len(warns) == 4
    assert any("zero length" in w for w in warns) and any("outside the video" in w for w in warns)


def test_normalise_accepts_alternative_keys_and_swaps_reversed():
    ranges, warns = normalise_trims([{"start": 6, "end": 4}], 20.0)
    assert ranges == [(4.0, 6.0)] and any("swapped" in w for w in warns)


@pytest.mark.parametrize("trims,duration,kwargs,message", [
    ([], 20.0, {}, "no valid trim"),
    ([tr(30, 40)], 20.0, {}, "no valid trim"),
    ([tr(0, 17)], 20.0, {}, "limit"),                           # 85 % > 80 %
    ([tr(0, 8)], 10.0, {"max_remove_fraction": 0.95}, "at least 5"),  # 2 s would remain
    ([tr(1, 2)], 0.0, {}, "duration"),
])
def test_normalise_raises_clear_errors(trims, duration, kwargs, message):
    with pytest.raises(ValueError, match=message):
        normalise_trims(trims, duration, **kwargs)


def test_normalise_boundaries_are_allowed():
    ranges, _ = normalise_trims([tr(0, 16)], 20.0, min_keep_s=4.0)  # exactly 80 % removed, 4 s kept
    assert ranges == [(0.0, 16.0)]
    ranges, _ = normalise_trims([tr(0, 15)], 20.0)  # 5 s kept, 75 % removed
    assert ranges == [(0.0, 15.0)]


# ---- keep_segments / edited_time ----

def test_keep_segments_is_the_complement():
    assert keep_segments(12.0, [(2.0, 4.0), (8.0, 9.0)]) == [(0.0, 2.0), (4.0, 8.0), (9.0, 12.0)]
    assert keep_segments(12.0, [(0.0, 3.0)]) == [(3.0, 12.0)]
    assert keep_segments(12.0, [(9.0, 12.0)]) == [(0.0, 9.0)]
    assert keep_segments(12.0, []) == [(0.0, 12.0)]
    assert keep_segments(12.0, [(0.0, 12.0)]) == []


def test_edited_time_maps_across_removed_ranges():
    trims = [(2.0, 4.0), (8.0, 9.0)]
    assert edited_time(1.0, trims) == 1.0
    assert edited_time(2.0, trims) is None
    assert edited_time(3.0, trims) is None
    assert edited_time(4.0, trims) == 2.0     # first frame after the cut
    assert edited_time(7.5, trims) == 5.5
    assert edited_time(8.5, trims) is None
    assert edited_time(9.0, trims) == 6.0
    assert edited_time(11.0, trims) == 8.0
    assert edited_time(-1.0, trims) is None


def test_edited_time_matches_keep_segment_lengths():
    trims = [(1.0, 2.5), (6.0, 7.0)]
    kept = keep_segments(10.0, trims)
    assert sum(b - a for a, b in kept) == pytest.approx(7.5)
    assert edited_time(10.0, trims) == pytest.approx(7.5)


# ---- apply_trims ----

@needs_ffmpeg
@pytest.mark.parametrize("audio", [True, False])
def test_apply_trims_round_trip(tmp_path, audio):
    src = make_clip(tmp_path / "src.mp4", 12, audio)
    out = apply_trims(src, [(2.0, 4.0), (8.0, 9.0)], tmp_path / "sub" / "out.mp4")
    duration, kinds = probe(out)
    assert out == tmp_path / "sub" / "out.mp4"
    assert duration == pytest.approx(9.0, abs=0.3)
    assert ("audio" in kinds) is audio and "video" in kinds
    assert list(out.parent.glob("*.part*")) == []


@needs_ffmpeg
def test_apply_trims_single_kept_segment_and_leading_trim(tmp_path):
    src = make_clip(tmp_path / "src.mp4", 12, audio=False)
    out = apply_trims(src, [(0.0, 5.0)], tmp_path / "out.mp4")
    duration, kinds = probe(out)
    assert duration == pytest.approx(7.0, abs=0.3) and kinds == ["video"]


@needs_ffmpeg
def test_apply_trims_tail_trim_keeps_pixel_format(tmp_path):
    src = make_clip(tmp_path / "src.mp4", 12)
    out = apply_trims(src, [(9.0, 12.0)], tmp_path / "out.mp4")
    assert probe(out)[0] == pytest.approx(9.0, abs=0.3)
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(out)],
                       check=True, capture_output=True, text=True)
    assert p.stdout.strip() == "h264,yuv420p"


def _fake_runner(fail_at: str, calls: list):
    """ffprobe succeeds (video + audio, 12 s); ffmpeg writes a partial output file and then fails."""
    def run(cmd, **kw):
        calls.append(cmd)
        if "ffprobe" in Path(cmd[0]).name:
            out = json.dumps({"format": {"duration": "12.0"}, "streams": [{"codec_type": "video"}, {"codec_type": "audio"}]})
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        Path(cmd[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stdout="", stderr="line1\nline2\n" + fail_at)
    return run


def test_apply_trims_failure_raises_and_leaves_no_part_file(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    out = tmp_path / "out.mp4"
    out.write_bytes(b"previous")
    calls: list = []
    with pytest.raises(RuntimeError, match="Invalid data found"):
        apply_trims(src, [(2.0, 4.0)], out, run=_fake_runner("Invalid data found when processing input", calls))
    assert out.read_bytes() == b"previous"              # the earlier output is untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4", "src.mp4"]
    ffmpeg_cmd = calls[-1]
    graph = ffmpeg_cmd[ffmpeg_cmd.index("-filter_complex") + 1]
    assert "atrim" in graph and "concat=n=2:v=1:a=1" in graph       # audio probed as present: audio filters and aac were requested
    assert ffmpeg_cmd[ffmpeg_cmd.index("-c:a") + 1] == "aac"


def test_apply_trims_probe_failure_is_a_runtime_error(tmp_path):
    def run(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="moov atom not found")
    with pytest.raises(RuntimeError, match="moov atom not found"):
        apply_trims(tmp_path / "x.mp4", [(1.0, 2.0)], tmp_path / "out.mp4", run=run)
    assert list(tmp_path.iterdir()) == []


def test_apply_trims_no_audio_graph_has_no_audio_filters(tmp_path):
    calls: list = []

    def run(cmd, **kw):
        calls.append(cmd)
        if "ffprobe" in Path(cmd[0]).name:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"format": {"duration": "12"}, "streams": [{"codec_type": "video"}]}), stderr="")
        Path(cmd[-1]).write_bytes(b"data")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    out = apply_trims(tmp_path / "x.mp4", [(1.0, 2.0)], tmp_path / "out.mp4", run=run)
    graph = calls[-1][calls[-1].index("-filter_complex") + 1]
    assert out.read_bytes() == b"data" and "atrim" not in graph and "a=0" in graph and "-an" in calls[-1]
    assert not list(tmp_path.glob("*.part*"))


# ---- compare_curves ----

def curve(n: int = 31, start: float = 1.0, step: float = 0.02) -> list[float]:
    return [max(start - step * i, 0.01) for i in range(n)]


TRIMS = [(2.0, 4.0)]


def metric(res: dict, label: str) -> dict:
    return next(m for m in res["metrics"] if m["label"] == label)


def test_compare_higher_verdict_and_metrics():
    orig, edited = curve(), curve(step=0.015)
    res = compare_curves(orig, edited, TRIMS)
    assert res["schema"] == "hookline.whatif" and res["trims"] == [[2.0, 4.0]] and res["removed_s"] == 2.0
    assert (res["original_duration_s"], res["edited_duration_s"]) == (30, 30)
    assert res["verdict"] == "edited curve holds higher"
    m3 = metric(res, "predicted hold at 3 s")
    assert m3["original"] == pytest.approx(0.94) and m3["edited"] == pytest.approx(0.955)
    assert m3["delta"] == pytest.approx(0.015) and m3["within_noise"] is False
    assert {m["label"] for m in res["metrics"]} == {"predicted hold at 3 s", "predicted hold at 5 s", "predicted hold at 10 s",
                                                    "predicted retention at the end of each video", "mean predicted hazard per second"}
    assert metric(res, "mean predicted hazard per second")["delta"] < 0   # a gentler fall per second


def test_compare_lower_verdict():
    res = compare_curves(curve(step=0.015), curve(), TRIMS)
    assert res["verdict"] == "edited curve holds lower"
    assert all(m["delta"] < 0 for m in res["metrics"] if "hold" in m["label"])


def test_compare_within_noise_is_no_clear_difference():
    orig = curve()
    edited = [v + 0.002 for v in orig]
    res = compare_curves(orig, edited, TRIMS)
    assert res["verdict"] == "no clear difference"
    assert all(m["within_noise"] for m in res["metrics"] if "hold" in m["label"])


def test_compare_between_floor_and_verdict_bar_is_not_a_verdict():
    orig = curve()
    edited = [v + 0.007 for v in orig]      # above the noise floor (0.0033) but below 3x (0.0099)
    res = compare_curves(orig, edited, TRIMS)
    assert res["verdict"] == "no clear difference"
    assert metric(res, "predicted hold at 3 s")["within_noise"]          # the table uses the same bar as the verdict (3x the floor), so it never contradicts it
    assert not metric(compare_curves(orig, [v + 0.015 for v in orig], TRIMS), "predicted hold at 3 s")["within_noise"]


def test_compare_needs_a_majority_of_hold_metrics():
    orig = curve()
    edited = list(orig)
    edited[3] += 0.05                       # only R(3) moves clearly; R(5), R(10) are identical
    res = compare_curves(orig, edited, TRIMS)
    assert res["verdict"] == "no clear difference"


def test_compare_holds_present_only_when_both_curves_reach_the_second():
    res = compare_curves(curve(31), curve(8), TRIMS)
    labels = [m["label"] for m in res["metrics"]]
    assert "predicted hold at 3 s" in labels and "predicted hold at 5 s" in labels and "predicted hold at 10 s" not in labels
    assert res["edited_duration_s"] == 7


def test_compare_curve_too_short_for_any_hold_says_so():
    res = compare_curves(curve(31), curve(3), TRIMS)
    assert res["verdict"] == "no clear difference"
    assert not any("hold at" in m["label"] for m in res["metrics"])
    assert any("no verdict" in c for c in res["caveats"])


def test_compare_caveats_always_present():
    for res in (compare_curves(curve(), curve(step=0.015), TRIMS), compare_curves(curve(), curve(), []), compare_curves([], [], TRIMS)):
        text = " ".join(res["caveats"])
        assert "not audience data" in text
        assert "12x" in text and "2x" in text and "order of events" in text
        assert "this video only" in text and "nothing is known about" in text
        assert "approximate" not in text


def test_compare_stitched_caveat_only_with_a_flag():
    assert not any("approximate" in c for c in compare_curves(curve(), curve(), TRIMS)["caveats"])
    for kw in ({"original_is_stitched": True}, {"edited_is_stitched": True}):
        assert any("approximate" in c for c in compare_curves(curve(), curve(), TRIMS, **kw)["caveats"])


def test_compare_custom_noise_floor_changes_verdict():
    orig = curve()
    edited = [v + 0.007 for v in orig]
    assert compare_curves(orig, edited, TRIMS, noise_floor=0.001)["verdict"] == "edited curve holds higher"


def test_compare_result_is_json_serialisable_and_module_has_no_network_import():
    res = compare_curves(curve(), curve(step=0.015), [(1.0, 2.5)])
    assert json.loads(json.dumps(res)) == res
    src = Path(whatif.__file__).read_text()
    assert "requests" not in src and "urllib" not in src and "socket" not in src
