from __future__ import annotations

import pytest

from hookline.segments import build_segments


def assert_contiguous(segs, duration: float) -> None:
    assert segs[0].start == 0.0
    assert segs[-1].end == pytest.approx(duration)
    for a, b in zip(segs, segs[1:]):
        assert a.end == b.start
    assert all(s.duration > 0 for s in segs)
    assert [s.index for s in segs] == list(range(len(segs)))


@pytest.mark.parametrize("duration", [None, 0, -1.0])
def test_no_duration_gives_no_segments(duration) -> None:
    assert build_segments(duration, 0.5, 0.2) == []


def test_exact_grid() -> None:
    segs = build_segments(2.0, 0.5, 0.2)
    assert [(s.start, s.end) for s in segs] == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    assert not any(s.is_partial for s in segs)
    assert segs[0].cut_at_start is None  # no previous segment to compare with


def test_partial_final_segment_kept_when_long_enough() -> None:
    segs = build_segments(2.3, 0.5, 0.2)
    assert_contiguous(segs, 2.3)
    assert (segs[-1].start, segs[-1].end) == (2.0, pytest.approx(2.3))
    assert segs[-1].is_partial


def test_tiny_final_piece_merges_instead_of_creating_a_sliver() -> None:
    segs = build_segments(2.1, 0.5, 0.2)
    assert_contiguous(segs, 2.1)
    assert (segs[-1].start, segs[-1].end) == (1.5, pytest.approx(2.1))
    assert segs[-1].is_partial


@pytest.mark.parametrize("duration", [0.05, 0.3, 0.4])
def test_clip_shorter_than_one_window(duration: float) -> None:
    segs = build_segments(duration, 0.5, 0.2)
    assert len(segs) == 1
    assert (segs[0].start, segs[0].end) == (0.0, duration)
    assert segs[0].is_partial


def test_scene_cut_becomes_boundary_and_increments_scene() -> None:
    segs = build_segments(3.0, 0.5, 0.2, cut_times=[1.3])
    assert_contiguous(segs, 3.0)
    starts = [s.start for s in segs]
    assert 1.3 in starts
    cut_seg = segs[starts.index(1.3)]
    assert cut_seg.cut_at_start is True and cut_seg.scene_index == 1
    assert all(s.scene_index == 0 for s in segs if s.start < 1.3)
    assert all(s.scene_index == 1 for s in segs if s.start >= 1.3)


def test_cut_near_grid_point_does_not_create_sliver() -> None:
    segs = build_segments(3.0, 0.5, 0.2, cut_times=[1.45])
    assert_contiguous(segs, 3.0)
    assert min(s.duration for s in segs) >= 0.2 - 1e-9
    assert 1.45 in [s.start for s in segs]  # the cut wins over the grid point


def test_close_cuts_keep_first_and_remember_both() -> None:
    segs = build_segments(3.0, 0.5, 0.2, cut_times=[1.2, 1.3])
    assert_contiguous(segs, 3.0)
    starts = [s.start for s in segs]
    assert 1.2 in starts and 1.3 not in starts
    seg = segs[starts.index(1.2)]
    assert 1.3 in seg.cut_times and 1.2 in seg.cut_times


def test_cut_at_zero_or_end_is_ignored_as_boundary() -> None:
    segs = build_segments(2.0, 0.5, 0.2, cut_times=[0.0, 2.0])
    assert_contiguous(segs, 2.0)
    assert len(segs) == 4


def test_many_cuts_never_create_empty_segments() -> None:
    cuts = [round(0.07 * i, 2) for i in range(1, 40)]
    segs = build_segments(3.0, 0.5, 0.2, cut_times=cuts)
    assert_contiguous(segs, 3.0)
