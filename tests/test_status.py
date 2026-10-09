from __future__ import annotations

from hookline.status import ABSENT, AVAILABLE, FAILED, UNCERTAIN, expects_data, stream_status, window_status


def test_inspection_failure_is_uncertain_not_absent() -> None:
    assert stream_status(None, None) == UNCERTAIN
    assert stream_status(None, True) == UNCERTAIN


def test_confirmed_missing_stream_is_absent() -> None:
    assert stream_status(False, None) == ABSENT


def test_present_stream_extraction_outcomes() -> None:
    assert stream_status(True, True) == AVAILABLE
    assert stream_status(True, False) == FAILED
    assert stream_status(True, None) == UNCERTAIN


def test_window_status_passes_through_non_available_streams() -> None:
    for s in (ABSENT, FAILED, UNCERTAIN):
        assert window_status(s, 1.0, True, 0.5) == s


def test_window_status_coverage_rules() -> None:
    assert window_status(AVAILABLE, 1.0, True, 0.5) == AVAILABLE
    assert window_status(AVAILABLE, 0.3, True, 0.5) == UNCERTAIN
    assert window_status(AVAILABLE, 0.0, True, 0.5) == FAILED  # data expected but missing
    assert window_status(AVAILABLE, 0.0, False, 0.5) == ABSENT  # outside the stream's span


def test_expects_data_ignores_sliver_overlap_at_stream_end() -> None:
    assert expects_data(1.0, 1.5, [0.0, 3.0], 0.5)
    assert not expects_data(45.0, 45.4, [0.0, 45.044], 0.5)  # declared audio ends 0.04 s into the window
    assert expects_data(45.0, 45.4, [0.0, 45.3], 0.5)
    assert not expects_data(0.0, 0.5, None, 0.5)
