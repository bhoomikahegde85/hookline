"""Modality status vocabulary.

A missing measurement is never a negative measurement. Every segment records,
per modality, which of these applies:

- ``available``: measured from data that covers the segment.
- ``absent``: the file has no such stream (confirmed by inspection), or the
  stream does not extend over this segment.
- ``failed``: the stream exists and should cover the segment, but extraction or
  decoding failed.
- ``uncertain``: presence could not be determined, or the measurement rests on
  partial coverage and should be treated with caution.
- ``not_applicable``: the measurement is undefined here (e.g. frame-to-frame
  change for the very first frame).
"""

from __future__ import annotations

AVAILABLE = "available"
ABSENT = "absent"
FAILED = "failed"
UNCERTAIN = "uncertain"
NOT_APPLICABLE = "not_applicable"

ALL_STATUSES = (AVAILABLE, ABSENT, FAILED, UNCERTAIN, NOT_APPLICABLE)

# Values are only reported for these statuses; all others leave values empty.
MEASURED_STATUSES = frozenset({AVAILABLE, UNCERTAIN})


def stream_status(has_stream: bool | None, extraction_ok: bool | None) -> str:
    """File-level status of a modality from inspection and extraction outcomes.

    ``has_stream`` is None when inspection could not tell (never treated as absent).
    ``extraction_ok`` is None when extraction was not attempted.
    """
    if has_stream is None:
        return UNCERTAIN
    if has_stream is False:
        return ABSENT
    if extraction_ok is False:
        return FAILED
    if extraction_ok is None:
        return UNCERTAIN
    return AVAILABLE


def expects_data(start: float, end: float, span: list[float] | None, min_coverage: float) -> bool:
    """True if the stream's declared span covers at least ``min_coverage`` of the segment.

    Containers often declare a few tenths of a second more than decodes (e.g. AAC
    padding), so a sliver of overlap at the end is not an expectation of data.
    """
    if not span or end <= start:
        return False
    return max(0.0, min(end, span[1]) - max(start, span[0])) >= min_coverage * (end - start) - 1e-9


def window_status(stream: str, coverage: float | None, within_expected_span: bool, min_coverage: float) -> str:
    """Segment-level status given the file-level stream status and data coverage.

    ``coverage`` is the fraction of the segment backed by decoded data.
    ``within_expected_span`` says whether the stream's declared duration covers the segment
    (see ``expects_data``).
    """
    if stream != AVAILABLE:
        return stream
    if not coverage:
        return FAILED if within_expected_span else ABSENT
    if coverage < min_coverage:
        return UNCERTAIN
    return AVAILABLE
