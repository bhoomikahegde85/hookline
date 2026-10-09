"""Timeline segmentation: fixed windows that respect scene cuts.

Boundaries are the fixed grid (k * window) plus detected cut times. Any piece
shorter than ``min_segment`` is merged by removing an adjacent *grid* boundary,
so segments never straddle a cut. A short trailing remainder merges into the
previous segment. No segment is ever empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

EPS = 1e-6
GRID, CUT = "grid", "cut"


@dataclass
class Segment:
    index: int
    start: float
    end: float
    scene_index: int
    cut_at_start: bool | None  # None for the first segment (no preceding frame)
    cut_times: list[float] = field(default_factory=list)  # cuts inside [start, end), incl. merged ones
    is_partial: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_segments(
    duration: float | None,
    window: float,
    min_segment: float,
    cut_times: Iterable[float] = (),
) -> list[Segment]:
    if duration is None or not math.isfinite(duration) or duration <= EPS:
        return []
    if window <= 0:
        raise ValueError("window must be positive")
    duration = round(duration, 6)

    kinds: dict[float, str] = {}
    k = 1
    while k * window < duration - EPS:
        kinds[round(k * window, 6)] = GRID
        k += 1
    all_cuts = sorted(round(c, 6) for c in cut_times if EPS < c < duration - EPS)
    for c in all_cuts:
        kinds[c] = CUT  # a cut on a grid point overrides it

    points = [0.0] + sorted(kinds) + [duration]
    i = 0
    while i < len(points) - 1:
        if len(points) <= 2 or points[i + 1] - points[i] >= min_segment - EPS:
            i += 1
            continue
        left_kind = kinds.get(points[i]) if i > 0 else None
        right_kind = kinds.get(points[i + 1]) if i + 1 < len(points) - 1 else None
        if right_kind == GRID:
            drop = points[i + 1]
        elif left_kind == GRID:
            drop = points[i]
        elif left_kind == CUT and right_kind == CUT:
            drop = points[i + 1]  # two cuts closer than min_segment: keep one boundary, remember both
        else:
            i += 1  # a short scene at the very start or end, bounded only by a cut: keep it
            continue
        points.remove(drop)
        kinds.pop(drop, None)
        i = max(i - 1, 0)

    segments: list[Segment] = []
    scene = 0
    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        is_cut = kinds.get(start) == CUT
        if is_cut:
            scene += 1
        segments.append(
            Segment(
                index=i,
                start=start,
                end=end,
                scene_index=scene,
                cut_at_start=None if i == 0 else is_cut,
                cut_times=[c for c in all_cuts if start - EPS <= c < end - EPS],
                is_partial=abs((end - start) - window) > 1e-3,
            )
        )
    return segments


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))
