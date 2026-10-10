"""Window planning and merging of TTCC predictions. Curves here are SYNTHETIC test fixtures, not model results."""

from __future__ import annotations

import math

import pytest

from hookline.ttcc_windows import MIN_WEIGHT, hazards, merge_windows, plan_windows, summarize_curve, weight


def true_hazard(t: int) -> float:
    """A smooth, positive per-second hazard on the absolute timeline (second t covers (t-1, t])."""
    return 0.02 + 0.10 * math.exp(-t / 6.0) + 0.03 * math.sin(t / 7.0) ** 2


def curve_from(hz: list[float]) -> list[float]:
    r = [1.0]
    for h in hz:
        r.append(r[-1] * math.exp(-h))
    return r


def consistent_windows(plan: list[tuple[int, int]], start_artifact: float = 1.0, artifact_s: int = 0) -> list[dict]:
    """Window predictions that agree with the true hazard; later windows may carry an inflated hazard in their first seconds."""
    out = []
    for k, (s, e) in enumerate(plan):
        hz = [true_hazard(s + tau) * (start_artifact if (k > 0 and tau <= artifact_s) else 1.0) for tau in range(1, e - s + 1)]
        out.append({"start": s, "end": e, "retention": curve_from(hz)})
    return out


# ---------------------------------------------------------------- planning

@pytest.mark.parametrize("n,expected", [
    (5, [(0, 5)]), (25, [(0, 25)]), (29, [(0, 29)]), (30, [(0, 30)]),                  # shorter than / equal to the window: one pass
    (31, [(0, 30), (1, 31)]), (32, [(0, 30), (2, 32)]),                                # a short remainder becomes a shifted full window
    (45, [(0, 30), (15, 45)]), (46, [(0, 30), (15, 45), (16, 46)]),
    (60, [(0, 30), (15, 45), (30, 60)]),
    (83, [(0, 30), (15, 45), (30, 60), (45, 75), (53, 83)]),
])
def test_plan_covers_the_video_with_full_windows(n, expected) -> None:
    plan = plan_windows(n, 30, 15)
    assert plan == expected
    assert plan[0][0] == 0 and plan[-1][1] == n
    assert all(b[0] > a[0] for a, b in zip(plan, plan[1:]))               # strictly advancing
    assert all(a[1] > b[0] for a, b in zip(plan, plan[1:]))               # consecutive windows overlap, so no second is uncovered
    assert all(e - s == 30 for s, e in plan) or plan == [(0, n)]          # never a short final chunk


@pytest.mark.parametrize("window,stride", [(30, 15), (30, 10), (20, 5), (60, 30), (10, 10), (25, 1)])
def test_plan_properties_for_many_durations(window, stride) -> None:
    for n in range(5, 181):
        plan = plan_windows(n, window, stride)
        covered = set()
        for s, e in plan:
            assert 0 <= s < e <= n and e - s <= window and e - s >= 5
            covered.update(range(s + 1, e + 1))
        assert covered == set(range(1, n + 1)), (n, plan)


def test_plan_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="shorter"):
        plan_windows(4, 30, 15)
    with pytest.raises(ValueError, match="window_s"):
        plan_windows(40, 61, 15)
    with pytest.raises(ValueError, match="stride_s"):
        plan_windows(40, 30, 31)
    with pytest.raises(ValueError, match="stride_s"):
        plan_windows(40, 30, 0)


# ---------------------------------------------------------------- merging

@pytest.mark.parametrize("n", [25, 30, 31, 45, 60, 83, 120])
def test_consistent_windows_reproduce_the_underlying_curve_exactly(n) -> None:
    truth = curve_from([true_hazard(t) for t in range(1, n + 1)])
    merged = merge_windows(consistent_windows(plan_windows(n, 30, 15)), n, warmup_s=5)
    assert merged["covered_until"] == n and len(merged["retention"]) == n + 1
    assert max(abs(a - b) for a, b in zip(merged["retention"], truth)) < 1e-12  # timestamps line up second for second
    assert merged["diagnostics"]["mean_abs_hazard_difference"] in (None, pytest.approx(0.0, abs=1e-12))


def test_merged_curve_is_continuous_monotone_and_anchored() -> None:
    n = 83
    merged = merge_windows(consistent_windows(plan_windows(n, 30, 15), start_artifact=6.0, artifact_s=3), n, warmup_s=5)
    r = merged["retention"]
    assert r[0] == 1.0 and all(b <= a + 1e-12 for a, b in zip(r, r[1:])) and all(v > 0 for v in r)
    drops = [a - b for a, b in zip(r, r[1:])]
    assert max(drops) == drops[0] or max(drops) < 0.2  # no one-second cliff apart from the genuine opening
    # no boundary jump: at each window start the error left by a x6 artifact is at most 40% of what an unsuppressed window (5 x hazard) would leave
    for s, _ in plan_windows(n, 30, 15)[1:]:
        for t in range(s + 1, s + 4):
            assert abs(merged["hazard"][t - 1] - true_hazard(t)) < 0.4 * 5.0 * true_hazard(t)


def test_the_start_of_window_artifact_is_suppressed_better_than_naive_chaining() -> None:
    """Each later window predicts a steep 'ad opening' (hazard x6 for 3 s). Chaining windows as-is would put it at every boundary."""
    n = 83
    plan = plan_windows(n, 30, 15)
    windows = consistent_windows(plan, start_artifact=6.0, artifact_s=3)
    merged = merge_windows(windows, n, warmup_s=5)
    flat = merge_windows(windows, n, warmup_s=0)  # warm-up off: every window counts fully, so the artifact leaks in
    err = lambda m: sum(abs(m["hazard"][t - 1] - true_hazard(t)) for s, _ in plan[1:] for t in range(s + 1, s + 4))  # noqa: E731
    assert err(merged) < 0.5 * err(flat)
    # and the weighting never reaches zero, so a window can still stand alone when it is the only one covering a second
    assert weight(1, 1, 5.0) == pytest.approx(max(MIN_WEIGHT, 0.04)) and weight(1, 0, 5.0) == 1.0 and weight(9, 3, 5.0) == 1.0


def test_a_failed_window_ends_the_curve_at_the_gap_and_nothing_is_invented() -> None:
    n = 83
    plan = plan_windows(n, 30, 15)
    ws = consistent_windows(plan)
    ok = merge_windows([w for i, w in enumerate(ws) if i != 2], n)          # window (30, 60) missing: seconds 46..? are still covered by (15, 45) ... then (45, 75)
    assert ok["covered_until"] == n                                         # the overlaps bridge a single missing window
    gap = merge_windows([ws[0], ws[1], ws[4]], n)                           # (0,30),(15,45),(53,83): seconds 46..53 nobody covers
    assert gap["covered_until"] == 45 and len(gap["retention"]) == 46
    first_missing = merge_windows(ws[1:], n)
    assert first_missing["covered_until"] == 0 and first_missing["retention"] == [1.0]  # nothing anchors R(0) = 1
    assert merge_windows([], n)["covered_until"] == 0


def test_hazard_spike_lands_on_the_right_absolute_second() -> None:
    n = 60
    plan = plan_windows(n, 30, 15)
    windows = []
    for s, e in plan:
        hz = [0.01 + (0.5 if s + tau == 40 else 0.0) for tau in range(1, e - s + 1)]  # an event at absolute second 40 (interval 39..40)
        windows.append({"start": s, "end": e, "retention": curve_from(hz)})
    m = merge_windows(windows, n, warmup_s=5)
    peak = max(range(len(m["hazard"])), key=lambda i: m["hazard"][i])
    assert peak + 1 == 40 and m["hazard"][peak] > 0.4                        # second 40 covers (39, 40]
    assert m["retention"][39] - m["retention"][40] == max(a - b for a, b in zip(m["retention"], m["retention"][1:]))


def test_windows_that_disagree_are_reported() -> None:
    n = 45
    a, b = consistent_windows(plan_windows(n, 30, 15))
    b = {"start": b["start"], "end": b["end"], "retention": curve_from([h * 2.0 for h in hazards(b["retention"])])}
    d = merge_windows([a, b], n)["diagnostics"]
    assert d["overlap_pairs"][0]["seconds_compared"] == 10 and d["mean_abs_hazard_difference"] > 0.01
    assert merge_windows([a, {"start": b["start"], "end": b["end"], "retention": b["retention"]}], n, warmup_s=5)["covered_until"] == n


def test_bad_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="positive"):
        hazards([1.0, 0.0])
    with pytest.raises(ValueError, match="hazards for"):
        merge_windows([{"start": 0, "end": 10, "retention": [1.0, 0.9, 0.8]}], 10)


def test_summary_matches_the_single_pass_fields() -> None:
    r = curve_from([0.5, 0.1, 0.05, 0.02, 0.02, 0.02])
    s = summarize_curve(r)
    assert s["steepest_one_second_drop"]["from_s"] == 0 and s["retention_at_horizon"] == r[-1] and set(s["retention_at_s"]) == {"1", "3", "5"}
