"""Segment-level editing assistant for one video (``review.html``).

One self-contained HTML file: no server, no external script or font, nothing is uploaded. The video plays from
its local path (``file://``), so open the page on the machine that has the file. Features:
- a video player synchronised with a clickable timeline strip and the predicted retention curve (a moving playhead,
  the active segment highlighted);
- a card per candidate segment: timestamps, what was measured in it, and rule-based edit proposals (trim / add / review);
- per proposal: jump to the timestamp, edit the trim range, preview the trim (the video skips the range), and decide
  accept / reject / unsure with a note;
- "play edited version": the video skips every accepted trim;
- export of all decisions as JSON or CSV (kept in the browser until exported; nothing is sent anywhere).

The curve is a pretrained model's PREDICTION (TTCC), not actual audience data, and its accuracy on these videos has not
been verified. The proposals are unvalidated rule-based hypotheses over measured features. All data is rendered with
``textContent`` (never as HTML), so text read from the video cannot inject markup.
"""

from __future__ import annotations

import json
from pathlib import Path

from .brain_networks import brain_view
from .segment_recommendations import build_segment_recommendations
from .ttcc import hook_metrics

REVIEW_FILE = "review.html"
REFERENCE_CURVES = Path(__file__).resolve().parents[2] / "artifacts" / "hook" / "features"  # TTCC predictions of the 100+ other videos
REFERENCE_CURVES_LONG = Path(__file__).resolve().parents[2] / "artifacts" / "hook_long" / "features"  # same reels, native 30 s pass
BAND_MIN_REELS = 30
# TTCC's prediction for a video whose frames are all black (the first 10 s). Measured on 3 reels in the sanity run 20261010T183353Z; the three
# curves were identical, so it is the model's content-free baseline. A real video's curve is read against it: above means TTCC sees content it likes.
BLANK_VIDEO_CURVE = [1.0, 0.3655, 0.2135, 0.17, 0.1381, 0.1196, 0.1014, 0.0883, 0.078, 0.0717, 0.0654]


def reference_band(root: Path | None = None) -> dict | None:
    """Per-second 10th / 50th / 90th percentile of the other videos' predicted retention (native pass), where at least BAND_MIN_REELS
    reels have a value. TTCC curves are close to one shared decay shape, so a video's curve is read against this band."""
    import numpy as np
    def read(r: Path) -> list:
        out = []
        for f in sorted(r.glob("*/ttcc_retention.json")) if r.is_dir() else []:
            a = _load(f)
            if a and a.get("status") == "success" and a.get("retention"):
                out.append(a["retention"])
        return out
    # the longer (stitched, to ~30 s) reference curves when they exist, else the 10 s native hook curves
    curves = read(REFERENCE_CURVES_LONG) if root is None else []
    if len(curves) < BAND_MIN_REELS:
        curves = read(REFERENCE_CURVES if root is None else root)
    if len(curves) < BAND_MIN_REELS:
        return None
    t_max = 0
    while sum(len(c) > t_max + 1 for c in curves) >= BAND_MIN_REELS:
        t_max += 1
    cols = [[c[t] for c in curves if len(c) > t] for t in range(t_max + 1)]
    q = lambda p: [round(float(np.percentile(v, p)), 4) for v in cols]  # noqa: E731
    return {"n_reels": len(curves), "p10": q(10), "p50": q(50), "p90": q(90), "blank": BLANK_VIDEO_CURVE,
            "note": "10th-90th percentile and median of TTCC's predicted retention on the other videos (one native pass over the first 30 s); predictions, not audience data"}


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _embed(data: dict) -> str:
    """JSON safe inside a <script> tag."""
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


def curve_audit(art: dict, video_dir: Path) -> dict:
    """Facts about the displayed curve, so what processing was (not) applied can be shown: nothing is smoothed or normalised here."""
    import numpy as np
    r = np.asarray(art.get("retention") or [], float)
    ts = art.get("timestamps_s") or []
    out = {"displayed_curve": "TTCC output as returned: Hookline applies no smoothing or rescaling (R(0)=1 is prepended by convention; raw values are in raw_file)",
           "points": int(len(r)), "grid_s": 1.0, "timestamps_are_whole_seconds_from_0": ts == list(range(len(r))),
           "non_increasing": bool((np.diff(r) <= 1e-9).all()) if len(r) > 1 else None,
           "values_in_0_1": bool(((r >= 0) & (r <= 1)).all()) if len(r) else None,
           "raw_file": "ttcc_raw.json" if (video_dir / "ttcc_raw.json").is_file() else None,
           "merge": "10 s windows every 5 s merged in hazard space (approximate)" if art.get("stitched") else "single native pass"}
    if art.get("stitched") and len(r) > 12:
        h = -np.log(np.maximum(r[1:], 1e-9) / np.maximum(r[:-1], 1e-9))
        dh = np.abs(np.diff(h))
        seam = np.arange(2, len(r))[: len(dh)] % 5 == 0
        if seam.any() and (~seam).any():
            out["hazard_change_at_window_seams_vs_elsewhere"] = [round(float(dh[seam].mean()), 4), round(float(dh[~seam].mean()), 4)]
    return out


def build_review(video_dir: Path, video_path: Path | None = None) -> Path | None:
    """Write ``review.html`` next to the artifacts. None without a drop-off analysis."""
    dr, art = _load(video_dir / "ttcc_dropoffs.json"), _load(video_dir / "ttcc_retention.json")
    if not dr or not art:
        return None
    rec = build_segment_recommendations(video_dir)  # TTCC segments + rule proposals + AI
    cards = []
    for sg in rec["segments"]:
        if sg["kind"] == "unflagged" and not sg["proposals"] and not sg["gemini"]:
            continue  # nothing to review there; it still appears in the score table
        cards.append({"id": sg["interval_id"] or sg["id"], "kind": sg["kind"], "start_s": sg["start_s"], "end_s": sg["end_s"],
                      "retention_start": sg["ttcc"]["retention_start"], "retention_end": sg["ttcc"]["retention_end"],
                      "share_of_total_predicted_loss": sg["ttcc"]["share_of_total_predicted_loss"], "observed": sg["observed"],
                      "candidate_causes": sg["proposals"], "gemini": sg["gemini"]})
    src = Path(video_path or art.get("source_path") or "")
    video_url = src.resolve().as_uri() if src.is_file() else None
    data = {
        "video_id": art["video_id"], "sha256": art.get("source_sha256"), "video_url": video_url, "video_path": str(src),
        "duration_s": art.get("duration_s"), "horizon_s": art["horizon_s"], "partial": bool(art.get("partial_coverage")),
        "retention": art["retention"], "intervals": cards, "segments": dr["segments"], "gemini_info": rec["gemini"], "seconds": dr["seconds"],
        "evidence": dr["evidence_availability"], "curve": dr["curve"], "note": dr["note"], "confidence_note": dr["confidence_note"],
        "model": (art.get("model") or {}).get("checkpoint_repo"),
        "deviations": (art.get("inference") or {}).get("deviations_from_upstream_eval") or [],
        "hook": art.get("hook") or hook_metrics(art["retention"], (art.get("inference") or {}).get("window_end_s")),
        "stitched": bool(art.get("stitched")),
        "scores": _load(video_dir / "hook_scores.json"),  # written by scripts/score_video.py; absent until it has run
        "hook_report": _load(video_dir / "hook_report.json"),  # scripts/explain_video.py: why the hook is strong or weak
        "moments": _load(video_dir / "moments.json"),
        "brain": brain_view(video_dir),  # TRIBE v2 network curves: exploratory view, in no score
        "ref_band": reference_band(), "curve_audit": curve_audit(art, video_dir), "stems": _load(video_dir / "stems.json"), "inference": art.get("inference"), "ttcc_status": _load(video_dir / "ttcc_status.json"),  # shown on the Diagnostics view only  # scripts/explain_video.py: strongest and weakest moments
    }
    path = video_dir / REVIEW_FILE
    path.write_text(PAGE.replace("__DATA__", _embed(data)))
    return path




PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hookline editing assistant</title>
<style>
:root{--bg:#0a0c12;--panel:#10131c;--panel2:#151926;--line:#232838;--line2:#2e3447;--fg:#e8eaf1;--mut:#8c92a6;--dim:#5d6377;
--g:#5fd38a;--p:#b47cf0;--r:#ff5468;--y:#ffcf47;--acc:var(--p);--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;color-scheme:dark}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg)}body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Helvetica,Arial,sans-serif}
a{color:inherit}.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}.mut{color:var(--mut);font-size:12.5px}.dim{color:var(--dim)}
/* shell */
.app{display:grid;grid-template-columns:220px minmax(0,1fr);min-height:100vh}
aside{position:sticky;top:0;height:100vh;border-right:1px solid var(--line);background:linear-gradient(180deg,#0d1018,#0a0c12);padding:18px 14px;display:flex;flex-direction:column;gap:18px}
.brand{display:flex;align-items:center;gap:10px;padding:2px 6px}.mark{display:flex;gap:3px;align-items:flex-end;height:22px}.mark i{display:block;width:5px;border-radius:2px}
.mark i:nth-child(1){height:22px;background:var(--g)}.mark i:nth-child(2){height:16px;background:var(--p)}.mark i:nth-child(3){height:11px;background:var(--r)}.mark i:nth-child(4){height:7px;background:var(--y)}
.brand b{font-size:17px;letter-spacing:.2px}.by{font:600 10.5px var(--mono);letter-spacing:2px;margin-left:auto}.by .w{color:var(--g)}.by .l{color:var(--p)}.by .d1{color:var(--r)}.by .d2{color:var(--y)}
nav{display:flex;flex-direction:column;gap:2px}nav a{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;text-decoration:none;color:var(--mut);font-size:13.5px;border:1px solid transparent}
nav a:hover{color:var(--fg);background:var(--panel)}nav a.on{color:var(--fg);background:var(--panel2);border-color:var(--line2)}nav a span.k{font:11px var(--mono);color:var(--dim);margin-left:auto}
.side-foot{margin-top:auto;font-size:11.5px;color:var(--dim);border-top:1px solid var(--line);padding-top:12px}
main{padding:22px 26px 40px;min-width:0}
header.top{display:flex;align-items:flex-start;gap:16px;margin-bottom:14px}header.top h1{font-size:21px;margin:0;font-weight:650;letter-spacing:.1px}
.top .meta{font:12px var(--mono);color:var(--mut);margin-top:3px}.top .right{margin-left:auto;display:flex;gap:10px;align-items:center}
.chip{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;border:1px solid var(--line2);background:var(--panel);font-size:12.5px;color:var(--fg)}
.chip.pred{border-color:color-mix(in srgb,var(--g) 45%,transparent);color:var(--g);background:color-mix(in srgb,var(--g) 9%,transparent)}
.banner{display:flex;gap:10px;align-items:flex-start;border:1px solid var(--line);border-left:3px solid var(--y);border-radius:10px;padding:9px 12px;background:var(--panel);margin:0 0 16px;font-size:12.5px;color:#cfd3df}
/* grid + panels */
.grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(330px,1fr);gap:16px;align-items:start}
.col{display:flex;flex-direction:column;gap:16px;min-width:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;min-width:0}.panel h2{font-size:15px;margin:0 0 2px;font-weight:620;display:flex;align-items:center;gap:8px}
.panel .sub{font-size:12px;color:var(--mut);margin:0 0 10px}.sticky{position:sticky;top:16px}
.legend{margin-left:auto;display:flex;gap:12px;font-size:11.5px;color:var(--mut);font-weight:400}.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}
/* video */
.player{background:#000;border-radius:10px;overflow:hidden;border:1px solid var(--line)}video{display:block;width:100%;height:min(58vh,560px);object-fit:contain;background:#000}
.status{display:flex;justify-content:space-between;gap:12px;font-size:12px;margin:8px 2px 0}
/* hook tiles */
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.tile{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px 12px;cursor:pointer}
.tile:hover{border-color:var(--line2)}.tile .t{font:11px var(--mono);color:var(--mut);letter-spacing:.5px}.tile .v{font:600 22px var(--mono);margin-top:2px}.tile .bar{height:4px;border-radius:3px;background:var(--line);margin-top:8px;overflow:hidden}
.tile .bar b{display:block;height:100%;background:var(--c,var(--g))}
.gauges{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.gauge{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:12px;text-align:center}
.gauge svg{width:96px;height:96px;margin:0 auto}.gauge .gl{font:600 12px var(--mono);letter-spacing:.6px;color:var(--mut);margin-top:6px}.gauge .gv{font:650 11.5px var(--mono);color:var(--fg);margin-top:2px}
.gauge .ga{font-size:11.5px;color:var(--mut);margin-top:6px;line-height:1.35}.gauge.weak .ga b{color:var(--y)}.gauge.ok .ga b{color:var(--g)}
.verdict{display:flex;align-items:center;gap:12px;margin:12px 0 6px}.vb{padding:5px 14px;border-radius:999px;font:700 13px var(--mono);letter-spacing:.8px;border:1px solid}
.vb.strong{color:var(--g);border-color:color-mix(in srgb,var(--g) 50%,transparent);background:color-mix(in srgb,var(--g) 10%,transparent)}.vb.weak{color:var(--r);border-color:color-mix(in srgb,var(--r) 50%,transparent);background:color-mix(in srgb,var(--r) 10%,transparent)}
.vb.average,.vb.notrated{color:var(--y);border-color:color-mix(in srgb,var(--y) 50%,transparent);background:color-mix(in srgb,var(--y) 8%,transparent)}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:12px}.fcard{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--c,var(--line2));border-radius:10px;padding:9px 11px;margin:6px 0;font-size:12.5px}
.fcard b{display:block;font-size:13px}.fcard .why{color:var(--mut);margin-top:3px}.moments{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.mcol h3{margin:0 0 6px;font-size:13px;font-weight:600}.mcol.good h3{color:var(--g)}.mcol.bad h3{color:var(--r)}.mcard{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--c);border-radius:10px;padding:10px 12px;margin:8px 0;cursor:pointer}
.mcard:hover{border-color:var(--line2)}.mcard .tm{font:650 14px var(--mono)}.mcard ul{margin:4px 0 2px 16px;padding:0;font-size:12.5px}.mcard .act{margin-top:6px;font-size:12.5px;color:#e3d2fd}
.wtable td.pos{color:var(--g)}.wtable td.neg{color:var(--r)}.wtable td.noise{color:var(--mut)}
/* chart + strip + film */
svg{width:100%;height:auto;display:block}
#strip{position:relative;height:12px;border-radius:999px;background:var(--panel2);overflow:hidden;cursor:pointer;margin:10px 0 8px;border:1px solid var(--line)}
#strip .seg{position:absolute;top:0;bottom:0;opacity:.85}#strip .seg.active{opacity:1;box-shadow:inset 0 0 0 2px #fff}
#strip .none{position:absolute;top:0;bottom:0;background:repeating-linear-gradient(45deg,#2a3042,#2a3042 4px,transparent 4px,transparent 8px)}
#strip .trim{position:absolute;top:0;bottom:0;background:#000;opacity:.65}#strip .head{position:absolute;top:-2px;bottom:-2px;width:2px;background:#fff;box-shadow:0 0 6px #fff}
.steep_drop{--c:var(--r)}.sustained_decline{--c:var(--y)}.holds{--c:var(--g)}.unflagged{--c:#5b6276}#strip .seg{background:var(--c)}
.film{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:6px}.film .fr{position:relative;cursor:pointer;border-radius:7px;overflow:hidden;border:1px solid var(--line);background:var(--panel2)}
.film canvas{display:block;width:100%;aspect-ratio:16/10;object-fit:cover}.film .fr span{position:absolute;left:5px;bottom:4px;font:10.5px var(--mono);color:#fff;text-shadow:0 1px 2px #000}
.film .fr.on{outline:2px solid var(--p);outline-offset:-2px}
.row-actions{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
/* buttons */
button{font:inherit;font-size:13px;padding:6px 12px;border:1px solid var(--line2);background:var(--panel2);color:var(--fg);border-radius:8px;cursor:pointer}
button:hover{border-color:var(--p)}button.primary{background:linear-gradient(135deg,#8b5cf6,var(--p));border-color:transparent;color:#fff}button.primary:hover{filter:brightness(1.1)}
button.ghost{background:transparent;border-color:color-mix(in srgb,var(--p) 55%,transparent);color:#d9c3fb}button.keep{background:transparent;border-color:color-mix(in srgb,var(--g) 55%,transparent);color:var(--g)}
button.small{font-size:12px;padding:4px 10px}
/* segment list */
.seglist{display:flex;flex-direction:column}.srow{display:grid;grid-template-columns:14px 110px minmax(0,1fr) auto;gap:12px;align-items:center;padding:11px 8px;border-bottom:1px solid var(--line);cursor:pointer;border-radius:8px}
.srow:hover{background:var(--panel2)}.srow.on{background:var(--panel2);box-shadow:inset 3px 0 0 var(--c)}.srow .dot{width:10px;height:10px;border-radius:50%;background:var(--c);box-shadow:0 0 8px var(--c)}
.srow .tm{font:600 13px var(--mono)}.srow .lb{font-weight:600;color:var(--c);margin-right:8px}.srow .ds{color:var(--mut);font-size:12.5px}
.srow.playing .tm{color:#fff}.srow.playing .dot{outline:2px solid #fff;outline-offset:2px}
/* detail panel (one .card per candidate segment; only the selected one is shown) */
.card{display:none}.card.sel{display:block}.card.active .badge{box-shadow:0 0 0 1px #fff}
.dhead{display:flex;align-items:center;gap:12px;margin-bottom:12px}.dhead .tm{font:650 20px var(--mono)}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;color:var(--c);background:color-mix(in srgb,var(--c) 14%,transparent);border:1px solid color-mix(in srgb,var(--c) 40%,transparent)}
.thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:6px}.thumbs canvas{width:100%;aspect-ratio:16/10;border-radius:7px;border:1px solid var(--line);background:var(--panel2);cursor:pointer}
.sec{border-top:1px solid var(--line);padding:12px 0 2px;margin-top:10px}.sec h4{margin:0 0 6px;font-size:13px;color:#d7c2fb;font-weight:600;display:flex;align-items:center;gap:8px}
.sec h4 i{display:inline-block;width:6px;height:6px;border-radius:2px;background:var(--p)}
.sec ul{margin:2px 0 4px 16px;padding:0}.sec li{margin:2px 0}
.quote{font-style:italic;color:#cfd3df;border-left:2px solid var(--line2);padding-left:10px;margin:4px 0}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0}.chips span{font-size:11.5px;padding:3px 9px;border-radius:999px;background:var(--panel2);border:1px solid var(--line2);color:#cfd3df}
.rbar{height:8px;border-radius:5px;background:var(--line);overflow:hidden;margin:8px 0 4px;position:relative}.rbar b{position:absolute;top:0;bottom:0;border-radius:5px}
.sug{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:8px 0}.sug .tag{font:700 11px var(--mono);letter-spacing:1px;color:var(--y);margin-right:6px}
.sug.gem{border-color:color-mix(in srgb,var(--p) 40%,var(--line))}.sug.gem .tag{color:var(--p)}
.sug textarea{width:100%;margin-top:8px;font:inherit;font-size:12.5px;padding:7px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px}
.sug input[type=number]{width:70px;font:12.5px var(--mono);padding:3px 5px;background:var(--bg);color:var(--fg);border:1px solid var(--line2);border-radius:6px}
.radios{display:flex;gap:6px;margin-top:8px}.radios label{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border:1px solid var(--line2);border-radius:999px;font-size:12px;cursor:pointer}
.radios input{accent-color:var(--p)}.radios label:has(input:checked){border-color:var(--p);color:#e3d2fd;background:color-mix(in srgb,var(--p) 12%,transparent)}
/* tables */
table{border-collapse:collapse;width:100%;font-size:12.5px}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--mut);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.6px}
tr.row{cursor:pointer}tr.row:hover{background:var(--panel2)}.pill{display:inline-block;padding:1px 8px;border-radius:999px;border:1px solid var(--line2);font-size:11.5px;color:var(--c,var(--fg))}
.bar{display:inline-block;height:6px;background:var(--c,#6b7280);border-radius:4px;vertical-align:middle;margin-right:6px}
details summary{cursor:pointer;color:var(--mut);font-size:13px}
@media (max-width:760px){.cols2,.moments{grid-template-columns:minmax(0,1fr)}}
@media (max-width:640px){.gauges{grid-template-columns:minmax(0,1fr)}}
@media (max-width:1180px){.grid{grid-template-columns:minmax(0,1fr)}.sticky{position:static}}
@media (max-width:860px){.app{grid-template-columns:minmax(0,1fr)}aside{position:static;height:auto;flex-direction:row;flex-wrap:wrap;align-items:center;padding:12px 16px;gap:10px}
nav{flex-direction:row;flex-wrap:wrap}nav a{padding:5px 9px}nav a span.k{display:none}.side-foot{display:none}main{padding:14px 16px 32px;overflow-x:hidden}
header.top{flex-wrap:wrap}.top .right{margin-left:0}.panel{padding:13px}.panel h2{flex-wrap:wrap}.legend{margin-left:0}.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}
.film{grid-auto-flow:row;grid-template-columns:repeat(4,minmax(0,1fr))}.srow{grid-template-columns:12px auto minmax(0,1fr) auto;gap:8px;padding:10px 4px}.srow .ds{display:none}.srow .lb{font-size:12.5px}
.status{flex-direction:column;gap:2px}.dhead .tm{font-size:17px}}
/* polish layer: aurora backdrop, glass panels, glow, motion */
body{background:radial-gradient(900px 500px at 85% -10%,rgba(180,124,240,.16),transparent 60%),radial-gradient(700px 420px at 0% 10%,rgba(95,211,138,.10),transparent 60%),radial-gradient(800px 500px at 60% 110%,rgba(255,84,104,.08),transparent 60%),var(--bg);background-attachment:fixed}
::selection{background:color-mix(in srgb,var(--p) 40%,transparent)}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--line2);border-radius:8px;border:2px solid var(--bg)}::-webkit-scrollbar-thumb:hover{background:var(--dim)}
aside{background:linear-gradient(180deg,rgba(18,21,32,.85),rgba(10,12,18,.9));backdrop-filter:blur(14px)}
.brand b{background:linear-gradient(90deg,#fff,#d9c3fb 60%,var(--g));-webkit-background-clip:text;background-clip:text;color:transparent}
.mark i{box-shadow:0 0 10px currentColor;animation:eq 1.6s ease-in-out infinite alternate}.mark i:nth-child(1){color:var(--g)}.mark i:nth-child(2){color:var(--p);animation-delay:.2s}.mark i:nth-child(3){color:var(--r);animation-delay:.4s}.mark i:nth-child(4){color:var(--y);animation-delay:.6s}
@keyframes eq{from{transform:scaleY(.65);transform-origin:bottom}to{transform:scaleY(1);transform-origin:bottom}}
nav a{transition:all .18s}nav a.on{background:linear-gradient(90deg,color-mix(in srgb,var(--p) 22%,transparent),transparent);border-color:color-mix(in srgb,var(--p) 40%,transparent);box-shadow:inset 3px 0 0 var(--p)}
header.top h1{font-size:25px;background:linear-gradient(90deg,#fff 30%,#cdb0f7);-webkit-background-clip:text;background-clip:text;color:transparent}
.chip.pred{box-shadow:0 0 18px color-mix(in srgb,var(--g) 25%,transparent)}.chip.pred::first-letter{animation:pulse 1.8s ease-in-out infinite}
@keyframes pulse{50%{opacity:.25}}
.panel{background:linear-gradient(160deg,rgba(24,28,43,.78),rgba(14,17,26,.82));border-color:rgba(120,130,170,.16);border-radius:18px;box-shadow:0 10px 34px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.04);backdrop-filter:blur(10px);animation:rise .5s ease both;transition:border-color .2s,box-shadow .2s}
.panel:hover{border-color:color-mix(in srgb,var(--p) 35%,var(--line));box-shadow:0 12px 40px rgba(0,0,0,.42),0 0 0 1px color-mix(in srgb,var(--p) 14%,transparent)}
.col>.panel:nth-child(2){animation-delay:.06s}.col>.panel:nth-child(3){animation-delay:.12s}.col>.panel:nth-child(4){animation-delay:.18s}.col>.panel:nth-child(5){animation-delay:.24s}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.panel h2{font-size:16px;letter-spacing:.2px}.panel h2::before{content:"";width:4px;height:16px;border-radius:3px;background:linear-gradient(180deg,var(--p),var(--g))}
.player{border-radius:14px;box-shadow:0 0 0 1px rgba(255,255,255,.06),0 14px 40px rgba(0,0,0,.55)}
.tile,.gauge,.fcard,.mcard,.sug{border-radius:14px;transition:transform .16s,border-color .16s,box-shadow .16s;background:linear-gradient(160deg,rgba(30,35,54,.7),rgba(18,21,32,.8))}
.tile:hover,.mcard:hover,.gauge:hover{transform:translateY(-2px);border-color:var(--p);box-shadow:0 8px 24px rgba(0,0,0,.4),0 0 18px color-mix(in srgb,var(--p) 20%,transparent)}
.tile .v{font-size:26px;background:linear-gradient(180deg,#fff,#b9bfd6);-webkit-background-clip:text;background-clip:text;color:transparent}
.tile .bar,.rbar{height:5px}.tile .bar b{background:linear-gradient(90deg,var(--c,var(--g)),color-mix(in srgb,var(--c,var(--g)) 50%,#fff));box-shadow:0 0 10px var(--c,var(--g))}
.gauge svg{filter:drop-shadow(0 0 10px color-mix(in srgb,var(--p) 35%,transparent))}
.vb{box-shadow:0 0 22px currentColor;text-shadow:0 0 10px currentColor}
#strip{height:14px;box-shadow:inset 0 1px 3px rgba(0,0,0,.5)}#strip .head{box-shadow:0 0 12px #fff,0 0 4px var(--p)}#strip .seg{box-shadow:0 0 10px var(--c)}
.film .fr{transition:transform .15s,border-color .15s}.film .fr:hover{transform:translateY(-2px) scale(1.03);border-color:var(--p)}.film .fr.on{box-shadow:0 0 16px color-mix(in srgb,var(--p) 55%,transparent)}
button{transition:all .15s;border-radius:10px}button:hover{transform:translateY(-1px);box-shadow:0 6px 18px color-mix(in srgb,var(--p) 25%,transparent)}button:active{transform:none}
button.primary{background:linear-gradient(135deg,#7c4dff,#b47cf0 55%,#e07cf0);box-shadow:0 6px 22px color-mix(in srgb,var(--p) 40%,transparent);font-weight:600}
.srow{transition:background .15s,transform .15s}.srow:hover{transform:translateX(3px)}.srow.on{background:linear-gradient(90deg,color-mix(in srgb,var(--c) 14%,transparent),var(--panel2))}
.badge{box-shadow:0 0 14px color-mix(in srgb,var(--c) 30%,transparent)}
.banner{border-radius:14px;background:linear-gradient(90deg,rgba(255,207,71,.08),var(--panel) 40%)}
/* views */
body[data-view="analysis"] [data-view]:not([data-view~="analysis"]){display:none}
body[data-view="insights"] [data-view]:not([data-view~="insights"]){display:none}
body[data-view="edit"] [data-view]:not([data-view~="edit"]){display:none}
body[data-view="analysis"] .grid{grid-template-columns:minmax(0,1fr)}
body[data-view="analysis"] #hookwhy,body[data-view="analysis"] #hooknote{display:none!important}
body[data-view="analysis"] video{height:min(62vh,620px)}
.tsug{display:grid;grid-template-columns:1fr auto;gap:6px 14px;padding:11px 4px;border-top:1px solid var(--line)}.tsug:first-child{border-top:0}
.tsug b{font-size:13.5px}.tsug .why{grid-column:1;color:var(--mut);font-size:12.5px}.tsug .acts{grid-column:2;grid-row:1/3;display:flex;gap:8px;align-items:center}
.tag{font:600 11px var(--mono);color:var(--mut);margin-right:6px}.tag.ai{color:var(--p)}
/* clean results view: numbers, curve, edits, moments only (explanations live in the other tabs) */
body[data-view="analysis"] .sub,body[data-view="analysis"] .tsug .why,body[data-view="analysis"] .mcard ul,body[data-view="analysis"] #vnote:empty,
body[data-view="analysis"] #logtoggle,body[data-view="analysis"] #bandverdict,body[data-view="analysis"] #logscalenote,body[data-view="analysis"] #playedited,body[data-view="analysis"] #editedinfo,
body[data-view="analysis"] .side-foot,body[data-view="analysis"] .chip.pred,body[data-view="analysis"] #topexport,body[data-view="analysis"] #activeinfo,body[data-view="analysis"] #foot
body[data-view="analysis"] .row-actions:has(#logtoggle){display:none}
body[data-view="analysis"] .gauges,body[data-view="insights"] .gauges{grid-template-columns:repeat(3,minmax(0,1fr))}
body[data-view="analysis"] .mcard .act{display:none}
body[data-view="analysis"] main{max-width:1000px}
/* calm layer: flat, static, no glow (overrides the polish layer above) */
*{animation:none!important;transition:none!important}
body{background:var(--bg)!important}
aside{background:var(--bg)!important;backdrop-filter:none!important}
.brand b,header.top h1,.tile .v{background:none!important;-webkit-background-clip:initial!important;background-clip:initial!important;color:var(--fg)!important}
.panel,.tile,.gauge,.fcard,.mcard,.sug{background:var(--panel)!important;box-shadow:none!important;backdrop-filter:none!important;border-radius:12px!important;transform:none!important}
.panel:hover,.tile:hover,.mcard:hover,.gauge:hover,.film .fr:hover,.srow:hover,button:hover{box-shadow:none!important;transform:none!important}
.mark i,.vb,.badge,button,button.primary,.gauge svg,.tile .bar b,#strip .seg,#strip .head,.film .fr.on,.srow .dot,.chip.pred{box-shadow:none!important;text-shadow:none!important;filter:none!important}
button.primary{background:var(--p)!important}
nav a.on{background:var(--panel2)!important;box-shadow:none!important}
.panel h2::before{display:none}
body[data-view="analysis"] #strip,body[data-view="analysis"] #film{display:none!important}
body[data-view="analysis"] .zone,body[data-view="analysis"] .band,body[data-view="analysis"] .mark,body[data-view="analysis"] .legend,body[data-view="analysis"] .tsug .acts .ghost,#navsegs{display:none!important}
/* fewer words on the dashboard views: keep the numbers and graphs, drop the prose */
body[data-view="insights"] #detailcol,body[data-view="analysis"] #foot,body[data-view="insights"] #foot,.side-foot,body[data-view="insights"] .chip.pred,body[data-view="insights"] #topexport,body[data-view="insights"] .tile .bar{display:none!important}
body[data-view="insights"] .grid{grid-template-columns:minmax(0,1fr)}
body[data-view="insights"] #hookwhy .verdict{margin:10px 0 0}
body[data-view="insights"] .tile .t{font-family:inherit!important;font-size:13px;letter-spacing:0}

body[data-view="insights"] .sub,body[data-view="insights"] #strip,body[data-view="insights"] #film,body[data-view="insights"] #playedited,body[data-view="insights"] #editedinfo,body[data-view="analysis"] #brainsub,body[data-view="insights"] #brainsub,body[data-view="analysis"] #brainread,body[data-view="insights"] #brainread,
body[data-view="analysis"] #brainpeaks,body[data-view="insights"] #brainpeaks,body[data-view="insights"] #hooknote,body[data-view="insights"] #momentssub,body[data-view="insights"] .mcard ul,body[data-view="insights"] .mcard .act,
body[data-view="insights"] #hookwhy .cols2,body[data-view="insights"] #hookwhy details,body[data-view="insights"] #hookwhy p,body[data-view="insights"] #hookwhy .verdict .mut,body[data-view="insights"] #hookwhy .fcard,
body[data-view="insights"] #gemsec,body[data-view="insights"] .srow .ds,body[data-view="insights"] #logtoggle,body[data-view="insights"] #bandverdict,body[data-view="insights"] #logscalenote{display:none!important}
body[data-view="insights"] .mono,body[data-view="insights"] .gauge .gl,body[data-view="insights"] .gauge .gv,body[data-view="insights"] svg text{font-family:inherit!important;text-transform:none!important;letter-spacing:0!important}
body[data-view="analysis"] .tsug b{font-weight:500}
body[data-view="analysis"] .mono,body[data-view="analysis"] .gauge .gl,body[data-view="analysis"] .gauge .gv,body[data-view="analysis"] .top .meta,body[data-view="analysis"] svg text{font-family:inherit!important;text-transform:none!important;letter-spacing:0!important}
body[data-view="analysis"] .gauge .gl{font-size:14px;color:var(--fg)}body[data-view="analysis"] .gauge .gv{font-size:12.5px;color:var(--mut);font-weight:400}
body[data-view="analysis"] .top .meta{font-size:14px;color:var(--mut)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="app">
<aside>
  <div class="brand"><div class="mark"><i></i><i></i><i></i><i></i></div><b>Hookline</b><span class="by"><span class="w">W</span><span class="l">L</span><span class="d1">D</span><span class="d2">D</span></span></div>
  <nav id="nav">
    <a href="#analysis" data-v="analysis">Results</a>
    <a href="#insights" data-v="insights">Analytics</a>
    <a href="#edit" data-v="edit">Edit Review<span class="k" id="navsegs"></span></a>
  </nav>
  <div class="side-foot" id="sidefoot"></div>
</aside>
<main>
<header class="top" id="editor">
  <div><h1 id="title">Hookline editing assistant</h1><div class="meta" id="meta"></div></div>
  <div class="right"><span class="chip pred" title="Estimate, not real viewer data">● Predicted retention</span><button class="primary" id="topexport">Export ↓</button></div>
</header>
<div class="grid">
<div class="col">
  <section class="panel" data-view="analysis edit">
    <div class="player"><video id="v" controls preload="metadata"></video></div>
    <p class="mut" id="vnote" style="margin:6px 2px 0"></p>
    <div class="status"><span id="now" class="mono">0:00</span><span id="activeinfo" class="mut"></span></div>
  </section>
  <section class="panel" data-view="analysis insights" id="scorespanel" style="display:none">
    <h2>Scores</h2>
    <div class="gauges" id="gauges"></div>
  </section>
  <section class="panel" data-view="insights" id="hookpanel">
    <h2>Opening hold</h2>
    <p class="sub">TTCC-predicted hold in the opening seconds.</p>
    <div class="tiles" id="tiles"></div>
    <p class="mut" id="hooknote" style="margin:8px 0 0"></p>
    <div id="hookwhy" style="display:none"></div>
  </section>
  <section class="panel" data-view="analysis insights" id="brainpanel" style="display:none">
    <h2>Brain response</h2>
    <p class="sub" id="brainsub"></p>
    <div id="brainread" class="fcard" style="--c:#b47cf0;display:none;margin-bottom:10px"></div>
    <div id="brainchart"></div>
    <div id="brainpeaks" class="mut" style="margin-top:6px"></div>
  </section>
  <section class="panel" data-view="analysis insights edit" id="retention">
    <h2>Viewer retention <span class="legend"><span><i style="background:var(--g)"></i>Holds</span><span><i style="background:var(--y)"></i>Decline</span><span><i style="background:var(--r)"></i>Steep drop</span></span></h2>
    <p class="sub">Click the curve, the bar or a frame to jump. The white line follows the video.</p>
    <svg id="chart" viewBox="0 0 760 250" role="img" aria-label="Predicted retention curve"></svg>
    <div class="row-actions" style="margin-top:6px"><button class="small" id="logtoggle">Zoom into the tail (log scale)</button></div>
    <p id="bandverdict" style="display:none;margin:6px 0 0;font-size:13px"></p>
    <p class="mut" id="logscalenote" style="display:none;margin:4px 0 0">Log scale: each gridline is ten times smaller than the one above. It shows the shape of the small remaining share; it does not change the prediction.</p>
    <div id="strip" title="Click to jump"></div>
    <div class="film" id="film"></div>
    <div class="row-actions"><button class="primary" id="playedited">▶ Play edited version (skips accepted trims)</button><span class="mut" id="editedinfo"></span></div>
  </section>
  <section class="panel" data-view="analysis" id="topsugs">
    <h2>Suggested edits</h2>
    <div id="topsugslist"></div>
  </section>
  <section class="panel" data-view="insights" id="momentspanel" style="display:none">
    <h2>Best and worst moments</h2>
    <p class="sub" id="momentssub"></p>
    <div class="moments" id="moments"></div>
  </section>
  <section class="panel" data-view="edit" id="whatifpanel">
    <h2>What if I make these edits?</h2>
    <p class="sub">Accept trims below, then predict the edited video with the same model. A model prediction of the edited cut, not audience data.</p>
    <div class="row-actions" style="margin-top:0"><button id="acceptall">Accept all suggested trims</button><button class="primary" id="whatifgo" disabled>Predict the edited version</button><span class="mut" id="whatifmsg"></span></div>
    <div id="whatifout" style="display:none"></div>
  </section>
  <section class="panel" data-view="insights" id="segments">
    <h2>Segments</h2>
    <p class="sub">The predicted span split into consecutive segments. Select one to see its evidence and the proposed edits.</p>
    <div class="seglist" id="seglist"></div>
  </section>
  <section class="panel" data-view="insights" id="gemsec"><div id="geminibox"></div></section>
  <section class="panel" data-view="edit" id="exports">
    <h2>Export</h2><p class="sub">Your decisions stay in this browser until you export them. Nothing is uploaded.</p>
    <div class="row-actions" style="margin-top:4px"><button class="primary" id="exportjson">Export edit decisions (JSON)</button><button class="primary" id="exportcsv">Export edit decisions (CSV)</button><button id="clear">Clear my decisions</button><span class="mut" id="saved"></span></div>
  </section>
  <p class="mut" id="foot"></p>
</div>
<div class="col" data-view="insights edit" id="detailcol">
  <section class="panel sticky" id="detail">
    <h2>Segment analysis <span class="legend mono" id="detailpos"></span></h2>
    <div id="segs"></div>
    <div id="nodetail" class="mut" style="display:none;padding:8px 0"></div>
  </section>
</div>
</div>
</main>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const $=(id)=>document.getElementById(id);
const el=(tag,props,...kids)=>{const e=document.createElement(tag);for(const[k,v]of Object.entries(props||{})){if(k==='class')e.className=v;else if(k==='text')e.textContent=v;else e.setAttribute(k,v);}for(const c of kids)if(c!=null&&c!==false)e.append(c);return e;};
const fmt=(t)=>{const m=Math.floor(t/60),s=t-60*m;return m+':'+(s<10?'0':'')+s.toFixed(1);};
const mmss=(t)=>{t=Math.round(t);return Math.floor(t/60)+':'+String(t%60).padStart(2,'0');};
const pc=(r)=>r==null?'–':Math.round(100*r)+'%';
const V=$('v'),TOTAL=D.duration_s||D.horizon_s;
const SERVED=location.protocol==='http:'||location.protocol==='https:';const SRC=SERVED?'video':null;  // served by scripts/serve_review.py, or the file:// path
const DISCLAIMER='PREDICTED retention from a pretrained model (TTCC). Not actual audience data; its accuracy on these videos has not been verified. Suggestions are unvalidated rules over measured features.';
const COLOR={steep_drop:'#ff5468',sustained_decline:'#ffcf47',holds:'#5fd38a',unflagged:'#5b6276'};
// ---- pure helpers (also exercised by the self-test)
function skipTarget(t,ranges){for(const r of ranges){if(t>=r.a-0.02&&t<r.b)return r.b;}return null;}
function csvCell(v){const s=v==null?'':String(v);return /[",\n\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function toCSV(items){const cols=['video_id','interval_id','interval_kind','interval_start_s','interval_end_s','predicted_R_start','predicted_R_end','source','rule','suggestion','rationale','confidence','evidence_refs','action','edit_start_s','edit_end_s','decision','note','data_kind'];
return [cols.join(',')].concat(items.map(i=>cols.map(c=>csvCell(i[c])).join(','))).join('\r\n')+'\r\n';}
// ---- state
const KEY='hookline-assistant:'+(D.sha256||D.video_id);let store={};try{store=JSON.parse(localStorage.getItem(KEY)||'{}');}catch(e){store={};}
const persist=()=>{try{localStorage.setItem(KEY,JSON.stringify(store));$('saved').textContent='saved in this browser';}catch(e){$('saved').textContent='could not save in this browser; export instead';}};
const SUGS=[];D.intervals.forEach(iv=>iv.candidate_causes.forEach((c,i)=>SUGS.push({id:iv.id+':'+c.rule+':'+i,iv,c})));
function cur(sg){const s=store[sg.id]||{};return{decision:s.decision||'undecided',note:s.note||'',a:s.a!=null?s.a:sg.c.edit.start_s,b:s.b!=null?s.b:sg.c.edit.end_s};}
function acceptedTrims(){return SUGS.filter(sg=>sg.c.edit.action==='trim').map(sg=>({sg,s:cur(sg)})).filter(x=>x.s.decision==='accepted'&&x.s.b>x.s.a).map(x=>({a:x.s.a,b:x.s.b})).sort((p,q)=>p.a-q.a).reduce((m,r)=>{const l=m[m.length-1];if(l&&r.a<=l.b)l.b=Math.max(l.b,r.b);else m.push({a:r.a,b:r.b});return m;},[]);}
function exportRow(sg){const s=cur(sg);const act=sg.c.edit.action;return{video_id:D.video_id,interval_id:sg.iv.id,interval_kind:sg.iv.kind,interval_start_s:sg.iv.start_s,interval_end_s:sg.iv.end_s,predicted_R_start:+sg.iv.retention_start.toFixed(4),predicted_R_end:+sg.iv.retention_end.toFixed(4),source:sg.c.source,rule:sg.c.rule,suggestion:sg.c.suggestion,rationale:sg.c.rationale||'',confidence:sg.c.confidence==null?'':sg.c.confidence,evidence_refs:(sg.c.evidence_refs||[]).join(' '),action:act,edit_start_s:act==='add'?sg.c.edit.start_s:+(+s.a).toFixed(2),edit_end_s:act==='add'?'':+(+s.b).toFixed(2),decision:s.decision,note:s.note,data_kind:'predicted_retention_unvalidated'};}
function buildExport(){const items=SUGS.map(exportRow);const tr=acceptedTrims();const removed=tr.reduce((a,r)=>a+(r.b-r.a),0);
return{video_id:D.video_id,sha256:D.sha256,exported_at:new Date().toISOString(),disclaimer:DISCLAIMER,ai_used:!!(D.gemini_info&&D.gemini_info.used),retention_is_prediction:true,model:D.model,
original_duration_s:TOTAL,hook:D.hook,segment_scores:segmentRows(),segment_scores_note:'predicted fall = TTCC model prediction; heuristic risk/quality = hand-set rules over measured features; not combined, not audience data',accepted_trims:tr.map(r=>({start_s:+r.a.toFixed(2),end_s:+r.b.toFixed(2)})),duration_after_accepted_trims_s:+(TOTAL-removed).toFixed(2),edits:items};}
// ---- header
document.title='Hookline — '+D.video_id;$('title').textContent=D.video_id;
{const rp=D.scores&&D.scores.retention_pct;$('meta').textContent=(D.duration_s?D.duration_s.toFixed(0)+' s':'')+(rp&&rp['5s']!=null?'  ·  about '+rp['5s']+'% of viewers predicted still watching at 5 s':'');}
$('sidefoot').textContent='Model: '+(D.model||'TTCC')+'. Curves are predictions, not audience data. Nothing on this page is uploaded.';
$('topexport').addEventListener('click',()=>{$('exports').scrollIntoView({behavior:'smooth'});});
if(SRC){V.src=SRC;V.addEventListener('error',()=>{$('vnote').textContent='The browser could not load the video ('+(SERVED?'the file may be missing or use a codec this browser cannot play':'opened as a local file')+'). Use the Hookline app, or: python3 scripts/serve_review.py '+D.video_id+' --artifacts <root> --video <file> --open. Timestamps still apply.';});}
else{V.style.display='none';if(!SERVED&&D.video_url){$('vnote').textContent='This page was opened as a local file, which browsers block from playing video. Open it through Hookline: python3 scripts/serve_review.py '+D.video_id+' --artifacts <root> --video <file> --open';}else $('vnote').textContent='The video file was not found. Timestamps still apply, but playback and previews are unavailable.';}
const GI=D.gemini_info||{used:false};const gb0=$('geminibox');
if(GI.used){gb0.append(el('h2',{text:'AI overview (model interpretation)'}));if(GI.overall){gb0.append(el('p',{text:GI.overall.hook_assessment}));if(GI.overall.top_priorities.length)gb0.append(el('p',{class:'mut',text:'AI\'s priority order: '+GI.overall.top_priorities.join(' → ')}));}
if(GI.audio){const A=GI.audio;gb0.append(el('h3',{text:'Audio (AI hypothesis, from listening)'}),el('p',{text:A.kind.replace(/_/g,' ')+(A.energy?' · '+A.energy+' energy':'')+'. '+A.description}),el('p',{class:'mut',text:A.familiarity+' Trend status is not known to this tool.'}));if(A.fit){gb0.append(el('div',{class:'fcard',style:'--c:'+(A.fit.verdict==='fits'?'#5fd38a':A.fit.verdict==='clashes'?'#ff5468':'#ffcf47')},el('b',{text:'Does the audio suit the video? '+A.fit.verdict+' (AI hypothesis)'}),el('div',{class:'why',text:A.fit.reason})));}
if(A.changes&&A.changes.length)gb0.append(el('p',{class:'mut',text:'Audio moments: '+A.changes.map(c=>mmss(c.at_s)+' '+c.what).join(' · ')}));}
gb0.append(el('p',{class:'mut',text:'AI explains the content and proposes edits; TTCC predicts retention. All of it is hypotheses until validated against real audience-retention data.'+(GI.validation&&GI.validation.dropped?' '+GI.validation.dropped+' item(s) of its answer failed validation and were dropped.':'')}));}
else{gb0.append(el('h2',{text:'AI'}),el('p',{class:'mut',text:'AI interpretation: not used'+(GI.reason?' ('+GI.reason+')':'')+'. Proposals are rule-based only.'}));}
// ---- control: jump / previews
let mode=null;function startPlay(){if(SRC)V.play().catch(()=>{});}
function ensurePlayer(){if(!['analysis','edit'].includes(document.body.getAttribute('data-view')))setView('edit',true);}
function jump(t,end){ensurePlayer();mode=end==null?null:{kind:'segment',until:end};V.currentTime=t;startPlay();V.scrollIntoView({behavior:'smooth',block:'nearest'});}
function previewTrim(a,b){ensurePlayer();mode={kind:'trim',a:a,b:b,until:Math.min(TOTAL,b+2)};V.currentTime=Math.max(0,a-2);startPlay();V.scrollIntoView({behavior:'smooth',block:'nearest'});}
function playEdited(){const tr=acceptedTrims();mode={kind:'edited',ranges:tr};V.currentTime=0;startPlay();}
function tick(){if(!SRC||!mode)return;const t=V.currentTime;
if(mode.kind==='segment'&&t>=mode.until){V.pause();mode=null;}
else if(mode.kind==='trim'){if(t>=mode.a&&t<mode.b)V.currentTime=mode.b;else if(t>=mode.until){V.pause();mode=null;}}
else if(mode.kind==='edited'){const to=skipTarget(t,mode.ranges);if(to!=null)V.currentTime=to;}}
// ---- frames: a hidden second <video> seeks and paints canvases (drawing works even when the frame cannot be read back)
const FQ=[];let FV=null,fbusy=false,PAINTED=0;
function frameAt(canvas,t,first){if(!SRC)return;if(first)FQ.unshift({canvas,t});else FQ.push({canvas,t});pump();}
function pump(){if(fbusy||!FQ.length)return;if(!FV){FV=document.createElement('video');FV.muted=true;FV.preload='auto';FV.src=SRC;FV.addEventListener('error',()=>{FQ.length=0;});}
const job=FQ.shift();fbusy=true;let fin=false;const done=()=>{if(fin)return;fin=true;clearTimeout(wd);FV.removeEventListener('seeked',draw);fbusy=false;setTimeout(pump,0);};const wd=setTimeout(()=>done(),5000);  // a seek that never reports back must not stall the queue
const draw=()=>{try{const c=job.canvas,ctx=c.getContext('2d');const vw=FV.videoWidth||16,vh=FV.videoHeight||10;c.width=320;c.height=200;const sc=Math.max(320/vw,200/vh);ctx.drawImage(FV,(320-vw*sc)/2,(200-vh*sc)/2,vw*sc,vh*sc);PAINTED++;}catch(e){}done();};
FV.addEventListener('seeked',draw);const go=()=>{try{FV.currentTime=Math.min(Math.max(0,job.t),Math.max(0,(FV.duration||TOTAL)-0.05));}catch(e){done();}};
if(FV.readyState>=1)go();else FV.addEventListener('loadedmetadata',go,{once:true});}
// ---- scores (0-100 positions in other videos, each with the measured accuracy of its model)
(function(){const S=D.scores;if(!S||S.status!=='ok')return;$('scorespanel').style.display='';const NS='http://www.w3.org/2000/svg';
const ring=(v,color)=>{const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 100 100');const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const[k,x]of Object.entries(a))e.setAttribute(k,x);return e;};
svg.append(mk('circle',{cx:50,cy:50,r:40,fill:'none',stroke:'#232838','stroke-width':9}));
if(v!=null){const c=2*Math.PI*40;svg.append(mk('circle',{cx:50,cy:50,r:40,fill:'none',stroke:color,'stroke-width':9,'stroke-linecap':'round','stroke-dasharray':(c*v/100)+' '+c,transform:'rotate(-90 50 50)'}));}
const t=mk('text',{x:50,y:58,'text-anchor':'middle','font-size':26,'font-weight':700,fill:'#e8eaf1','font-family':'ui-monospace,Menlo,monospace'});t.textContent=v==null?'–':String(v);svg.append(t);return svg;};
const g=$('gauges');const add=(label,v,color,line,acc,cls)=>{const box=el('div',{class:'gauge '+(cls||'')});box.append(ring(v,color),el('div',{class:'gl',text:label}),el('div',{class:'gv',text:line}));if(acc)box.append(acc);g.append(box);};
const B=S.breakout_score||{},HS=S.hook_score||{},RP=S.retention_pct;
const PP=S.popularity_score||{};const pacc=PP.accuracy||{};const pnote=el('div',{class:'ga'});
pnote.title=pacc.verdict||'Exploratory brain-response signal, not a viral score.';
pnote.append(el('b',{text:'TRIBE signal · '}),document.createTextNode(PP.score==null?(PP.reason||''):'brain response over time'));
add('BRAIN RESPONSE (TRIBE)',PP.score==null?null:PP.score,'#b47cf0',PP.score==null?'not available':'exploratory',pnote,pacc.evaluated&&pacc.auc_ci95&&pacc.auc_ci95[0]>0.5?'ok':'weak');
add('HOOK',HS.score==null?null:HS.score,'#5fd38a',HS.score==null?'needs TTCC on other videos':'early hold '+Math.round(100*HS.early_hold)+'%',HS.score==null?el('div',{class:'ga',text:HS.reason||''}):null);
add('RETENTION (predicted)',RP?RP['5s']:null,'#ffcf47',RP?['1s','3s','5s','10s'].filter(k=>RP[k]!=null).map(k=>k+' '+RP[k]+'%').join(' · '):'no TTCC prediction',null);
})();
// ---- hook strength: why it is strong or weak (from hook_report.json)
(function(){const R=D.hook_report;if(!R||R.status!=='ok')return;const box=$('hookwhy');box.style.display='';const V=R.verdict||{};
const label=(V.label||'not rated');const vrow=el('div',{class:'verdict'},el('span',{class:'vb '+label.replace(' ',''),text:'PREDICTED HOLD: '+({strong:'ABOVE TYPICAL',weak:'BELOW TYPICAL',average:'TYPICAL'}[label]||label.toUpperCase())}),el('span',{class:'mut',text:V.basis||'Predicted curve summary.'}));box.append(vrow);
const byKey={};(R.factors||[]).forEach(f=>byKey[f.key]=f);
const card=(f,good)=>{const c=el('div',{class:'fcard'});c.style.setProperty('--c',good?'#5fd38a':'#ff5468');
const pctTxt=f.percentile_vs_reference==null?'':' · '+Math.round(f.percentile_vs_reference)+'th pct';
c.append(el('b',{text:f.label+': '+(f.value==null?'n/a':(typeof f.value==='number'?(+f.value.toFixed(2)):f.value)+(f.unit?' '+f.unit:''))+pctTxt}),el('div',{text:f.reading||''}),el('div',{class:'why',text:f.why_it_matters||''}));return c;};
const cols=el('div',{class:'cols2'});const a=el('div',{},el('div',{class:'mut',text:'Working for the hook'}));const b=el('div',{},el('div',{class:'mut',text:'Working against it'}));
(R.strengths||[]).forEach(k=>byKey[k]&&a.append(card(byKey[k],true)));(R.weaknesses||[]).forEach(k=>byKey[k]&&b.append(card(byKey[k],false)));
if(a.children.length===1)a.append(el('div',{class:'mut',text:'Nothing stood out as helping, against the other videos.'}));if(b.children.length===1)b.append(el('div',{class:'mut',text:'Nothing stood out as hurting, against the other videos.'}));cols.append(a,b);box.append(cols);
if(R.gemini_hook_assessment){box.append(el('div',{class:'fcard',style:'--c:#b47cf0'},el('b',{text:'AI on the opening (model interpretation)'}),el('div',{text:R.gemini_hook_assessment})));}
const det=el('details',{style:'margin-top:8px'},el('summary',{text:'All hook factors'}));const t=el('table');t.append(el('thead',{},el('tr',{},...['Factor','Value','Percentile','Reading'].map(h=>el('th',{text:h})))));const tb=el('tbody');
(R.factors||[]).forEach(f=>tb.append(el('tr',{},el('td',{text:f.label}),el('td',{class:'mono',text:f.value==null?'n/a':(typeof f.value==='number'?+f.value.toFixed(3):f.value)+(f.unit?' '+f.unit:'')}),el('td',{class:'mono',text:f.percentile_vs_reference==null?'–':Math.round(f.percentile_vs_reference)}),el('td',{text:(f.direction||'')+' · '+(f.reading||'')}))));t.append(tb);det.append(t);box.append(det);
box.append(el('p',{class:'mut',style:'margin:8px 0 0',text:R.note||''}));})();
// ---- strongest and weakest moments (from moments.json)
(function(){const B=D.brain;if(!B||B.status!=='ok')return;$('brainpanel').style.display='';$('brainsub').textContent=B.note;
if(B.reading){const br=$('brainread');br.style.display='';const top=(B.reading.top||[]).map(x=>x.label+' '+x.pct+'%').join(' + ');
const emo=B.reading.emotional_signal;const bits=[el('b',{text:'Dominant networks: '+top})];
if(emo)bits.push(el('div',{text:'Emotional signal: '+emo.text+' (Limbic '+emo.pct+'%)'}));
bits.push(el('div',{text:B.reading.summary||''}),el('div',{class:'why',text:B.reading.recommendation||''}),el('div',{class:'mut',text:B.reading.caveat||''}));
br.append(...bits);}
const NS='http://www.w3.org/2000/svg',W=760,H=240,L=40,R=10,T=10,Bm=26,pw=W-L-R,ph=H-T-Bm;const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e;};
const ts=B.timestamps_s,tmax=Math.max(1,ts[ts.length-1]);const all=Object.values(B.networks).flatMap(n=>n.curve).concat(B.attention_curve);const lo=Math.min(...all),hi=Math.max(...all),pad=(hi-lo)*0.08||0.1;
const x=t=>L+pw*t/tmax,y=v=>T+ph*(1-(v-(lo-pad))/((hi+pad)-(lo-pad)));const svg=mk('svg',{viewBox:'0 0 '+W+' '+H});
for(let i=0;i<=4;i++){const v=lo-pad+((hi+pad)-(lo-pad))*i/4;svg.append(mk('line',{x1:L,x2:L+pw,y1:y(v),y2:y(v),stroke:'#232838'}));const tx=mk('text',{x:L-6,y:y(v)+4,'text-anchor':'end','font-size':10,fill:'#8c92a6'});tx.textContent=v.toFixed(2);svg.append(tx);}
const step=tmax<=20?2:tmax<=60?10:20;for(let t=0;t<=tmax;t+=step){const tx=mk('text',{x:x(t),y:H-8,'text-anchor':'middle','font-size':10,fill:'#8c92a6'});tx.textContent=t+'s';svg.append(tx);}
const COL={Vis:'#b47cf0',SomMot:'#6aa0d8',DorsAttn:'#5fd38a',SalVentAttn:'#d96cf0',Limbic:'#d8e07a',Cont:'#e8a24a',Default:'#ff5468'};const legend=el('div',{class:'mut',style:'display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:4px'});
Object.entries(B.networks).forEach(([k,n])=>{svg.append(mk('polyline',{points:n.curve.map((v,i)=>x(ts[i])+','+y(v)).join(' '),fill:'none',stroke:COL[k]||'#888','stroke-width':1.8,opacity:.85}));legend.append(el('span',{},el('b',{text:'● ',style:'color:'+(COL[k]||'#888')}),document.createTextNode(n.label)));});
svg.append(mk('polyline',{points:B.attention_curve.map((v,i)=>x(ts[i])+','+y(v)).join(' '),fill:'none',stroke:'#e8eaf1','stroke-width':2.6,'stroke-dasharray':'5 3'}));legend.append(el('span',{},el('b',{text:'- - '}),document.createTextNode('Attention curve = mean of '+B.attention_networks.join(' + '))));
B.peaks.forEach(p=>{const i=ts.indexOf(p.t_s);if(i<0)return;svg.append(mk('circle',{cx:x(p.t_s),cy:y(B.attention_curve[i]),r:5,fill:'#ffcf47',stroke:'#0a0c12'}));});
$('brainchart').append(svg,legend);
const pk=$('brainpeaks');pk.append(document.createTextNode('Hook / surprise candidates (attention-curve peaks): '));B.peaks.forEach((p,i)=>{const a=el('a',{href:'#',text:p.t_s.toFixed(0)+' s ('+(p.height>=0?'+':'')+p.height.toFixed(1)+' sd)'});a.addEventListener('click',e=>{e.preventDefault();jump(p.t_s);});pk.append(a);if(i<B.peaks.length-1)pk.append(document.createTextNode(' · '));});
if(!B.peaks.length)pk.append(document.createTextNode('none'));})();
(function(){const M=D.moments;if(!M||!((M.strongest||[]).length||(M.weakest||[]).length))return;$('momentspanel').style.display='';
$('momentssub').textContent='Best and weakest '+M.window_s+' s windows after the hook'+(M.curve_is_stitched?' (approximate curve)':'')+'.';
const mk=(title,cls,list,color)=>{const col=el('div',{class:'mcol '+cls},el('h3',{text:title}));
if(!list.length)col.append(el('div',{class:'mut',text:M.note||'none found'}));
for(const m of list){const c=el('div',{class:'mcard'});c.style.setProperty('--c',color);
c.append(el('div',{},el('span',{class:'tm',text:mmss(m.start_s)+' – '+mmss(m.end_s)}),el('span',{class:'mut',text:'   retention '+pc(m.retention_start)+' → '+pc(m.retention_end)})));
const ul=el('ul');(m.why||[]).forEach(w=>ul.append(el('li',{text:w})));if(ul.children.length)c.append(ul);
if(m.suggested_action&&m.suggested_action.text)c.append(el('div',{class:'act',text:(m.suggested_action.action||'review').toUpperCase()+': '+m.suggested_action.text}));
c.addEventListener('click',()=>jump(m.start_s,m.end_s));col.append(c);}return col;};
$('moments').append(mk('Strongest moments','good',M.strongest||[],'#5fd38a'),mk('Weakest moments','bad',M.weakest||[],'#ff5468'));})();
// ---- what if: apply the accepted trims, predict the edited video (needs the upload page server)
const APPMODE=SERVED&&/^\/review\/UP[0-9a-z]+\/?$/.test(location.pathname),WID=APPMODE?location.pathname.replace(/^\/review\/|\/$/g,''):null;
function whatifState(){const n=acceptedTrims().length;const go=$('whatifgo');if(!APPMODE){go.disabled=true;$('whatifmsg').textContent='Available when you open an analysis from the upload page (python3 scripts/serve_app.py).';return;}
go.disabled=n===0;$('playedited').disabled=n===0;$('whatifmsg').textContent=n?(n+' accepted trim(s) will be applied'):'Accept at least one trim first, or use "Accept all suggested trims".';}
function renderWhatif(r){const out=$('whatifout');out.style.display='';out.textContent='';
const wrap=el('div',{style:'margin-top:10px'});const NS='http://www.w3.org/2000/svg',W=760,H=230,L=44,Rr=10,T=10,B=26,pw=W-L-Rr,ph=H-T-B;const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 '+W+' '+H);
const mkS=(t,a)=>{const e=document.createElementNS(NS,t);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e;};
const n=Math.max((r.original_retention||[]).length,(r.edited_retention||[]).length)-1;const x=(t)=>L+pw*t/Math.max(1,n),y=(v)=>T+ph*(1-v);
for(let i=0;i<=4;i++){const v=i/4;svg.append(mkS('line',{x1:L,x2:L+pw,y1:y(v),y2:y(v),stroke:'#232838'}));const tx=mkS('text',{x:L-8,y:y(v)+4,'text-anchor':'end','font-size':11,fill:'#8c92a6'});tx.textContent=Math.round(v*100)+'%';svg.append(tx);}
const step=n<=20?2:n<=60?10:20;for(let t=0;t<=n;t+=step){const tx=mkS('text',{x:x(t),y:H-8,'text-anchor':'middle','font-size':11,fill:'#8c92a6'});tx.textContent=t+'s';svg.append(tx);}
const line=(arr,color,dash)=>svg.append(mkS('polyline',{points:arr.map((v,t)=>x(t)+','+y(v)).join(' '),fill:'none',stroke:color,'stroke-width':2.2,'stroke-dasharray':dash||''}));
line(r.original_retention||[],'#5fd38a');line(r.edited_retention||[],'#b47cf0','5 3');wrap.append(svg);
wrap.append(el('div',{class:'mut',text:'Green: original (predicted). Purple dashed: edited (predicted), on the edited video\'s own timeline. Removed: '+(+r.removed_s).toFixed(1)+' s ('+(+r.original_duration_s).toFixed(1)+' s → '+(+r.edited_duration_s).toFixed(1)+' s).'}));
const t=el('table',{class:'wtable',style:'margin-top:8px'});t.append(el('thead',{},el('tr',{},...['Metric','Original','Edited','Change','vs noise'].map(h=>el('th',{text:h})))));const tb=el('tbody');
for(const m of (r.metrics||[])){const d=m.delta;tb.append(el('tr',{},el('td',{text:m.label}),el('td',{class:'mono',text:m.original==null?'–':(+m.original).toFixed(3)}),el('td',{class:'mono',text:m.edited==null?'–':(+m.edited).toFixed(3)}),
el('td',{class:'mono '+(m.within_noise?'noise':d>0?'pos':'neg'),text:d==null?'–':(d>0?'+':'')+(+d).toFixed(3)}),el('td',{class:'mut',text:m.within_noise?'within noise (3x re-encoding noise)':'beyond noise'})));}t.append(tb);wrap.append(t);
wrap.append(el('div',{class:'fcard',style:'--c:#ffcf47;margin-top:10px'},el('b',{text:'Predicted effect: '+(r.verdict||'no clear difference')}),...(r.caveats||[]).map(c=>el('div',{class:'why',text:c}))));
for(const w of (r.warnings||[]))wrap.append(el('div',{class:'mut',text:'Note: '+w}));out.append(wrap);}
async function whatifPoll(){try{const res=await fetch('/api/whatif/'+WID);const j=await res.json();
if(j.status==='running'){$('whatifmsg').textContent='predicting the edited video… '+j.seconds+' s';setTimeout(whatifPoll,2500);return;}
$('whatifgo').disabled=false;if(j.status==='failed'){$('whatifmsg').textContent='failed: '+j.error;return;}$('whatifmsg').textContent='done in '+j.seconds+' s';renderWhatif(j.result);}catch(e){$('whatifmsg').textContent='lost contact with the local server';}}
$('acceptall').addEventListener('click',()=>{let k=0;for(const sg of SUGS){if(sg.c.edit.action!=='trim')continue;const a=sg.c.edit.start_s,b=sg.c.edit.end_s;if(!(b>a))continue;store[sg.id]=Object.assign(store[sg.id]||{},{decision:'accepted'});k++;}persist();if(!k){$('whatifmsg').textContent='There are no suggested trims on this video.';return;}try{location.reload();}catch(e){refreshEdited();}});
$('whatifgo').addEventListener('click',async()=>{$('whatifgo').disabled=true;$('whatifmsg').textContent='starting…';
try{const res=await fetch('/api/whatif/'+WID,{method:'POST',headers:{'X-Hookline':'1','Content-Type':'application/json'},body:JSON.stringify({trims:acceptedTrims().map(r=>({start_s:r.a,end_s:r.b}))})});
const j=await res.json();if(!res.ok){$('whatifmsg').textContent=j.error||'could not start';$('whatifgo').disabled=false;return;}whatifPoll();}catch(e){$('whatifmsg').textContent='could not start: '+e;$('whatifgo').disabled=false;}});
if(APPMODE){fetch('/api/whatif/'+WID).then(r=>r.ok?r.json():null).then(j=>{if(j&&j.status==='done'&&j.result)renderWhatif(j.result);}).catch(()=>{});}
// ---- hook tiles
const tiles=$('tiles');const H=D.hook||{};const secKind=(t)=>{const s=D.seconds.find(x=>x.t===t);return s?s.kind:null;};
const KC={steep:'#ff5468',decline:'#ffcf47',hold:'#5fd38a',gradual:'#b47cf0'};
for(const t of [1,3,5,10]){const r=H['R'+t];const tile=el('div',{class:'tile',title:r==null?'not predicted (video or analysed opening shorter)':'click to jump to '+t+' s'});
const bar=el('div',{class:'bar'},el('b',{}));bar.firstChild.style.width=r==null?'0%':Math.round(100*r)+'%';tile.style.setProperty('--c',KC[secKind(t)]||'#5fd38a');
tile.append(el('div',{class:'t',text:'Watching at '+t+' s'}),el('div',{class:'v mono',text:pc(r)}),bar);tile.addEventListener('click',()=>{if(r!=null)jump(Math.max(0,t-1));});tiles.append(tile);}
const sd=H.steepest_early_drop;$('hooknote').textContent=sd?('Steepest predicted drop in the first '+H.early_window_s+' s: '+sd.from_s+'→'+sd.to_s+' s, '+Math.round(100*sd.drop)+' points.'):'';
// ---- timeline strip, chart, film, playhead
const strip=$('strip');const pct=(t)=>(100*Math.min(1,Math.max(0,t/TOTAL)))+'%';
const segEls=D.segments.map(sg=>{const e=el('div',{class:'seg '+sg.kind,title:sg.kind.replace('_',' ')+' '+sg.start_s+'-'+sg.end_s+' s: predicted fall '+sg.predicted_loss.toFixed(3)});e.style.left=pct(sg.start_s);e.style.width=(100*(sg.end_s-sg.start_s)/TOTAL)+'%';strip.append(e);return e;});
if(D.horizon_s<TOTAL-0.5){const n=el('div',{class:'none',title:'no prediction here'});n.style.left=pct(D.horizon_s);n.style.right='0';strip.append(n);}
const trimEls=[];const head=el('div',{class:'head'});strip.append(head);
strip.addEventListener('click',(ev)=>{const r=strip.getBoundingClientRect();jump(TOTAL*(ev.clientX-r.left)/r.width);});
let chartHead=null,chartX=null;
function drawChart(logScale){const svg=$('chart');svg.textContent='';const NS='http://www.w3.org/2000/svg',W=760,Hh=250,L=44,R=10,T=12,B=28,pw=W-L-R,ph=Hh-T-B,n=D.retention.length-1;
const LOGMIN=-3,pos=(r)=>logScale?(Math.log10(Math.max(r,1e-3))-LOGMIN)/(-LOGMIN):r;const x=(t)=>L+pw*t/n,y=(r)=>T+ph*(1-pos(r));chartX=x;const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e;};
const defs=mk('defs',{});const gr=mk('linearGradient',{id:'gfill',x1:0,y1:0,x2:0,y2:1});gr.append(mk('stop',{offset:'0%','stop-color':'#5fd38a','stop-opacity':0.35}),mk('stop',{offset:'100%','stop-color':'#5fd38a','stop-opacity':0}));defs.append(gr);svg.append(defs);
const GRID=logScale?[1,0.1,0.01,0.001]:[0,0.25,0.5,0.75,1];for(const r of GRID){svg.append(mk('line',{x1:L,x2:L+pw,y1:y(r),y2:y(r),stroke:'#232838','stroke-width':1}));const tx=mk('text',{x:L-8,y:y(r)+4,'text-anchor':'end','font-size':11,fill:'#8c92a6','font-family':'ui-monospace,Menlo,monospace'});tx.textContent=(logScale?(+(r*100).toPrecision(2)):Math.round(r*100))+'%';svg.append(tx);}
const step=n<=10?1:n<=40?5:10;for(let t=0;t<=n;t+=step){svg.append(mk('line',{x1:x(t),x2:x(t),y1:T,y2:T+ph,stroke:'#1a1e2b','stroke-width':1}));const tx=mk('text',{x:x(t),y:Hh-8,'text-anchor':'middle','font-size':11,fill:'#8c92a6','font-family':'ui-monospace,Menlo,monospace'});tx.textContent=t+'s';svg.append(tx);}
for(const iv of D.intervals){svg.append(mk('rect',{class:'zone',x:x(iv.start_s),y:T,width:Math.max(2,x(iv.end_s)-x(iv.start_s)),height:ph,fill:COLOR[iv.kind]||'#5b6276','fill-opacity':'0.07'}));}
if(D.ref_band){const B=D.ref_band,m=Math.min(n,B.p50.length-1);const up=[],dn=[],md=[];for(let t=0;t<=m;t++){up.push(x(t)+','+y(B.p90[t]));dn.unshift(x(t)+','+y(B.p10[t]));md.push(x(t)+','+y(B.p50[t]));}
const bp=mk('polygon',{class:'band',points:up.concat(dn).join(' '),fill:'#8c92a6','fill-opacity':0.18});const bt=mk('title',{});bt.textContent='Where 80% of the other videos\' predicted curves fall';bp.append(bt);svg.append(bp);
svg.append(mk('polyline',{class:'band',points:md.join(' '),fill:'none',stroke:'#8c92a6','stroke-width':1.4,'stroke-dasharray':'5 4'}));
if(B.blank){const bl=[];for(let t=0;t<=Math.min(n,B.blank.length-1);t++)bl.push(x(t)+','+y(B.blank[t]));svg.append(mk('polyline',{class:'band',points:bl.join(' '),fill:'none',stroke:'#ff5468','stroke-width':1.2,'stroke-dasharray':'1.5 3.5','stroke-linecap':'round'}));}}
const pts=D.retention.map((r,t)=>x(t)+','+y(r));svg.append(mk('polygon',{points:x(0)+','+(T+ph)+' '+pts.join(' ')+' '+x(n)+','+(T+ph),fill:'url(#gfill)'}));
svg.append(mk('polyline',{points:pts.join(' '),fill:'none',stroke:'#5fd38a','stroke-width':2.2,'stroke-linejoin':'round'}));
D.retention.forEach((r,t)=>{const c=mk('circle',{cx:x(t),cy:y(r),r:6,fill:'transparent',style:'cursor:pointer'});const ti=mk('title',{});ti.textContent='t='+t+' s, predicted R='+r.toFixed(3)+' — click to jump';c.append(ti);c.addEventListener('click',()=>jump(t));svg.append(c);});
for(const iv of D.intervals){if(iv.kind==='holds')continue;const t=iv.start_s;const c=mk('circle',{class:'mark',cx:x(t),cy:y(D.retention[Math.min(t,n)]),r:5.5,fill:COLOR[iv.kind],stroke:'#0a0c12','stroke-width':2,style:'cursor:pointer'});const ti=mk('title',{});ti.textContent=iv.kind.replace('_',' ')+' '+iv.start_s+'–'+iv.end_s+' s';c.append(ti);c.addEventListener('click',()=>{select(iv.id);jump(iv.start_s);});svg.append(c);}
chartHead=mk('line',{x1:L,x2:L,y1:T,y2:T+ph,stroke:'#ffffff','stroke-width':1.5,'stroke-dasharray':'3 3'});svg.append(chartHead);}
drawChart(false);(function(){const B=D.ref_band;if(!B)return;const m=Math.min(10,D.retention.length-1,B.p50.length-1);if(m<3)return;let inn=0,above=0,below=0;for(let t=1;t<=m;t++){const r=D.retention[t];if(r>B.p90[t])above++;else if(r<B.p10[t])below++;else inn++;}
const el2=$('bandverdict');const f=(inn/m);
el2.textContent=f>=0.8?'Typical predicted hold over the first '+m+' s.':(above>=below?'Above-typical predicted hold for part of the first '+m+' s.':'Below-typical predicted hold for part of the first '+m+' s.');})();
if(!D.ref_band){const bn=$('bandnote');if(bn)bn.style.display='none';}
(function(){const b=$('logtoggle');let lg=false;b.addEventListener('click',()=>{lg=!lg;drawChart(lg);b.textContent=lg?'Show normal scale':'Zoom into the tail (log scale)';$('logscalenote').style.display=lg?'':'none';});})();
const film=$('film');const NF=Math.max(4,Math.min(8,Math.round(TOTAL/5)));const filmEls=[];
for(let i=0;i<NF;i++){const t=TOTAL*(i+0.5)/NF;const c=el('canvas',{width:320,height:200});const f=el('div',{class:'fr',title:'jump to '+fmt(t)},c,el('span',{text:mmss(t)}));f.addEventListener('click',()=>jump(t));film.append(f);filmEls.push({f,t});frameAt(c,t);}
// ---- segment list and the detail panel
const LABEL={steep_drop:'Major drop',sustained_decline:'Decline',holds:'Holds',unflagged:'Steady'};
function kindLabel(k){return {steep_drop:'Steep predicted drop',sustained_decline:'Sustained predicted decline',holds:'Predicted to hold',unflagged:'No predicted drop'}[k]||k;}
const ivOf=(sg)=>D.intervals.find(iv=>iv.start_s===sg.start_s&&iv.end_s===sg.end_s&&iv.kind===sg.kind);
function shortDesc(sg,iv){const bits=['R '+pc(sg.retention_start)+' → '+pc(sg.retention_end)];if(sg.share_of_total_predicted_loss!=null)bits.push(Math.round(100*sg.share_of_total_predicted_loss)+'% of the predicted fall');
const n=iv?iv.candidate_causes.length:0;if(n)bits.push(n+' proposal'+(n>1?'s':''));if(iv&&iv.gemini)bits.push('AI notes');return bits.join(' · ');}
const rows=[];const seglist=$('seglist');
D.segments.forEach((sg,i)=>{const iv=ivOf(sg);const r=el('div',{class:'srow '+sg.kind});r.style.setProperty('--c',COLOR[sg.kind]);
const lab=el('div',{},el('span',{class:'lb',text:(sg.start_s===0?'Hook · ':'')+(LABEL[sg.kind]||sg.kind)}),el('span',{class:'ds',text:shortDesc(sg,iv)}));
const btn=iv?(sg.kind==='holds'&&!iv.candidate_causes.length?el('button',{class:'keep small',text:'Keep'}):el('button',{class:'ghost small',text:'View details'})):el('button',{class:'keep small',text:'Jump to '+mmss(sg.start_s)});
btn.addEventListener('click',(ev)=>{ev.stopPropagation();if(iv){select(iv.id);}else jump(sg.start_s);});
r.append(el('span',{class:'dot'}),el('span',{class:'tm',text:mmss(sg.start_s)+' – '+mmss(sg.end_s)}),lab,btn);
r.addEventListener('click',()=>{if(iv)select(iv.id);jump(sg.start_s);});seglist.append(r);rows.push({r,sg,iv});});
$('navsegs').textContent=String(D.segments.length);
function obsList(o){const li=[];const w=o.words,t=o.on_screen_text,c=o.scene_cuts,q=o.quiet_audio;
li.push('Speech: '+(w==='not run'?'not run':w.count+' word(s)'+(w.count?'':' — no speech')));
li.push('On-screen text: '+(t==='not run'?'not run':t.length?t.map(x=>x.text).join(', '):'none seen'));
li.push('Scene cuts: '+(c==='not run'?'not run':c.count+(c.count?' (at '+c.times_s.map(x=>x.toFixed(1)).join(', ')+' s)':'')));
li.push('Quiet audio: '+(q==='not run'?'not run':q.length?q.length+' interval(s)':'none'));
if(o.motion_mean!=null)li.push('Mean motion: '+o.motion_mean);return li;}
function chips(o){const out=[];const c=o.scene_cuts,t=o.on_screen_text,q=o.quiet_audio,w=o.words;
if(c!=='not run')out.push(c.count?c.count+' scene cut'+(c.count>1?'s':''):'No scene cut');if(t!=='not run')out.push(t.length?'Text on screen':'No text on screen');
if(q!=='not run'&&q.length)out.push('Quiet audio');if(w!=='not run'&&!w.count)out.push('No speech');if(o.motion_mean!=null)out.push('Motion '+o.motion_mean);return out;}
const ACTION={trim:'TRIM',add:'ADD',review:'REVIEW'};
const segs=$('segs');const cards=[];let selected=null;
D.intervals.forEach((iv)=>{const card=el('div',{class:'card '+iv.kind});card.style.setProperty('--c',COLOR[iv.kind]);card.dataset.id=iv.id;cards.push(card);
card.append(el('div',{class:'dhead'},el('span',{class:'tm',text:mmss(iv.start_s)+' – '+mmss(iv.end_s)}),el('span',{class:'badge',text:kindLabel(iv.kind)})));
const th=el('div',{class:'thumbs'});const tt=[iv.start_s+0.2,(iv.start_s+iv.end_s)/2,Math.max(iv.start_s,iv.end_s-0.3)];const tcs=tt.map(t=>{const c=el('canvas',{width:320,height:200,title:'jump to '+fmt(t)});c.addEventListener('click',()=>jump(t));th.append(c);return c;});card.append(th);card._thumbs={tcs,tt,done:false};
const nav=el('div',{class:'row-actions',style:'margin-top:6px'});const b1=el('button',{class:'primary small',text:'▶ Jump to '+fmt(iv.start_s)});b1.addEventListener('click',()=>jump(iv.start_s));
const b2=el('button',{class:'small',text:'Play this segment'});b2.addEventListener('click',()=>jump(iv.start_s,iv.end_s));nav.append(b1,b2);card.append(nav);
const what=el('div',{class:'sec'},el('h4',{},el('i',{}),'What was measured here'));const ul=el('ul');for(const s of obsList(iv.observed))ul.append(el('li',{text:s}));what.append(ul);card.append(what);
const ev=el('div',{class:'sec'},el('h4',{},el('i',{}),'Evidence'));const w=iv.observed.words;
if(w!=='not run'&&w.count&&w.text){ev.append(el('div',{class:'mut',text:'Transcript'}),el('div',{class:'quote',text:'“'+w.text+'”'}));}
ev.append(el('div',{class:'mut',text:'Frame and audio analysis'}));const ch=el('div',{class:'chips'});for(const c of chips(iv.observed))ch.append(el('span',{text:c}));ev.append(ch);card.append(ev);
if(iv.gemini){const gb=el('div',{class:'sec'},el('h4',{},el('i',{}),'AI: what it observed (model interpretation)'));if(iv.gemini.summary)gb.append(el('div',{text:iv.gemini.summary}));
const gl=el('ul');for(const o of iv.gemini.observations)gl.append(el('li',{text:(o.kind==='observation'?'Observed: ':'Interpretation: ')+o.text+(o.evidence_refs.length?'  ['+o.evidence_refs.join(', ')+']':'')}));gb.append(gl);
if(iv.gemini.hypothesis)gb.append(el('div',{class:'mut',text:'Hypothesis'+(iv.gemini.hypothesis_confidence!=null?' (confidence '+iv.gemini.hypothesis_confidence.toFixed(2)+', how well the cited evidence supports it, not a retention probability)':'')+': '+iv.gemini.hypothesis}));card.append(gb);}
const pr=el('div',{class:'sec'},el('h4',{},el('i',{}),'Predicted retention (model prediction)'));
pr.append(el('div',{},el('span',{class:'mono',style:'font-size:18px;font-weight:650',text:pc(iv.retention_start)+' → '+pc(iv.retention_end)}),el('span',{class:'mut',text:'   '+(iv.share_of_total_predicted_loss!=null?Math.round(100*iv.share_of_total_predicted_loss)+'% of the predicted fall':'')})));
const rb=el('div',{class:'rbar'});const b_a=el('b',{});b_a.style.left='0';b_a.style.width=Math.round(100*iv.retention_end)+'%';b_a.style.background='var(--c)';const b_b=el('b',{});b_b.style.left=Math.round(100*iv.retention_end)+'%';b_b.style.width=Math.max(0,Math.round(100*(iv.retention_start-iv.retention_end)))+'%';b_b.style.background='color-mix(in srgb,var(--c) 30%,transparent)';rb.append(b_a,b_b);pr.append(rb);
pr.append(el('div',{class:'mut',text:'Share of viewers predicted to still watch at the start → end of this segment. Not audience data; an edit\'s effect is not predicted.'}));card.append(pr);
const rec=el('div',{class:'sec'},el('h4',{},el('i',{}),'Recommendations'));card.append(rec);
if(iv.kind==='holds'&&!iv.candidate_causes.length){rec.append(el('div',{class:'mut',text:'Retention is predicted to stay nearly flat here: keep what holds.'}));}
else if(!iv.candidate_causes.length){rec.append(el('div',{class:'mut',text:'No rule matched this segment. Review the measured items above by hand.'}));}
SUGS.filter(sg=>sg.iv===iv).forEach((sg)=>{const c=sg.c,e=c.edit,s=cur(sg);const box=el('div',{class:'sug'+(c.source==='gemini'?' gem':'')});
box.append(el('div',{},el('span',{class:'tag',text:ACTION[e.action]}),document.createTextNode(c.suggestion)));
if(c.source==='gemini'){const bits=['AI (model interpretation, hypothesis)'];if(c.confidence!=null)bits.push('confidence '+c.confidence.toFixed(2));box.append(el('div',{class:'mut',text:bits.join(' · ')}));
if(c.rationale)box.append(el('div',{class:'mut',text:'Why: '+c.rationale}));if(c.expected_effect)box.append(el('div',{class:'mut',text:'Expected effect (hypothesis): '+c.expected_effect}));
if(c.evidence_refs.length)box.append(el('div',{class:'mut',text:'Evidence: '+c.evidence_refs.join(', ')}));}
else box.append(el('div',{class:'mut',text:'Unvalidated hypothesis from the rule “'+c.rule+'”.'}));
const row=el('div',{class:'row-actions',style:'margin-top:8px'});
const save=(patch)=>{store[sg.id]=Object.assign(store[sg.id]||{},patch,{});persist();refreshEdited();};
let ia=null,ib=null;
if(e.action==='trim'){ia=el('input',{type:'number',step:'0.1',min:'0',value:String(+s.a.toFixed(2))});ib=el('input',{type:'number',step:'0.1',min:'0',value:String(+s.b.toFixed(2))});
ia.addEventListener('input',()=>save({a:parseFloat(ia.value)}));ib.addEventListener('input',()=>save({b:parseFloat(ib.value)}));
row.append(el('span',{class:'mut',text:'Original '+fmt(iv.start_s)+'–'+fmt(iv.end_s)+' → trim'}),ia,el('span',{class:'mut',text:'to'}),ib,el('span',{class:'mut',text:'s'}));const pv=el('button',{class:'primary small',text:'Preview trim'});pv.addEventListener('click',()=>previewTrim(parseFloat(ia.value),parseFloat(ib.value)));row.append(pv);}
const jb=el('button',{class:'small',text:'Jump to '+fmt(e.start_s)});jb.addEventListener('click',()=>jump(e.start_s));row.append(jb);box.append(row);
const radios=el('div',{class:'radios'});
for(const[val,lab]of [['accepted','Accept'],['rejected','Reject'],['unsure','Unsure']]){const inp=el('input',{type:'radio',name:sg.id,value:val});if(s.decision===val)inp.checked=true;inp.addEventListener('change',()=>save({decision:val}));radios.append(el('label',{},inp,lab));}
box.append(radios);const ta=el('textarea',{rows:2,placeholder:'Your note (what you will change, or why not)'});ta.value=s.note||'';ta.addEventListener('input',()=>save({note:ta.value}));box.append(ta);rec.append(box);});
const exp=el('div',{class:'sec'},el('h4',{},el('i',{}),'Export this segment'));const ej=el('button',{class:'small',text:'JSON'});const ec=el('button',{class:'small',text:'CSV'});
ej.addEventListener('click',()=>download('hookline_'+D.video_id+'_'+iv.id+'.json','application/json',JSON.stringify({video_id:D.video_id,disclaimer:DISCLAIMER,segment:iv,edits:SUGS.filter(sg=>sg.iv===iv).map(exportRow)},null,2)));
ec.addEventListener('click',()=>download('hookline_'+D.video_id+'_'+iv.id+'.csv','text/csv',toCSV(SUGS.filter(sg=>sg.iv===iv).map(exportRow))));
exp.append(el('div',{class:'row-actions',style:'margin-top:2px'},ej,ec));card.append(exp);
segs.append(card);});
function select(id){selected=id;cards.forEach(c=>{const on=c.dataset.id===id;c.classList.toggle('sel',on);if(on&&!c._thumbs.done){c._thumbs.done=true;c._thumbs.tcs.slice().reverse().forEach((cv,j)=>{const i=c._thumbs.tcs.length-1-j;frameAt(cv,c._thumbs.tt[i],true);});}});
rows.forEach(x=>x.r.classList.toggle('on',!!x.iv&&x.iv.id===id));const i=D.intervals.findIndex(iv=>iv.id===id);$('detailpos').textContent=i>=0?(i+1)+' / '+D.intervals.length:'';}
if(D.intervals.length){const first=D.intervals.find(iv=>iv.kind!=='holds')||D.intervals[0];select(first.id);}else{$('nodetail').style.display='block';$('nodetail').textContent='No candidate segment was flagged on this video.';}
$('playedited').addEventListener('click',playEdited);
// ---- sync
function sync(){const t=SRC?V.currentTime:0;$('now').textContent=fmt(t)+' / '+fmt(TOTAL);head.style.left=pct(t);
if(chartHead&&t<=D.horizon_s){const xx=chartX(Math.min(t,D.retention.length-1));chartHead.setAttribute('x1',xx);chartHead.setAttribute('x2',xx);}
let act=null;D.segments.forEach((sg,i)=>{const on=t>=sg.start_s&&t<sg.end_s;segEls[i].classList.toggle('active',on);rows[i].r.classList.toggle('playing',on);if(on)act=sg;});
D.intervals.forEach((iv,i)=>{cards[i].classList.toggle('active',t>=iv.start_s&&t<iv.end_s);});
filmEls.forEach((x,i)=>{const a=TOTAL*i/NF,b=TOTAL*(i+1)/NF;x.f.classList.toggle('on',t>=a&&t<b);});
$('activeinfo').textContent=act?('inside: '+act.kind.replace('_',' ')+' '+act.start_s+'–'+act.end_s+' s (predicted R '+act.retention_start.toFixed(2)+' → '+act.retention_end.toFixed(2)+', rank '+act.rank_by_predicted_loss+' by predicted fall)'):'';}
function refreshEdited(){whatifState();const tr=acceptedTrims();trimEls.forEach(e=>e.remove());trimEls.length=0;for(const r of tr){const e=el('div',{class:'trim',title:'accepted trim '+r.a.toFixed(1)+'–'+r.b.toFixed(1)+' s'});e.style.left=pct(r.a);e.style.width=(100*(r.b-r.a)/TOTAL)+'%';strip.insertBefore(e,head);trimEls.push(e);}
const rem=tr.reduce((a,r)=>a+(r.b-r.a),0);$('editedinfo').textContent=tr.length?(tr.length+' accepted trim(s) remove '+rem.toFixed(1)+' s: '+TOTAL.toFixed(1)+' s → '+(TOTAL-rem).toFixed(1)+' s'):'no accepted trims yet';}
setInterval(()=>{tick();sync();},100);V.addEventListener('seeked',sync);
// ---- scores table
const KINDS={steep_drop:'Steep drop',sustained_decline:'Sustained decline',holds:'Holds',unflagged:'No action'};
const ms=(a)=>a.mean==null?'n/a':a.mean.toFixed(1)+' / '+a.max.toFixed(1);
const prettyR=(a,b)=>a<0.01&&b<0.01?'<1% tail':pc(a)+' → '+pc(b);
const prettyLoss=(sg)=>sg.retention_start<0.01&&sg.retention_end<0.01?'no meaningful viewer share left':sg.predicted_loss.toFixed(3)+(sg.share_of_total_predicted_loss==null?'':' ('+Math.round(100*sg.share_of_total_predicted_loss)+'%)');
function segmentRows(){return D.segments.map(sg=>({video_id:D.video_id,segment_id:sg.id,kind:sg.kind,start_s:sg.start_s,end_s:sg.end_s,length_s:sg.length_s,predicted_R_start:+sg.retention_start.toFixed(4),predicted_R_end:+sg.retention_end.toFixed(4),predicted_fall:+sg.predicted_loss.toFixed(4),share_of_total_predicted_fall:sg.share_of_total_predicted_loss==null?'':+sg.share_of_total_predicted_loss.toFixed(4),rank_by_predicted_fall:sg.rank_by_predicted_loss,heuristic_risk_mean:sg.heuristic.attention_risk.mean==null?'':sg.heuristic.attention_risk.mean,heuristic_risk_max:sg.heuristic.attention_risk.max==null?'':sg.heuristic.attention_risk.max,heuristic_quality_mean:sg.heuristic.moment_quality.mean==null?'':sg.heuristic.moment_quality.mean,heuristic_quality_max:sg.heuristic.moment_quality.max==null?'':sg.heuristic.moment_quality.max,data_kind:'predicted_retention_unvalidated'}));}
function toCSVRows(rows){const cols=Object.keys(rows[0]||{});return [cols.join(',')].concat(rows.map(r=>cols.map(c=>csvCell(r[c])).join(','))).join('\r\n')+'\r\n';}
// ---- export
function download(name,type,text){const a=el('a',{href:URL.createObjectURL(new Blob([text],{type:type})),download:name});document.body.append(a);a.click();a.remove();}
$('exportjson').addEventListener('click',()=>download('hookline_edits_'+D.video_id+'.json','application/json',JSON.stringify(buildExport(),null,2)));
$('exportcsv').addEventListener('click',()=>download('hookline_edits_'+D.video_id+'.csv','text/csv',toCSV(buildExport().edits)));
$('clear').addEventListener('click',()=>{store={};try{localStorage.removeItem(KEY);}catch(e){}location.reload();});
$('foot').textContent='Evidence sources: '+Object.entries(D.evidence).map(([k,v])=>k+' '+v).join(', ')+'. Your decisions stay in this browser unless you export them. Preview skipping is checked every 0.1 s, so it can overshoot a trim edge slightly.';
// ---- nav highlight
const navLinks=[...document.querySelectorAll('#nav a')];navLinks.forEach(a=>a.addEventListener('click',()=>{navLinks.forEach(b=>b.classList.toggle('on',b===a));}));
refreshEdited();sync();
// ---- video check (open with #videocheck): reports whether the browser loaded the video
if(location.hash==='#videocheck'){setTimeout(()=>{const pre=el('pre',{id:'videocheck'});pre.textContent='VIDEOCHECK readyState='+V.readyState+' duration='+V.duration+' error='+(V.error?V.error.code:'none')+' frames_painted='+PAINTED+' queued='+FQ.length;document.body.append(pre);},3000);}
// ---- self-test (open the page with #selftest): exercises the logic without a person clicking
if(location.hash==='#selftest'){const out=[];const ok=(c,m)=>out.push((c?'PASS ':'FAIL ')+m);
ok(skipTarget(3.0,[{a:2,b:5}])===5&&skipTarget(1.0,[{a:2,b:5}])===null&&skipTarget(5.0,[{a:2,b:5}])===null,'skipTarget skips inside a range only');
ok(toCSV([{video_id:'v',note:'a "q", b\nc'}]).includes('"a ""q"", b\nc"'),'CSV quoting');
ok(document.querySelectorAll('.card').length===D.intervals.length,'one card per segment');
ok(document.querySelectorAll('.card.sel').length===(D.intervals.length?1:0),'exactly one segment shown in the detail panel');
ok(document.querySelectorAll('#seglist .srow').length===D.segments.length,'one list row per scored segment');
ok(document.querySelectorAll('#strip .seg').length===D.segments.length,'one strip band per scored segment');
ok(D.segments[0].start_s===0&&D.segments[D.segments.length-1].end_s===D.horizon_s&&D.segments.every((sg,i)=>i===0||sg.start_s===D.segments[i-1].end_s),'segments cover the predicted span with no gaps');
ok(toCSVRows(segmentRows()).split('\r\n').length===D.segments.length+2,'segment-score CSV has header + rows');
ok(Math.abs(D.segments.reduce((a,sg)=>a+(sg.share_of_total_predicted_loss||0),0)-1)<0.01,'shares of the predicted fall add up to 1');
ok(document.querySelectorAll('#tiles .tile').length===4,'four hook tiles');
ok(!(D.hook_report&&D.hook_report.status==='ok')||document.querySelectorAll('#hookwhy .vb').length===1,'hook verdict shown when the hook report exists');
ok(!(D.moments&&((D.moments.strongest||[]).length||(D.moments.weakest||[]).length))||document.querySelectorAll('#moments .mcard').length===(D.moments.strongest||[]).length+(D.moments.weakest||[]).length,'one card per moment');
ok($('whatifgo')!==null&&$('whatifgo').disabled===true,'what-if is disabled until a trim is accepted (and outside the upload page)');
ok(!(D.scores&&D.scores.status==='ok')||document.querySelectorAll('.gauge').length===3,'three score gauges when scores exist');
if(D.intervals.length>1){const last=D.intervals[D.intervals.length-1];select(last.id);ok(document.querySelector('.card.sel').dataset.id===last.id,'selecting a segment shows its detail');select(D.intervals[0].id);}
const first=SUGS.find(sg=>sg.c.edit.action==='trim');
if(first){store[first.id]={decision:'accepted',a:1,b:2.5,note:'x'};refreshEdited();const ex=buildExport();ok(ex.accepted_trims.length===1&&ex.accepted_trims[0].start_s===1&&ex.accepted_trims[0].end_s===2.5,'accepted trim reaches the export');
ok(Math.abs(ex.duration_after_accepted_trims_s-(TOTAL-1.5))<0.01,'duration after trims');ok(ex.disclaimer.includes('PREDICTED')&&ex.retention_is_prediction===true,'export is labelled as a prediction');
ok(ex.edits.every(e=>e.data_kind==='predicted_retention_unvalidated'),'every row is labelled');ok(toCSV(ex.edits).split('\r\n').length===ex.edits.length+2,'CSV has header + rows');store={};refreshEdited();}
else ok(true,'no trim proposals on this video (skipped trim checks)');
ok(SUGS.every(sg=>['trim','add','review'].includes(sg.c.edit.action)),'every proposal has a known action');
const pre=el('pre',{id:'selftest'});pre.textContent=(out.every(l=>l.startsWith('PASS'))?'SELFTEST PASS':'SELFTEST FAIL')+'\n'+out.join('\n');document.body.append(pre);}

/* three views: Results, Analytics, Edit Review (same DOM, shown per view) */
const VIEWS=['analysis','insights','edit'];
function setView(v,quiet){if(!VIEWS.includes(v))v='analysis';document.body.setAttribute('data-view',v);document.querySelectorAll('#nav a').forEach(a=>a.classList.toggle('on',a.dataset.v===v));if(!quiet){try{history.replaceState(null,'','#'+v);}catch(e){}}window.scrollTo(0,0);}
document.querySelectorAll('#nav a').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();setView(a.dataset.v);}));
const RANK={steep_drop:0,sustained_decline:1,holds:2,unflagged:3};
const TOP=SUGS.slice().sort((x,y)=>(RANK[x.iv.kind]??4)-(RANK[y.iv.kind]??4)||((y.c.confidence||0)-(x.c.confidence||0))).slice(0,3);
const tl=$('topsugslist');if(!TOP.length){tl.append(el('p',{class:'mut',text:'No edits suggested.'}));}
TOP.forEach(sg=>{const c=sg.c,e=c.edit;const row=el('div',{class:'tsug'});
const b1=el('button',{class:'small primary',text:'Play'});b1.addEventListener('click',()=>{jump(sg.iv.start_s,sg.iv.end_s);});
const b2=el('button',{class:'small ghost',text:'Details'});b2.addEventListener('click',()=>{setView('edit');select(sg.iv.id);});
row.append(el('b',{text:mmss(sg.iv.start_s)+'–'+mmss(sg.iv.end_s)+'  '+c.suggestion}),el('div',{class:'acts'},b1,b2),
el('div',{class:'why'},el('span',{class:'tag'+(c.source==='gemini'?' ai':''),text:c.source==='gemini'?'AI SUGGESTION':'RULE-BASED'}),c.rationale||''));tl.append(row);});
setView((location.hash||'').slice(1),true);
</script></body></html>
"""
