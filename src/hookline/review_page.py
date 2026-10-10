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

from .segment_recommendations import build_segment_recommendations
from .ttcc import hook_metrics

REVIEW_FILE = "review.html"


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _embed(data: dict) -> str:
    """JSON safe inside a <script> tag."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")


def build_review(video_dir: Path, video_path: Path | None = None) -> Path | None:
    """Write ``review.html`` next to the artifacts. None without a drop-off analysis."""
    dr, art = _load(video_dir / "ttcc_dropoffs.json"), _load(video_dir / "ttcc_retention.json")
    if not dr or not art:
        return None
    rec = build_segment_recommendations(video_dir)  # TTCC segments + rule proposals + Gemini (when it ran)
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
@media (max-width:1180px){.grid{grid-template-columns:minmax(0,1fr)}.sticky{position:static}}
@media (max-width:860px){.app{grid-template-columns:minmax(0,1fr)}aside{position:static;height:auto;flex-direction:row;flex-wrap:wrap;align-items:center;padding:12px 16px;gap:10px}
nav{flex-direction:row;flex-wrap:wrap}nav a{padding:5px 9px}nav a span.k{display:none}.side-foot{display:none}main{padding:14px 16px 32px;overflow-x:hidden}
header.top{flex-wrap:wrap}.top .right{margin-left:0}.panel{padding:13px}.panel h2{flex-wrap:wrap}.legend{margin-left:0}.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}
.film{grid-auto-flow:row;grid-template-columns:repeat(4,minmax(0,1fr))}.srow{grid-template-columns:12px auto minmax(0,1fr) auto;gap:8px;padding:10px 4px}.srow .ds{display:none}.srow .lb{font-size:12.5px}
.status{flex-direction:column;gap:2px}.dhead .tm{font-size:17px}}
</style></head><body>
<div class="app">
<aside>
  <div class="brand"><div class="mark"><i></i><i></i><i></i><i></i></div><b>Hookline</b><span class="by"><span class="w">W</span><span class="l">L</span><span class="d1">D</span><span class="d2">D</span></span></div>
  <nav id="nav">
    <a href="#editor" class="on">Editor<span class="k">E</span></a>
    <a href="#hookpanel">Hook<span class="k">0–10s</span></a>
    <a href="#retention">Retention<span class="k">R(t)</span></a>
    <a href="#segments">Segments<span class="k" id="navsegs"></span></a>
    <a href="#scores">Scores</a>
    <a href="#exports">Export</a>
    <a href="#seconds">Second by second</a>
  </nav>
  <div class="side-foot" id="sidefoot"></div>
</aside>
<main>
<header class="top" id="editor">
  <div><h1 id="title">Hookline editing assistant</h1><div class="meta" id="meta"></div></div>
  <div class="right"><span class="chip pred" title="TTCC model prediction, not measured audience data">● Predicted retention</span><button class="primary" id="topexport">Export ↓</button></div>
</header>
<div class="banner" id="banner"></div>
<div class="grid">
<div class="col">
  <section class="panel">
    <div class="player"><video id="v" controls preload="metadata"></video></div>
    <p class="mut" id="vnote" style="margin:6px 2px 0"></p>
    <div class="status"><span id="now" class="mono">0:00</span><span id="activeinfo" class="mut"></span></div>
  </section>
  <section class="panel" id="hookpanel">
    <h2>Hook: is the opening predicted to hold?</h2>
    <p class="sub">Share of viewers TTCC predicts are still watching at each second (viewers who start at 0). A model prediction, not audience data.</p>
    <div class="tiles" id="tiles"></div>
    <p class="mut" id="hooknote" style="margin:8px 0 0"></p>
  </section>
  <section class="panel" id="retention">
    <h2>Predicted viewer retention <span class="legend"><span><i style="background:var(--g)"></i>Holds</span><span><i style="background:var(--y)"></i>Decline</span><span><i style="background:var(--r)"></i>Steep drop</span></span></h2>
    <p class="sub">Click the curve, the bar or a frame to jump. The white line follows the video.</p>
    <svg id="chart" viewBox="0 0 760 250" role="img" aria-label="Predicted retention curve"></svg>
    <div id="strip" title="Click to jump"></div>
    <div class="film" id="film"></div>
    <div class="row-actions"><button class="primary" id="playedited">▶ Play edited version (skips accepted trims)</button><span class="mut" id="editedinfo"></span></div>
  </section>
  <section class="panel" id="segments">
    <h2>Video segments &amp; recommendations</h2>
    <p class="sub">The predicted span split into consecutive segments. Select one to see its evidence and the proposed edits.</p>
    <div class="seglist" id="seglist"></div>
  </section>
  <section class="panel" id="scores">
    <h2>Scores per segment</h2>
    <p class="sub">Two different kinds of number, never combined: <b>predicted retention fall</b> (TTCC model prediction) and <b>heuristic risk / quality</b> (0–100, hand-set rules over measured features). Neither is audience data.</p>
    <div style="overflow-x:auto"><table id="segscores"><thead><tr><th>Segment</th><th>Type</th><th>Predicted R</th><th>Predicted fall (share)</th><th>Rank</th><th>Heuristic risk mean / max</th><th>Heuristic quality mean / max</th><th></th></tr></thead><tbody></tbody></table></div>
    <div class="row-actions"><button id="exportsegs">Export segment scores (CSV)</button></div>
  </section>
  <section class="panel" id="gemsec"><div id="geminibox"></div></section>
  <section class="panel" id="exports">
    <h2>Export</h2><p class="sub">Your decisions stay in this browser until you export them. Nothing is uploaded.</p>
    <div class="row-actions" style="margin-top:4px"><button class="primary" id="exportjson">Export edit decisions (JSON)</button><button class="primary" id="exportcsv">Export edit decisions (CSV)</button><button id="clear">Clear my decisions</button><span class="mut" id="saved"></span></div>
  </section>
  <section class="panel" id="seconds">
    <details><summary>Second by second (predicted R(t), drop, class, what was measured)</summary>
    <div style="overflow-x:auto;margin-top:8px"><table id="secs"><thead><tr><th>t</th><th>R(t) predicted</th><th>drop</th><th>class</th><th>measured in that second</th></tr></thead><tbody></tbody></table></div></details>
  </section>
  <p class="mut" id="foot"></p>
</div>
<div class="col">
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
const SERVED=location.protocol==='http:'||location.protocol==='https:';const SRC=SERVED?'video':D.video_url;  // served by scripts/serve_review.py, or the file:// path
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
function acceptedTrims(){return SUGS.filter(sg=>sg.c.edit.action==='trim').map(sg=>({sg,s:cur(sg)})).filter(x=>x.s.decision==='accepted'&&x.s.b>x.s.a).map(x=>({a:x.s.a,b:x.s.b})).sort((p,q)=>p.a-q.a);}
function exportRow(sg){const s=cur(sg);const act=sg.c.edit.action;return{video_id:D.video_id,interval_id:sg.iv.id,interval_kind:sg.iv.kind,interval_start_s:sg.iv.start_s,interval_end_s:sg.iv.end_s,predicted_R_start:+sg.iv.retention_start.toFixed(4),predicted_R_end:+sg.iv.retention_end.toFixed(4),source:sg.c.source,rule:sg.c.rule,suggestion:sg.c.suggestion,rationale:sg.c.rationale||'',confidence:sg.c.confidence==null?'':sg.c.confidence,evidence_refs:(sg.c.evidence_refs||[]).join(' '),action:act,edit_start_s:act==='add'?sg.c.edit.start_s:+(+s.a).toFixed(2),edit_end_s:act==='add'?'':+(+s.b).toFixed(2),decision:s.decision,note:s.note,data_kind:'predicted_retention_unvalidated'};}
function buildExport(){const items=SUGS.map(exportRow);const tr=acceptedTrims();const removed=tr.reduce((a,r)=>a+(r.b-r.a),0);
return{video_id:D.video_id,sha256:D.sha256,exported_at:new Date().toISOString(),disclaimer:DISCLAIMER,gemini_used:!!(D.gemini_info&&D.gemini_info.used),gemini_model:D.gemini_info&&D.gemini_info.used?D.gemini_info.model:null,retention_is_prediction:true,model:D.model,
original_duration_s:TOTAL,hook:D.hook,segment_scores:segmentRows(),segment_scores_note:'predicted fall = TTCC model prediction; heuristic risk/quality = hand-set rules over measured features; not combined, not audience data',accepted_trims:tr.map(r=>({start_s:+r.a.toFixed(2),end_s:+r.b.toFixed(2)})),duration_after_accepted_trims_s:+(TOTAL-removed).toFixed(2),edits:items};}
// ---- header
document.title='Hookline — '+D.video_id;$('title').textContent='Video analysis & editing — '+D.video_id;
$('meta').textContent=[D.duration_s?D.duration_s.toFixed(1)+' s':null,'predicted 0–'+D.horizon_s+' s',D.model?D.model.split('/').pop():null,D.stitched?'stitched windows (approximate)':null].filter(Boolean).join('  ·  ');
$('banner').append(el('b',{text:'⚠'}),el('span',{text:DISCLAIMER+' '+D.confidence_note+'.'+(D.partial?' PARTIAL: only the first '+D.horizon_s+' s were predicted.':'')+(D.deviations.length?' Differs from upstream evaluation: '+D.deviations.join('; ')+'.':'')}));
$('sidefoot').textContent='Model: '+(D.model||'TTCC')+'. Curves are predictions, not audience data. Nothing on this page is uploaded.';
$('topexport').addEventListener('click',()=>{$('exports').scrollIntoView({behavior:'smooth'});});
if(SRC){V.src=SRC;V.addEventListener('error',()=>{$('vnote').textContent='The browser could not load the video ('+(SERVED?'served over http':D.video_path)+'). If you opened this file directly, try: python3 scripts/serve_review.py '+D.video_id+' --artifacts <root> --video <file> --open. Timestamps still apply.';});}
else{V.style.display='none';$('vnote').textContent='Video file not found at '+D.video_path+'. Timestamps still apply, but playback and previews are unavailable.';}
const GI=D.gemini_info||{used:false};const gb0=$('geminibox');
if(GI.used){gb0.append(el('h2',{text:'Gemini overview (model interpretation)'}));if(GI.overall){gb0.append(el('p',{text:GI.overall.hook_assessment}));if(GI.overall.top_priorities.length)gb0.append(el('p',{class:'mut',text:'Gemini\'s priority order: '+GI.overall.top_priorities.join(' → ')}));}
gb0.append(el('p',{class:'mut',text:'Gemini ('+(GI.model||'model')+') explains the content and proposes edits; TTCC predicts retention. All of it is hypotheses until validated against real audience-retention data.'+(GI.validation&&GI.validation.dropped?' '+GI.validation.dropped+' item(s) of its answer failed validation and were dropped.':'')}));}
else{gb0.append(el('h2',{text:'Gemini'}),el('p',{class:'mut',text:'Gemini interpretation: not used'+(GI.reason?' ('+GI.reason+')':'')+'. Proposals are rule-based only.'}));}
// ---- control: jump / previews
let mode=null;function startPlay(){if(SRC)V.play().catch(()=>{});}
function jump(t,end){mode=end==null?null:{kind:'segment',until:end};V.currentTime=t;startPlay();V.scrollIntoView({behavior:'smooth',block:'nearest'});}
function previewTrim(a,b){mode={kind:'trim',a:a,b:b,until:Math.min(TOTAL,b+2)};V.currentTime=Math.max(0,a-2);startPlay();V.scrollIntoView({behavior:'smooth',block:'nearest'});}
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
// ---- hook tiles
const tiles=$('tiles');const H=D.hook||{};const secKind=(t)=>{const s=D.seconds.find(x=>x.t===t);return s?s.kind:null;};
const KC={steep:'#ff5468',decline:'#ffcf47',hold:'#5fd38a',gradual:'#b47cf0'};
for(const t of [1,3,5,10]){const r=H['R'+t];const tile=el('div',{class:'tile',title:r==null?'not predicted (video or analysed opening shorter)':'click to jump to '+t+' s'});
const bar=el('div',{class:'bar'},el('b',{}));bar.firstChild.style.width=r==null?'0%':Math.round(100*r)+'%';tile.style.setProperty('--c',KC[secKind(t)]||'#5fd38a');
tile.append(el('div',{class:'t',text:'R('+t+'s)  HOLD @ '+t+'s'}),el('div',{class:'v mono',text:pc(r)}),bar);tile.addEventListener('click',()=>{if(r!=null)jump(Math.max(0,t-1));});tiles.append(tile);}
const sd=H.steepest_early_drop;$('hooknote').textContent=sd?('Steepest predicted drop in the first '+H.early_window_s+' s: '+sd.from_s+'→'+sd.to_s+' s, '+Math.round(100*sd.drop)+' points.'):'';
// ---- timeline strip, chart, film, playhead
const strip=$('strip');const pct=(t)=>(100*Math.min(1,Math.max(0,t/TOTAL)))+'%';
const segEls=D.segments.map(sg=>{const e=el('div',{class:'seg '+sg.kind,title:sg.kind.replace('_',' ')+' '+sg.start_s+'-'+sg.end_s+' s: predicted fall '+sg.predicted_loss.toFixed(3)});e.style.left=pct(sg.start_s);e.style.width=(100*(sg.end_s-sg.start_s)/TOTAL)+'%';strip.append(e);return e;});
if(D.horizon_s<TOTAL-0.5){const n=el('div',{class:'none',title:'no prediction here'});n.style.left=pct(D.horizon_s);n.style.right='0';strip.append(n);}
const trimEls=[];const head=el('div',{class:'head'});strip.append(head);
strip.addEventListener('click',(ev)=>{const r=strip.getBoundingClientRect();jump(TOTAL*(ev.clientX-r.left)/r.width);});
let chartHead=null,chartX=null;
(function(){const svg=$('chart'),NS='http://www.w3.org/2000/svg',W=760,Hh=250,L=44,R=10,T=12,B=28,pw=W-L-R,ph=Hh-T-B,n=D.retention.length-1;
const x=(t)=>L+pw*t/n,y=(r)=>T+ph*(1-r);chartX=x;const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e;};
const defs=mk('defs',{});const gr=mk('linearGradient',{id:'gfill',x1:0,y1:0,x2:0,y2:1});gr.append(mk('stop',{offset:'0%','stop-color':'#5fd38a','stop-opacity':0.35}),mk('stop',{offset:'100%','stop-color':'#5fd38a','stop-opacity':0}));defs.append(gr);svg.append(defs);
for(let i=0;i<=4;i++){const r=i/4;svg.append(mk('line',{x1:L,x2:L+pw,y1:y(r),y2:y(r),stroke:'#232838','stroke-width':1}));const tx=mk('text',{x:L-8,y:y(r)+4,'text-anchor':'end','font-size':11,fill:'#8c92a6','font-family':'ui-monospace,Menlo,monospace'});tx.textContent=Math.round(r*100)+'%';svg.append(tx);}
const step=n<=10?1:n<=40?5:10;for(let t=0;t<=n;t+=step){svg.append(mk('line',{x1:x(t),x2:x(t),y1:T,y2:T+ph,stroke:'#1a1e2b','stroke-width':1}));const tx=mk('text',{x:x(t),y:Hh-8,'text-anchor':'middle','font-size':11,fill:'#8c92a6','font-family':'ui-monospace,Menlo,monospace'});tx.textContent=t+'s';svg.append(tx);}
for(const iv of D.intervals){svg.append(mk('rect',{x:x(iv.start_s),y:T,width:Math.max(2,x(iv.end_s)-x(iv.start_s)),height:ph,fill:COLOR[iv.kind]||'#5b6276','fill-opacity':'0.07'}));}
const pts=D.retention.map((r,t)=>x(t)+','+y(r));svg.append(mk('polygon',{points:x(0)+','+y(0)+' '+pts.join(' ')+' '+x(n)+','+y(0),fill:'url(#gfill)'}));
svg.append(mk('polyline',{points:pts.join(' '),fill:'none',stroke:'#5fd38a','stroke-width':2.2,'stroke-linejoin':'round'}));
D.retention.forEach((r,t)=>{const c=mk('circle',{cx:x(t),cy:y(r),r:6,fill:'transparent',style:'cursor:pointer'});const ti=mk('title',{});ti.textContent='t='+t+' s, predicted R='+r.toFixed(3)+' — click to jump';c.append(ti);c.addEventListener('click',()=>jump(t));svg.append(c);});
for(const iv of D.intervals){if(iv.kind==='holds')continue;const t=iv.start_s;const c=mk('circle',{cx:x(t),cy:y(D.retention[Math.min(t,n)]),r:5.5,fill:COLOR[iv.kind],stroke:'#0a0c12','stroke-width':2,style:'cursor:pointer'});const ti=mk('title',{});ti.textContent=iv.kind.replace('_',' ')+' '+iv.start_s+'–'+iv.end_s+' s';c.append(ti);c.addEventListener('click',()=>{select(iv.id);jump(iv.start_s);});svg.append(c);}
chartHead=mk('line',{x1:L,x2:L,y1:T,y2:T+ph,stroke:'#ffffff','stroke-width':1.5,'stroke-dasharray':'3 3'});svg.append(chartHead);})();
const film=$('film');const NF=Math.max(4,Math.min(8,Math.round(TOTAL/5)));const filmEls=[];
for(let i=0;i<NF;i++){const t=TOTAL*(i+0.5)/NF;const c=el('canvas',{width:320,height:200});const f=el('div',{class:'fr',title:'jump to '+fmt(t)},c,el('span',{text:mmss(t)}));f.addEventListener('click',()=>jump(t));film.append(f);filmEls.push({f,t});frameAt(c,t);}
// ---- segment list and the detail panel
const LABEL={steep_drop:'Major drop',sustained_decline:'Decline',holds:'Holds',unflagged:'Steady'};
function kindLabel(k){return {steep_drop:'Steep predicted drop',sustained_decline:'Sustained predicted decline',holds:'Predicted to hold',unflagged:'No predicted drop'}[k]||k;}
const ivOf=(sg)=>D.intervals.find(iv=>iv.start_s===sg.start_s&&iv.end_s===sg.end_s&&iv.kind===sg.kind);
function shortDesc(sg,iv){const bits=['R '+pc(sg.retention_start)+' → '+pc(sg.retention_end)];if(sg.share_of_total_predicted_loss!=null)bits.push(Math.round(100*sg.share_of_total_predicted_loss)+'% of the predicted fall');
const n=iv?iv.candidate_causes.length:0;if(n)bits.push(n+' proposal'+(n>1?'s':''));if(iv&&iv.gemini)bits.push('Gemini notes');return bits.join(' · ');}
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
if(iv.gemini){const gb=el('div',{class:'sec'},el('h4',{},el('i',{}),'Gemini: what it observed (model interpretation)'));if(iv.gemini.summary)gb.append(el('div',{text:iv.gemini.summary}));
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
if(c.source==='gemini'){const bits=['Gemini (model interpretation, hypothesis)'];if(c.confidence!=null)bits.push('confidence '+c.confidence.toFixed(2));box.append(el('div',{class:'mut',text:bits.join(' · ')}));
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
function refreshEdited(){const tr=acceptedTrims();trimEls.forEach(e=>e.remove());trimEls.length=0;for(const r of tr){const e=el('div',{class:'trim',title:'accepted trim '+r.a.toFixed(1)+'–'+r.b.toFixed(1)+' s'});e.style.left=pct(r.a);e.style.width=(100*(r.b-r.a)/TOTAL)+'%';strip.insertBefore(e,head);trimEls.push(e);}
const rem=tr.reduce((a,r)=>a+(r.b-r.a),0);$('editedinfo').textContent=tr.length?(tr.length+' accepted trim(s) remove '+rem.toFixed(1)+' s: '+TOTAL.toFixed(1)+' s → '+(TOTAL-rem).toFixed(1)+' s'):'no accepted trims yet';}
setInterval(()=>{tick();sync();},100);V.addEventListener('seeked',sync);
// ---- scores table
const KINDS={steep_drop:'Steep drop',sustained_decline:'Sustained decline',holds:'Holds',unflagged:'No flag'};
const ms=(a)=>a.mean==null?'n/a':a.mean.toFixed(1)+' / '+a.max.toFixed(1);
function segmentRows(){return D.segments.map(sg=>({video_id:D.video_id,segment_id:sg.id,kind:sg.kind,start_s:sg.start_s,end_s:sg.end_s,length_s:sg.length_s,predicted_R_start:+sg.retention_start.toFixed(4),predicted_R_end:+sg.retention_end.toFixed(4),predicted_fall:+sg.predicted_loss.toFixed(4),share_of_total_predicted_fall:sg.share_of_total_predicted_loss==null?'':+sg.share_of_total_predicted_loss.toFixed(4),rank_by_predicted_fall:sg.rank_by_predicted_loss,heuristic_risk_mean:sg.heuristic.attention_risk.mean==null?'':sg.heuristic.attention_risk.mean,heuristic_risk_max:sg.heuristic.attention_risk.max==null?'':sg.heuristic.attention_risk.max,heuristic_quality_mean:sg.heuristic.moment_quality.mean==null?'':sg.heuristic.moment_quality.mean,heuristic_quality_max:sg.heuristic.moment_quality.max==null?'':sg.heuristic.moment_quality.max,data_kind:'predicted_retention_unvalidated'}));}
function toCSVRows(rows){const cols=Object.keys(rows[0]||{});return [cols.join(',')].concat(rows.map(r=>cols.map(c=>csvCell(r[c])).join(','))).join('\r\n')+'\r\n';}
const stb=document.querySelector('#segscores tbody');
for(const sg of D.segments){const tr=el('tr',{class:'row'});const share=sg.share_of_total_predicted_loss;tr.style.setProperty('--c',COLOR[sg.kind]);
const bar=el('span',{class:'bar'});bar.style.width=Math.max(2,Math.round(60*(share||0)))+'px';
const play=el('button',{class:'small',text:'▶ clip'});play.addEventListener('click',(ev)=>{ev.stopPropagation();jump(sg.start_s,sg.end_s);});
tr.append(el('td',{class:'mono',text:sg.id+'  '+fmt(sg.start_s)+'–'+fmt(sg.end_s)+' ('+sg.length_s+' s)'}),el('td',{},el('span',{class:'pill',text:KINDS[sg.kind]||sg.kind})),el('td',{class:'mono',text:sg.retention_start.toFixed(3)+' → '+sg.retention_end.toFixed(3)}),
el('td',{class:'mono'},bar,document.createTextNode(sg.predicted_loss.toFixed(3)+(share==null?'':' ('+Math.round(100*share)+'%)'))),el('td',{class:'mono',text:'#'+sg.rank_by_predicted_loss}),el('td',{class:'mono',text:sg.heuristic.available?ms(sg.heuristic.attention_risk):'not run'}),el('td',{class:'mono',text:sg.heuristic.available?ms(sg.heuristic.moment_quality):'not run'}),el('td',{},play));
tr.addEventListener('click',()=>jump(sg.start_s));stb.append(tr);}
// ---- export
function download(name,type,text){const a=el('a',{href:URL.createObjectURL(new Blob([text],{type:type})),download:name});document.body.append(a);a.click();a.remove();}
$('exportjson').addEventListener('click',()=>download('hookline_edits_'+D.video_id+'.json','application/json',JSON.stringify(buildExport(),null,2)));
$('exportsegs').addEventListener('click',()=>download('hookline_segment_scores_'+D.video_id+'.csv','text/csv',toCSVRows(segmentRows())));
$('exportcsv').addEventListener('click',()=>download('hookline_edits_'+D.video_id+'.csv','text/csv',toCSV(buildExport().edits)));
$('clear').addEventListener('click',()=>{store={};try{localStorage.removeItem(KEY);}catch(e){}location.reload();});
// ---- seconds table
const tb=document.querySelector('#secs tbody');
for(const s of D.seconds){const tr=el('tr',{class:'row'});const o=s.observed;const w=o.words,c=o.scene_cuts,t=o.on_screen_text;
const meas=[w==='not run'?'speech ?':(w.count?'words: '+w.text:'no words'),t==='not run'?'':(t.length?'text: '+t.map(x=>x.text).join(','):''),c==='not run'?'':c.count+' cut(s)'].filter(Boolean).join(' · ');
tr.append(el('td',{class:'mono',text:s.t+' s'}),el('td',{class:'mono',text:s.retention_end.toFixed(3)}),el('td',{class:'mono',text:s.drop.toFixed(3)+' ('+(100*s.relative_loss).toFixed(0)+'%)'}),el('td',{},el('span',{class:'pill',text:s.kind})),el('td',{text:meas}));
tr.addEventListener('click',()=>jump(s.t-1));tb.append(tr);}
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
ok(document.querySelectorAll('#segscores tbody tr').length===D.segments.length,'one score row per segment');
ok(toCSVRows(segmentRows()).split('\r\n').length===D.segments.length+2,'segment-score CSV has header + rows');
ok(Math.abs(D.segments.reduce((a,sg)=>a+(sg.share_of_total_predicted_loss||0),0)-1)<0.01,'shares of the predicted fall add up to 1');
ok(document.querySelectorAll('#tiles .tile').length===4,'four hook tiles');
if(D.intervals.length>1){const last=D.intervals[D.intervals.length-1];select(last.id);ok(document.querySelector('.card.sel').dataset.id===last.id,'selecting a segment shows its detail');select(D.intervals[0].id);}
const first=SUGS.find(sg=>sg.c.edit.action==='trim');
if(first){store[first.id]={decision:'accepted',a:1,b:2.5,note:'x'};refreshEdited();const ex=buildExport();ok(ex.accepted_trims.length===1&&ex.accepted_trims[0].start_s===1&&ex.accepted_trims[0].end_s===2.5,'accepted trim reaches the export');
ok(Math.abs(ex.duration_after_accepted_trims_s-(TOTAL-1.5))<0.01,'duration after trims');ok(ex.disclaimer.includes('PREDICTED')&&ex.retention_is_prediction===true,'export is labelled as a prediction');
ok(ex.edits.every(e=>e.data_kind==='predicted_retention_unvalidated'),'every row is labelled');ok(toCSV(ex.edits).split('\r\n').length===ex.edits.length+2,'CSV has header + rows');store={};refreshEdited();}
else ok(true,'no trim proposals on this video (skipped trim checks)');
ok(SUGS.every(sg=>['trim','add','review'].includes(sg.c.edit.action)),'every proposal has a known action');
ok($('banner').textContent.includes('PREDICTED')&&$('banner').textContent.includes('not been verified'),'banner states prediction and unverified accuracy');
const pre=el('pre',{id:'selftest'});pre.textContent=(out.every(l=>l.startsWith('PASS'))?'SELFTEST PASS':'SELFTEST FAIL')+'\n'+out.join('\n');document.body.append(pre);}
</script></body></html>
"""
