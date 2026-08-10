"""Generate a self-contained interactive Pareto page from a DynoSim sweep CSV.

Axes per the pareto-benchmarking guide: x = tok/s/user (interactivity),
y = tok/s/GPU (efficiency). One frontier trace per policy family; SLA-failing
cells drawn hollow. Hover tooltips carry full cell metadata; a table view is
included for accessibility. Palette: dataviz reference instance (3 categorical
slots + muted reference series), light/dark theme aware.

Usage: gen_pareto_html.py <sweep.csv> <out.html> [--gpus 24]
"""
import argparse
import csv
import html as html_mod
import json

FAMILIES = {  # display name -> (csv-policy matcher, slot)
    "Round-robin": (lambda p: p == "rr", "s1"),
    "KV-aware (NVDA defaults)": (lambda p: p == "kv-nvda", "s2"),
    "KV-aware (Strategy C tuned)": (lambda p: p.startswith("kv-t"), "s3"),
    "Least-loaded": (lambda p: p == "ll", "ref"),
}
MARKERS = {"s1": "circle", "s2": "square", "s3": "diamond", "ref": "triangle"}


def pareto_front(pts):
    """Upper-right frontier on (x=tok/s/user, y=tok/s/gpu)."""
    pts = sorted(pts, key=lambda p: (-p["x"], -p["y"]))
    front, best_y = [], -1
    for p in pts:
        if p["y"] > best_y:
            front.append(p)
            best_y = p["y"]
    return sorted(front, key=lambda p: p["x"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in")
    ap.add_argument("out")
    ap.add_argument("--gpus", type=int, default=24)
    args = ap.parse_args()

    series = {name: [] for name in FAMILIES}
    with open(args.csv_in) as f:
        for r in csv.DictReader(f):
            tokps = float(r["throughput_tok_s"])
            tpot = float(r["tpot_mean_ms"])
            pt = {
                "x": 1000.0 / tpot if tpot > 0 else 0.0,
                "y": tokps / args.gpus,
                "pd": r["pd"], "conc": int(r["conc"]), "policy": r["policy"],
                "ttft95": float(r["ttft_p95_s"]), "tpot": tpot,
                "hit": float(r["hit_rate"]) * 100, "tokps": tokps,
                "sla": r["sla_pass"] in ("True", "true", "1"),
            }
            for name, (match, _) in FAMILIES.items():
                if match(r["policy"]):
                    series[name].append(pt)

    payload = {
        name: {"slot": FAMILIES[name][1], "marker": MARKERS[FAMILIES[name][1]],
               "points": pts, "front": pareto_front([p for p in pts if p["sla"]])}
        for name, pts in series.items() if pts
    }

    rows_html = "\n".join(
        f"<tr><td>{html_mod.escape(n)}</td><td>{p['pd']}</td><td>{p['conc']}</td>"
        f"<td>{p['x']:.1f}</td><td>{p['y']:.1f}</td><td>{p['ttft95']:.2f}</td>"
        f"<td>{p['tpot']:.1f}</td><td>{p['hit']:.1f}</td>"
        f"<td>{'pass' if p['sla'] else 'FAIL'}</td></tr>"
        for n, s in payload.items() for p in sorted(s["points"], key=lambda q: (q["pd"], q["conc"]))
    )

    # highlight: best SLA-passing KV cell by throughput x capped ratio vs same-cell RR
    rr_by_cell = {(p["pd"], p["conc"]): p for p in series["Round-robin"]}
    best, best_score = None, -1
    for name in ("KV-aware (Strategy C tuned)", "KV-aware (NVDA defaults)"):
        for p in series.get(name, []):
            rr = rr_by_cell.get((p["pd"], p["conc"]))
            if not (p["sla"] and rr):
                continue
            ratio = p["tokps"] / max(1e-9, rr["tokps"])
            score = p["tokps"] * min(ratio, 10.0)
            if score > best_score:
                best_score, best = score, (p, rr, ratio)
    hl, rr, ratio = best
    page = (PAGE_TEMPLATE.replace("__DATA__", json.dumps(payload))
            .replace("__TABLE_ROWS__", rows_html).replace("__GPUS__", str(args.gpus))
            .replace("__KV_TOKS__", f"{hl['tokps']:,.0f}").replace("__HL_PD__", hl["pd"])
            .replace("__HL_CONC__", str(hl["conc"])).replace("__KV_TTFT__", f"{hl['ttft95']:.2f}")
            .replace("__RR_TOKS__", f"{rr['tokps']:,.0f}").replace("__RR_TTFT__", f"{rr['ttft95']:.1f}")
            .replace("__RATIO__", f"{ratio:.2f}").replace("__TTFT_X__", f"{rr['ttft95']/max(.01,hl['ttft95']):.0f}"))
    with open(args.out, "w") as f:
        f.write(page)
    n = sum(len(s["points"]) for s in payload.values())
    print(f"wrote {args.out}: {n} cells, {len(payload)} series")


PAGE_TEMPLATE = """<title>DynoSim Pareto — KV-aware routing impact (GB300, __GPUS__ GPUs)</title>
<style>
:root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ref:#898781}
body{margin:0;background:var(--page);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{padding:24px;min-height:100vh;box-sizing:border-box}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
--surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
:root[data-theme="dark"]{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;
--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:20px;max-width:980px;margin:0 auto}
h1{font-size:18px;margin:0 0 2px}
.sub{color:var(--ink-2);font-size:13px;margin-bottom:14px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink-2);margin:10px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
svg{width:100%;height:auto;display:block}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);
border-radius:6px;padding:8px 10px;font-size:12px;color:var(--ink);box-shadow:0 2px 10px rgba(0,0,0,.15);
opacity:0;transition:opacity .08s;max-width:260px;line-height:1.5}
.tip b{color:var(--ink)} .tip .m{color:var(--ink-2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:14px 0}
.tile{border:1px solid var(--border);border-radius:6px;padding:10px 12px}
.tl{font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.tv{font-size:22px;font-weight:650;margin:2px 0;font-variant-numeric:tabular-nums}
.ts{font-size:12px;color:var(--ink-2);font-variant-numeric:tabular-nums}
details{margin-top:14px;font-size:12.5px;color:var(--ink-2)}
table{border-collapse:collapse;font-size:12px;margin-top:8px;font-variant-numeric:tabular-nums}
th,td{padding:3px 10px;border-bottom:1px solid var(--grid);text-align:right;color:var(--ink-2)}
th{color:var(--muted);font-weight:600} td:first-child,th:first-child{text-align:left}
.note{font-size:12px;color:var(--muted);margin-top:10px}
</style>
<div class="viz-root">
<div class="card">
<h1>KV-aware routing impact — DynoSim Pareto sweep</h1>
<div class="sub">GB300 &middot; __GPUS__ GPUs disaggregated &middot; Weka 256k agentic trace (closed-loop replay)
&middot; frontier lines connect SLA-passing points (TTFT p95 &le; 5 s, TPOT &le; 20 ms); hollow marks fail the SLA</div>
<div class="tiles">
<div class="tile"><div class="tl">Highlight point &middot; KV-tuned</div><div class="tv">__KV_TOKS__ tok/s</div><div class="ts">P:D __HL_PD__ &middot; conc __HL_CONC__ &middot; TTFT p95 __KV_TTFT__ s</div></div>
<div class="tile"><div class="tl">Round-robin, same deployment</div><div class="tv">__RR_TOKS__ tok/s</div><div class="ts">TTFT p95 __RR_TTFT__ s &mdash; SLA fail</div></div>
<div class="tile"><div class="tl">KV-aware advantage</div><div class="tv">__RATIO__&times; throughput</div><div class="ts">at __TTFT_X__&times; lower TTFT p95</div></div>
</div>
<div id="chart"></div>
<div class="legend" id="legend"></div>
<div class="note">x — interactivity (tokens/s per user, from mean TPOT); y — efficiency (output tokens/s per GPU).
Hover any mark for the full cell: P:D split, concurrency, policy flags, TTFT p95, cache hit rate.</div>
<details><summary>Table view (all cells)</summary>
<div style="overflow-x:auto"><table><thead><tr><th>series</th><th>P:D</th><th>conc</th><th>tok/s/user</th>
<th>tok/s/GPU</th><th>TTFT p95 (s)</th><th>TPOT (ms)</th><th>hit %</th><th>SLA</th></tr></thead>
<tbody>__TABLE_ROWS__</tbody></table></div></details>
</div>
<div class="tip" id="tip"></div>
</div>
<script>
const DATA = __DATA__;
const W=940,H=520,M={t:16,r:16,b:46,l:56};
const pts=Object.values(DATA).flatMap(s=>s.points);
const xmax=Math.max(...pts.map(p=>p.x))*1.06, ymax=Math.max(...pts.map(p=>p.y))*1.08;
const X=v=>M.l+(v/xmax)*(W-M.l-M.r), Y=v=>H-M.b-(v/ymax)*(H-M.t-M.b);
const css=v=>getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
function marker(shape,x,y,fill,hollow){
 const f=hollow?'none':fill, st=`stroke="${fill}" stroke-width="2"`;
 if(shape==='circle')  return `<circle cx="${x}" cy="${y}" r="4.5" fill="${f}" ${st}/>`;
 if(shape==='square')  return `<rect x="${x-4}" y="${y-4}" width="8" height="8" rx="1.5" fill="${f}" ${st}/>`;
 if(shape==='diamond') return `<path d="M${x} ${y-5.5}L${x+5.5} ${y}L${x} ${y+5.5}L${x-5.5} ${y}Z" fill="${f}" ${st}/>`;
 return `<path d="M${x} ${y-5}L${x+5} ${y+4.5}L${x-5} ${y+4.5}Z" fill="${f}" ${st}/>`;
}
function render(){
 let g='';
 const nx=6, ny=5;
 for(let i=0;i<=nx;i++){const v=xmax*i/nx;
  g+=`<line x1="${X(v)}" y1="${M.t}" x2="${X(v)}" y2="${H-M.b}" stroke="var(--grid)" stroke-width="1"/>`;
  g+=`<text x="${X(v)}" y="${H-M.b+18}" text-anchor="middle" font-size="11" fill="var(--muted)">${v.toFixed(0)}</text>`;}
 for(let i=0;i<=ny;i++){const v=ymax*i/ny;
  g+=`<line x1="${M.l}" y1="${Y(v)}" x2="${W-M.r}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/>`;
  g+=`<text x="${M.l-8}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)">${v.toFixed(0)}</text>`;}
 g+=`<line x1="${M.l}" y1="${H-M.b}" x2="${W-M.r}" y2="${H-M.b}" stroke="var(--axis)" stroke-width="1"/>`;
 g+=`<text x="${(M.l+W-M.r)/2}" y="${H-8}" text-anchor="middle" font-size="12" fill="var(--ink-2)">tokens/s per user (interactivity) &rarr;</text>`;
 g+=`<text x="14" y="${(M.t+H-M.b)/2}" text-anchor="middle" font-size="12" fill="var(--ink-2)" transform="rotate(-90 14 ${(M.t+H-M.b)/2})">output tokens/s per GPU &rarr;</text>`;
 let marks='';
 for(const [name,s] of Object.entries(DATA)){
  const col=`var(--${s.slot==='ref'?'ref':s.slot})`;
  if(s.front.length>1){
   const d=s.front.map((p,i)=>`${i?'L':'M'}${X(p.x)} ${Y(p.y)}`).join('');
   g+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="2" opacity="0.85"/>`;
   const last=s.front[s.front.length-1];
   g+=`<text x="${X(last.x)+8}" y="${Y(last.y)-6}" font-size="11.5" fill="var(--ink-2)">${name.split(' (')[0]}</text>`;
  }
  for(const p of s.points){
   marks+=`<g class="pt" data-tip='${JSON.stringify({name,...p}).replace(/'/g,"&#39;")}'>
     <circle cx="${X(p.x)}" cy="${Y(p.y)}" r="11" fill="transparent"/>
     ${marker(s.marker,X(p.x),Y(p.y),col,!p.sla)}</g>`;
  }
 }
 document.getElementById('chart').innerHTML=
  `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Pareto scatter of routing policies">${g}${marks}</svg>`;
 document.getElementById('legend').innerHTML=Object.entries(DATA).map(([name,s])=>
  `<span><span class="sw" style="background:var(--${s.slot==='ref'?'ref':s.slot})"></span>${name}</span>`).join('')+
  `<span><span class="sw" style="background:none;border:2px solid var(--muted);box-sizing:border-box"></span>SLA fail (hollow)</span>`;
 const tip=document.getElementById('tip');
 document.querySelectorAll('.pt').forEach(el=>{
  el.addEventListener('mousemove',e=>{
   const p=JSON.parse(el.dataset.tip);
   tip.innerHTML=`<b>${p.name}</b><br><span class="m">P:D ${p.pd} &middot; conc ${p.conc} &middot; ${p.policy}</span><br>
    ${p.tokps.toFixed(0)} tok/s &middot; ${p.y.toFixed(1)} tok/s/GPU<br>
    ${p.x.toFixed(1)} tok/s/user (TPOT ${p.tpot.toFixed(1)} ms)<br>
    TTFT p95 ${p.ttft95.toFixed(2)} s &middot; hit ${p.hit.toFixed(1)}%<br>
    <span class="m">${p.sla?'SLA pass':'SLA FAIL'}</span>`;
   tip.style.left=Math.min(e.clientX+14,innerWidth-280)+'px'; tip.style.top=(e.clientY+14)+'px'; tip.style.opacity=1;});
  el.addEventListener('mouseleave',()=>tip.style.opacity=0);
 });
}
render();
</script>
"""

if __name__ == "__main__":
    main()
