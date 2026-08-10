"""Generate the six-curve learnings dashboard from DynoSim sweeps + AIC candidates.

Panels: 1 TTFT p95 vs concurrency (knee, log-y) · 2 prefix reuse vs concurrency ·
3 TPOT vs concurrency · 4 throughput vs concurrency by P:D (ordinal ramp) ·
5 efficiency frontier (tok/s/user x tok/s/GPU) · 6 theory overlay (sim + AIC, silicon
pending). Headline cell (3:3, conc 48) ringed. Dataviz reference palette, both themes.

Usage: gen_learnings_html.py <out.html>
"""
import csv
import json
from pathlib import Path

AR = Path.home() / "kv-cache-aware-bench/aic-results"
GPUS = 24
POL = {"rr": ("Round-robin", "s1"), "kv-nvda": ("KV-aware (defaults)", "s2"),
       "kv-t": ("KV-aware (tuned)", "s3"), "ll": ("Least-loaded", "ref")}


def fam(policy):
    return "kv-t" if policy.startswith("kv-t") else policy


def load_cells():
    cells = {}
    for f in ["dynosim_sweep_v4.csv", "dynosim_sweep_v5.csv"]:  # v5 wins overlaps
        for r in csv.DictReader(open(AR / f)):
            key = (r["pd"], int(r["conc"]), fam(r["policy"]))
            cur = cells.get(key)
            # within kv-t family keep best-throughput credit variant
            if cur is None or f.endswith("v5.csv") or \
               float(r["throughput_tok_s"]) > float(cur["throughput_tok_s"]):
                cells[key] = r
    return cells


def aic_points():
    pts = []
    p = AR / "silicon-warm-disagg-topn.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            pts.append({"x": float(r["tokens/s/user"]), "y": float(r["tokens/s/gpu"]),
                        "label": f"AIC conc {r['concurrency']} P{r['(p)workers']}:D{r['(d)workers']}"})
    return pts


def series(cells, pd, metric, scale=1.0):
    out = {}
    for (cpd, conc, f), r in sorted(cells.items(), key=lambda kv: kv[0][1]):
        if cpd != pd:
            continue
        out.setdefault(f, []).append(
            {"conc": conc, "v": float(r[metric]) * scale,
             "tokps": float(r["throughput_tok_s"]), "ttft95": float(r["ttft_p95_s"]),
             "ttft99": float(r["ttft_p99_s"]), "tpot": float(r["tpot_mean_ms"]),
             "hit": 100 * float(r["hit_rate"])})
    return out


def main():
    import sys
    out = sys.argv[1]
    cells = load_cells()
    data = {
        "ttft":  series(cells, "3:3", "ttft_p95_s"),
        "reuse": series(cells, "3:3", "hit_rate", 100.0),
        "tpot":  series(cells, "3:3", "tpot_mean_ms"),
        "pd": {pd: series(cells, pd, "throughput_tok_s")
               for pd in ("1:5", "2:4", "3:3")},
        "aic": aic_points(),
    }
    # panel 5/6 source: per-policy point clouds at 3:3
    cloud = {}
    for (pd, conc, f), r in cells.items():
        if pd != "3:3":
            continue
        cloud.setdefault(f, []).append({
            "x": 1000.0 / float(r["tpot_mean_ms"]), "y": float(r["throughput_tok_s"]) / GPUS,
            "conc": conc, "ttft95": float(r["ttft_p95_s"])})
    data["cloud"] = cloud
    page = TEMPLATE.replace("__DATA__", json.dumps(data))
    Path(out).write_text(page)
    print(f"wrote {out}")


TEMPLATE = r"""<title>KV-aware routing — six-curve learnings (Kimi-K2.5, GB300 24 GPU)</title>
<style>
:root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ref:#898781;--o1:#86b6ef;--o2:#2a78d6;--o3:#104281}
body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
--surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--o1:#6da7ec;--o2:#3987e5;--o3:#9ec5f4}}
:root[data-theme="dark"]{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--o1:#6da7ec;--o2:#3987e5;--o3:#9ec5f4}
.wrap{max-width:1060px;margin:0 auto;padding:24px}
h1{font-size:19px;margin:0 0 2px} .sub{color:var(--ink-2);font-size:13px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}
.panel{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
.panel h2{font-size:13.5px;margin:0 0 2px} .panel .d{font-size:11.5px;color:var(--muted);margin-bottom:6px}
svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--ink-2);margin-top:6px}
.legend span{display:inline-flex;align-items:center;gap:5px}
.sw{width:9px;height:9px;border-radius:2px;display:inline-block}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);
border-radius:6px;padding:7px 9px;font-size:12px;color:var(--ink);box-shadow:0 2px 10px rgba(0,0,0,.15);
opacity:0;max-width:250px;line-height:1.45;font-variant-numeric:tabular-nums}
.note{font-size:11.5px;color:var(--muted);margin-top:12px}
</style>
<div class="wrap">
<h1>KV-cache-aware routing — the six curves</h1>
<div class="sub">Kimi-K2.5-NVFP4 · GB300 · 24 GPUs disaggregated · session-interleaved Weka 256k trace ·
DynoSim v4+v5 (silicon-calibrated) · headline cell 3:3 / conc 48 is ringed · hover any mark for detail</div>
<div class="grid" id="grid"></div>
<div class="note">Simulation, calibrated against AIC SILICON solves and live smoke/canary telemetry; live round-1
measurements (running) overlay next. Policies use fixed colors across panels; panel 4's blues are an ordinal
ramp over prefill-worker count. TTFT reference line = NVIDIA recipe goodput threshold (5 s).</div>
</div>
<div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const PW=460,PH=300,M={t:12,r:14,b:38,l:52};
const FAM={"rr":["Round-robin","s1"],"kv-nvda":["KV defaults","s2"],"kv-t":["KV tuned","s3"],"ll":["Least-loaded","ref"]};
const ORDER=["rr","ll","kv-nvda","kv-t"];
let tipsN=0;
function panel(id,title,desc,build,legendItems){
 const el=document.createElement('div');el.className='panel';
 el.innerHTML=`<h2>${title}</h2><div class="d">${desc}</div><div id="${id}"></div>
  <div class="legend">${legendItems.map(([n,c])=>`<span><span class="sw" style="background:var(--${c})"></span>${n}</span>`).join('')}</div>`;
 document.getElementById('grid').appendChild(el);
 document.getElementById(id).innerHTML=`<svg viewBox="0 0 ${PW} ${PH}">${build()}</svg>`;
}
function axes(X,Y,xt,yt,xl,yl,fmt){
 let g='';
 for(const v of xt){g+=`<line x1="${X(v)}" y1="${M.t}" x2="${X(v)}" y2="${PH-M.b}" stroke="var(--grid)"/>`;
  g+=`<text x="${X(v)}" y="${PH-M.b+15}" text-anchor="middle" font-size="10" fill="var(--muted)">${v}</text>`;}
 for(const v of yt){g+=`<line x1="${M.l}" y1="${Y(v)}" x2="${PW-M.r}" y2="${Y(v)}" stroke="var(--grid)"/>`;
  g+=`<text x="${M.l-6}" y="${Y(v)+3.5}" text-anchor="end" font-size="10" fill="var(--muted)">${fmt?fmt(v):v}</text>`;}
 g+=`<text x="${(M.l+PW-M.r)/2}" y="${PH-6}" text-anchor="middle" font-size="11" fill="var(--ink-2)">${xl}</text>`;
 g+=`<text x="12" y="${(M.t+PH-M.b)/2}" text-anchor="middle" font-size="11" fill="var(--ink-2)" transform="rotate(-90 12 ${(M.t+PH-M.b)/2})">${yl}</text>`;
 return g;
}
function mark(x,y,c,tip,ring){
 const id='m'+(tipsN++);
 return `<g class="pt" data-tip='${tip.replace(/'/g,"&#39;")}'>${ring?`<circle cx="${x}" cy="${y}" r="8" fill="none" stroke="var(--ink-2)" stroke-width="1.4" stroke-dasharray="2 2"/>`:''}<circle cx="${x}" cy="${y}" r="9" fill="transparent"/><circle cx="${x}" cy="${y}" r="3.4" fill="var(--${c})"/></g>`;
}
function lineChart(id,title,desc,ser,vfmt,ylabel,opts={}){
 panel(id,title,desc,()=> {
  const concs=[...new Set(Object.values(ser).flat().map(p=>p.conc))].sort((a,b)=>a-b);
  const xmin=Math.log(concs[0]),xmax=Math.log(concs[concs.length-1]);
  const X=c=>M.l+((Math.log(c)-xmin)/(xmax-xmin))*(PW-M.l-M.r);
  const vals=Object.values(ser).flat().map(p=>p.v);
  const logy=opts.logy, ymin=logy?Math.log10(Math.max(.01,Math.min(...vals))):0, ymax=logy?Math.log10(Math.max(...vals)*1.3):Math.max(...vals)*1.12;
  const Y=v=>{const t=logy?Math.log10(Math.max(.01,v)):v;return PH-M.b-((t-ymin)/(ymax-ymin))*(PH-M.t-M.b);};
  const yt=logy?[0.1,1,10,100].filter(v=>Math.log10(v)>=ymin&&Math.log10(v)<=ymax):[0,.25,.5,.75,1].map(f=>Math.round(ymax*f*10)/10);
  let g=axes(X,Y,concs,yt,'concurrency (log)',ylabel,vfmt);
  if(opts.ref5&&Y(5)>M.t){g+=`<line x1="${M.l}" y1="${Y(5)}" x2="${PW-M.r}" y2="${Y(5)}" stroke="var(--axis)" stroke-dasharray="4 3"/><text x="${PW-M.r-2}" y="${Y(5)-4}" text-anchor="end" font-size="9.5" fill="var(--muted)">recipe 5 s</text>`;}
  for(const f of ORDER){const pts=ser[f];if(!pts)continue;const c=FAM[f][1];
   g+=`<path d="${pts.map((p,i)=>`${i?'L':'M'}${X(p.conc)} ${Y(p.v)}`).join('')}" fill="none" stroke="var(--${c})" stroke-width="2"/>`;
   for(const p of pts) g+=mark(X(p.conc),Y(p.v),c,
     JSON.stringify({s:FAM[f][0],conc:p.conc,v:vfmt?vfmt(p.v):p.v,tokps:p.tokps,ttft95:p.ttft95,ttft99:p.ttft99,tpot:p.tpot,hit:p.hit}),
     p.conc===48&&(f==='rr'||f==='kv-t'));
  }
  return g;
 },ORDER.filter(f=>ser[f]).map(f=>[FAM[f][0],FAM[f][1]]));
}
lineChart('p1','1 · TTFT p95 vs concurrency — the capacity knee',
 'KV-aware stays flat while cache-blind policies queue-diverge; 3:3 topology',D.ttft,v=>v>=10?v.toFixed(0)+'s':v.toFixed(1)+'s','TTFT p95 (s, log)',{logy:true,ref5:true});
lineChart('p2','2 · Prefix-cache reuse vs concurrency',
 'Emergent hit rates from real trace hash_ids; the mechanism behind every other panel',D.reuse,v=>v.toFixed(0)+'%','reuse rate (%)',{});
lineChart('p3','3 · TPOT vs concurrency',
 'Routing costs decode nothing — all policies overlap until the throughput ceiling',D.tpot,v=>v.toFixed(0)+'ms','TPOT (ms)',{});
panel('p4','4 · Throughput vs concurrency by P:D split',
 'Ordinal blues = prefill workers 1→3 (KV-aware best); dashed gray = round-robin at 3:3',()=> {
  const all=Object.values(D.pd).flatMap(s=>Object.values(s).flat());
  const concs=[...new Set(all.map(p=>p.conc))].sort((a,b)=>a-b);
  const xmin=Math.log(concs[0]),xmax=Math.log(concs[concs.length-1]);
  const X=c=>M.l+((Math.log(c)-xmin)/(xmax-xmin))*(PW-M.l-M.r);
  const ymax=Math.max(...all.map(p=>p.v))*1.1;
  const Y=v=>PH-M.b-(v/ymax)*(PH-M.t-M.b);
  let g=axes(X,Y,concs,[0,1000,2000,3000],'concurrency (log)','output tok/s (system)');
  const ramp={'1:5':'o1','2:4':'o2','3:3':'o3'};
  for(const pd of ['1:5','2:4','3:3']){
   const pts=(D.pd[pd]['kv-t']||D.pd[pd]['kv-nvda']||[]);
   g+=`<path d="${pts.map((p,i)=>`${i?'L':'M'}${X(p.conc)} ${Y(p.v)}`).join('')}" fill="none" stroke="var(--${ramp[pd]})" stroke-width="2"/>`;
   for(const p of pts) g+=mark(X(p.conc),Y(p.v),ramp[pd],JSON.stringify({s:'KV '+pd,conc:p.conc,v:p.v.toFixed(0)+' tok/s',ttft95:p.ttft95}),pd==='3:3'&&p.conc===48);
  }
  const rr=D.pd['3:3']['rr']||[];
  g+=`<path d="${rr.map((p,i)=>`${i?'L':'M'}${X(p.conc)} ${Y(p.v)}`).join('')}" fill="none" stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 4"/>`;
  for(const p of rr) g+=mark(X(p.conc),Y(p.v),'ref',JSON.stringify({s:'RR 3:3',conc:p.conc,v:p.v.toFixed(0)+' tok/s',ttft95:p.ttft95}),p.conc===48);
  return g;
 },[['KV 1 prefill','o1'],['KV 2 prefill','o2'],['KV 3 prefill','o3'],['RR 3:3 (dashed)','ref']]);
function frontier(pts){const s=[...pts].sort((a,b)=>b.x-a.x||b.y-a.y);const f=[];let by=-1;
 for(const p of s){if(p.y>by){f.push(p);by=p.y;}}return f.sort((a,b)=>a.x-b.x);}
panel('p5','5 · Efficiency frontier — tok/s/GPU vs tok/s/user',
 'All 3:3 cells; frontier lines per policy; hollow = TTFT p95 above 5 s',()=> {
  const all=Object.values(D.cloud).flat();
  const xmax=Math.max(...all.map(p=>p.x))*1.08,ymax=Math.max(...all.map(p=>p.y))*1.1;
  const X=v=>M.l+(v/xmax)*(PW-M.l-M.r),Y=v=>PH-M.b-(v/ymax)*(PH-M.t-M.b);
  let g=axes(X,Y,[0,20,40,60],[0,40,80,120],'tok/s per user','tok/s per GPU',v=>v);
  for(const f of ORDER){const pts=D.cloud[f];if(!pts)continue;const c=FAM[f][1];
   const fr=frontier(pts);
   g+=`<path d="${fr.map((p,i)=>`${i?'L':'M'}${X(p.x)} ${Y(p.y)}`).join('')}" fill="none" stroke="var(--${c})" stroke-width="2" opacity=".85"/>`;
   for(const p of pts){const hollow=p.ttft95>5;
    g+=`<g class="pt" data-tip='${JSON.stringify({s:FAM[f][0],conc:p.conc,v:p.y.toFixed(1)+" tok/s/GPU",ttft95:p.ttft95})}'><circle cx="${X(p.x)}" cy="${Y(p.y)}" r="9" fill="transparent"/><circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3.6" fill="${hollow?'none':`var(--${c})`}" stroke="var(--${c})" stroke-width="1.8"/></g>`;}
  }
  return g;
 },ORDER.map(f=>[FAM[f][0],FAM[f][1]]));
panel('p6','6 · Theory overlay — DynoSim vs AIC (silicon pending)',
 'KV frontier from DynoSim vs AIC SILICON warm candidates; live round-1 points land here next',()=> {
  const sim=(D.cloud['kv-t']||[]).concat(D.cloud['kv-nvda']||[]);
  const pts=sim.concat(D.aic.map(p=>({...p,aic:true})));
  const xmax=Math.max(...pts.map(p=>p.x))*1.08,ymax=Math.max(...pts.map(p=>p.y))*1.15;
  const X=v=>M.l+(v/xmax)*(PW-M.l-M.r),Y=v=>PH-M.b-(v/ymax)*(PH-M.t-M.b);
  let g=axes(X,Y,[0,20,40,60,80],[0,40,80,120],'tok/s per user','tok/s per GPU',v=>v);
  const fr=frontier(sim);
  g+=`<path d="${fr.map((p,i)=>`${i?'L':'M'}${X(p.x)} ${Y(p.y)}`).join('')}" fill="none" stroke="var(--s3)" stroke-width="2"/>`;
  for(const p of sim) g+=mark(X(p.x),Y(p.y),'s3',JSON.stringify({s:'DynoSim KV',conc:p.conc,v:p.y.toFixed(1)+' tok/s/GPU',ttft95:p.ttft95}),false);
  for(const p of D.aic) g+=`<g class="pt" data-tip='${JSON.stringify({s:p.label,v:p.y.toFixed(1)+" tok/s/GPU at "+p.x.toFixed(1)+" tok/s/user"})}'><circle cx="${X(p.x)}" cy="${Y(p.y)}" r="9" fill="transparent"/><rect x="${X(p.x)-3.4}" y="${Y(p.y)-3.4}" width="6.8" height="6.8" rx="1.4" fill="var(--s1)"/></g>`;
  g+=`<text x="${PW-M.r-4}" y="${M.t+12}" text-anchor="end" font-size="10.5" fill="var(--muted)">silicon points: round-1 chain in flight</text>`;
  return g;
 },[['DynoSim KV frontier','s3'],['AIC SILICON candidates','s1']]);
const tip=document.getElementById('tip');
document.querySelectorAll('.pt').forEach(el=>{
 el.addEventListener('mousemove',e=>{const p=JSON.parse(el.dataset.tip);
  tip.innerHTML=`<b>${p.s}</b>${p.conc?` · conc ${p.conc}`:''}<br>${p.v}`+
   (p.ttft95!==undefined?`<br><span style="color:var(--ink-2)">TTFT p95 ${(+p.ttft95).toFixed(2)} s${p.ttft99?` · p99 ${(+p.ttft99).toFixed(2)} s`:''}</span>`:'')+
   (p.hit!==undefined?`<br><span style="color:var(--ink-2)">reuse ${(+p.hit).toFixed(1)}% · ${(+p.tokps).toFixed(0)} tok/s</span>`:'');
  tip.style.left=Math.min(e.clientX+13,innerWidth-260)+'px';tip.style.top=(e.clientY+13)+'px';tip.style.opacity=1;});
 el.addEventListener('mouseleave',()=>tip.style.opacity=0);
});
</script>
"""

if __name__ == "__main__":
    main()
