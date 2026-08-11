"""Standalone sglang simulation pareto/curves page (separate from the combined
learnings dashboard). Data: DynoSim sgl-sim v1 sweeps in sglang/results/ plus the
trtllm 3:3 cells for the cross-backend panel.

Usage: gen_sglang_curves_html.py <out.html>
"""
import csv
import json
import sys
from pathlib import Path

AR = Path.home() / "kv-cache-aware-bench/aic-results"
SR = Path.home() / "kv-cache-aware-bench/sglang/results"


def fam(policy):
    return "kv-t" if policy.startswith("kv-t") else policy


def load_one(path):
    cells = {}
    if not Path(path).exists():
        return cells
    for r in csv.DictReader(open(path)):
        key = (r["pd"], int(r["conc"]), fam(r["policy"]))
        cur = cells.get(key)
        if cur is None or float(r["throughput_tok_s"]) > float(cur["throughput_tok_s"]):
            cells[key] = r
    return cells


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


def cloud_of(cells, pd, gpus):
    out = {}
    for (cpd, conc, f), r in cells.items():
        if cpd != pd:
            continue
        osl = 1131.0
        tpot = float(r["tpot_mean_ms"]); t50 = float(r["ttft_p50_s"])
        out.setdefault(f, []).append({
            "x": 1000.0 / tpot, "y": float(r["throughput_tok_s"]) / gpus,
            "xe": osl / (t50 + osl * tpot / 1000.0),
            "conc": conc, "ttft95": float(r["ttft_p95_s"])})
    return out


def main():
    out = sys.argv[1]
    sgl24 = load_one(SR / "dynosim_sgl_disagg24_v1.csv")
    sgl72 = load_one(SR / "dynosim_sgl_disagg72_v1.csv")
    sglagg = load_one(SR / "dynosim_sgl_agg_v2.csv")
    # trtllm 3:3 cells for the cross-backend panel
    trt = {}
    for f in ["dynosim_sweep_v4.csv", "dynosim_sweep_v5.csv"]:
        for r in csv.DictReader(open(AR / f)):
            key = (r["pd"], int(r["conc"]), fam(r["policy"]))
            cur = trt.get(key)
            if cur is None or f.endswith("v5.csv") or \
               float(r["throughput_tok_s"]) > float(cur["throughput_tok_s"]):
                trt[key] = r

    data = {
        "d24ttft": series(sgl24, "3:3", "ttft_p95_s"),
        "d24reuse": series(sgl24, "3:3", "hit_rate", 100.0),
        "d24tput": series(sgl24, "3:3", "throughput_tok_s"),
        "d24cloud": cloud_of(sgl24, "3:3", 24),
        "aggttft": series(sglagg, "agg6", "ttft_p95_s"),
        "aggtput": series(sglagg, "agg6", "throughput_tok_s"),
        "aggcloud": cloud_of(sglagg, "agg6", 24),
        "d72ttft": series(sgl72, "6:12", "ttft_p95_s"),
        "d72pd": {pd: series(sgl72, pd, "throughput_tok_s")
                  for pd in ("3:15", "6:12", "9:9", "12:6", "15:3")},
        "d72cloud": cloud_of(sgl72, "6:12", 72),
        "trtcloud": cloud_of(trt, "3:3", 24),
    }
    page = TEMPLATE.replace("__DATA__", json.dumps(data))
    Path(out).write_text(page)
    print(f"wrote {out}")


TEMPLATE = r"""<title>SGLang — DynoSim pareto curves (Kimi-K2.5, GB300)</title>
<style>
:root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ref:#898781;--s4:#8250c4;
--o0:#cfe0f5;--o1:#86b6ef;--o2:#2a78d6;--o3:#104281;--o4:#0a1f40}
body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
--surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9d7ad6;
--o0:#31465f;--o1:#6da7ec;--o2:#3987e5;--o3:#9ec5f4;--o4:#c8dcf6}}
:root[data-theme="dark"]{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--s4:#9d7ad6;--o0:#31465f;--o1:#6da7ec;--o2:#3987e5;--o3:#9ec5f4;--o4:#c8dcf6}
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
<h1>SGLang backend — DynoSim pareto curves</h1>
<div class="sub">Kimi-K2.5-NVFP4 · GB300 · sgl-sim v1 seed (AIC SILICON ratio transfer onto live-calibrated
trtllm constants) · session-interleaved Weka 256k trace · selected cells ringed (agg conc 16 · 72-GPU 6:12/384) ·
hover marks for detail · silicon overlays land after the live sglang ladders</div>
<div class="grid" id="grid"></div>
<div class="note">Engine pin: sglang 0.5.14 + dynamo 1.3.1 (v0.5.17 rejected — glue incompatibility).
Router: KV defaults + temperature 0.0 (sweep: tuned variants within noise at selected cells; wrong tuning
costs up to 50% on agg). Curves regenerate from sglang/results/*.csv; recalibrated constants (v2) will
re-render after the first live points.</div>
</div>
<div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const PW=460,PH=300,M={t:12,r:14,b:38,l:52};
const FAM={"rr":["Round-robin","s1"],"kv-nvda":["KV defaults","s2"],"kv-t":["KV tuned","s3"],"ll":["Least-loaded","ref"]};
const ORDER=["rr","ll","kv-nvda","kv-t"];
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
 return `<g class="pt" data-tip='${tip.replace(/'/g,"&#39;")}'>${ring?`<circle cx="${x}" cy="${y}" r="8" fill="none" stroke="var(--ink-2)" stroke-width="1.4" stroke-dasharray="2 2"/>`:''}<circle cx="${x}" cy="${y}" r="9" fill="transparent"/><circle cx="${x}" cy="${y}" r="3.4" fill="var(--${c})"/></g>`;
}
function frontier(pts){const s=[...pts].sort((a,b)=>b.x-a.x||b.y-a.y);const f=[];let by=-1;
 for(const p of s){if(p.y>by){f.push(p);by=p.y;}}return f.sort((a,b)=>a.x-b.x);}
function lineChart(id,title,desc,ser,vfmt,ylabel,opts={}){
 panel(id,title,desc,()=> {
  const concs=[...new Set(Object.values(ser).flat().map(p=>p.conc))].sort((a,b)=>a-b);
  if(!concs.length) return '';
  const xmin=Math.log(concs[0]),xmax=Math.log(concs[concs.length-1]);
  const X=c=>M.l+((Math.log(c)-xmin)/(xmax-xmin))*(PW-M.l-M.r);
  const vals=Object.values(ser).flat().map(p=>p.v);
  const logy=opts.logy, ymin=logy?Math.log10(Math.max(.01,Math.min(...vals))):0, ymax=logy?Math.log10(Math.max(...vals)*1.3):Math.max(...vals)*1.12;
  const Y=v=>{const t=logy?Math.log10(Math.max(.01,v)):v;return PH-M.b-((t-ymin)/(ymax-ymin))*(PH-M.t-M.b);};
  const yt=logy?[0.1,1,10,100].filter(v=>Math.log10(v)>=ymin&&Math.log10(v)<=ymax):[0,.25,.5,.75,1].map(f=>Math.round(ymax*f*10)/10);
  let g=axes(X,Y,concs,yt,'concurrency (log)',ylabel,vfmt);
  if(opts.ref5&&Y(5)>M.t){g+=`<line x1="${M.l}" y1="${Y(5)}" x2="${PW-M.r}" y2="${Y(5)}" stroke="var(--axis)" stroke-dasharray="4 3"/><text x="${PW-M.r-2}" y="${Y(5)-4}" text-anchor="end" font-size="9.5" fill="var(--muted)">goodput 5 s</text>`;}
  for(const f of ORDER){const pts=ser[f];if(!pts)continue;const c=FAM[f][1];
   g+=`<path d="${pts.map((p,i)=>`${i?'L':'M'}${X(p.conc)} ${Y(p.v)}`).join('')}" fill="none" stroke="var(--${c})" stroke-width="2"/>`;
   for(const p of pts) g+=mark(X(p.conc),Y(p.v),c,
     JSON.stringify({s:FAM[f][0],conc:p.conc,v:vfmt?vfmt(p.v):p.v,tokps:p.tokps,ttft95:p.ttft95,ttft99:p.ttft99,hit:p.hit}),
     opts.ring&&p.conc===opts.ring&&(f==='rr'||f==='kv-t'||f==='kv-nvda'));
  }
  return g;
 },ORDER.filter(f=>ser[f]).map(f=>[FAM[f][0],FAM[f][1]]));
}
function hullPanel(id,title,desc,cloud,useXe){
 panel(id,title,desc,()=>{
  const all=Object.values(cloud).flat();if(!all.length)return '';
  const xv=p=>useXe?p.xe:p.x;
  const xmax=Math.max(...all.map(xv))*1.08,ymax=Math.max(...all.map(p=>p.y))*1.1;
  const X=v=>M.l+(v/xmax)*(PW-M.l-M.r),Y=v=>PH-M.b-(v/ymax)*(PH-M.t-M.b);
  const xt=[0,.25,.5,.75,1].map(f=>Math.round(xmax*f)),yt=[0,.25,.5,.75,1].map(f=>Math.round(ymax*f));
  let g=axes(X,Y,xt,yt,(useXe?'E2E ':'')+'tok/s per user','tok/s per GPU',v=>v);
  for(const f of ORDER){const raw=cloud[f];if(!raw)continue;const c=FAM[f][1];
   const pts=raw.map(p=>({...p,x:xv(p)}));const fr=frontier(pts);
   g+=`<path d="${fr.map((p,i)=>`${i?'L':'M'}${X(p.x)} ${Y(p.y)}`).join('')}" fill="none" stroke="var(--${c})" stroke-width="2" opacity=".85"/>`;
   for(const p of fr){const hollow=p.ttft95>5;
    g+=`<g class="pt" data-tip='${JSON.stringify({s:FAM[f][0],conc:p.conc,v:p.y.toFixed(1)+" tok/s/GPU",ttft95:p.ttft95})}'><circle cx="${X(p.x)}" cy="${Y(p.y)}" r="9" fill="transparent"/><circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3.6" fill="${hollow?'none':`var(--${c})`}" stroke="var(--${c})" stroke-width="1.8"/></g>`;}
  }
  return g;
 },ORDER.filter(f=>cloud[f]).map(f=>[FAM[f][0],FAM[f][1]]));
}
function sectionHeader(txt){document.getElementById('grid').insertAdjacentHTML('beforeend',
 `<div style="grid-column:1/-1;font-size:14px;font-weight:600;color:var(--ink);margin-top:8px">${txt}</div>`);}

sectionHeader('Disaggregated 24 GPU (3:3) — reference scale for cross-backend comparison');
lineChart('p1','1 · TTFT p95 vs concurrency (knee)','KV extends bounded-TTFT operation conc 32 → 96+',D.d24ttft,v=>v>=10?v.toFixed(0)+'s':v.toFixed(1)+'s','TTFT p95 (s, log)',{logy:true,ref5:true});
lineChart('p2','2 · Throughput vs concurrency','',D.d24tput,v=>v.toFixed(0),'output tok/s',{});
hullPanel('p3','3 · Efficiency frontier (hull)','Hollow = TTFT p95 above 5 s',D.d24cloud,false);
hullPanel('p4','4 · E2E-normalized frontier','TTFT included in per-user rate — routing separates here',D.d24cloud,true);

sectionHeader('Aggregated 24 GPU (6 × TP4) — selected cell conc 16 ringed');
lineChart('p5','5 · TTFT p95 vs concurrency','RR p95 ≈ 6.3 s at every conc (recompute floor, not queueing) — never goodput-feasible',D.aggttft,v=>v.toFixed(1)+'s','TTFT p95 (s)',{ref5:true,ring:16});
lineChart('p6','6 · Throughput vs concurrency','Both curves peak at conc 16; the bs≥8 TPOT cliff punishes deeper batching; no RR crossover (unlike trtllm)',D.aggtput,v=>v.toFixed(0),'output tok/s',{ring:16});
hullPanel('p7','7 · Efficiency frontier (aggregated, hull)','',D.aggcloud,false);
hullPanel('p8','8 · E2E-normalized frontier (aggregated)','',D.aggcloud,true);

sectionHeader('Disaggregated 72 GPU (18 × TP4) — selected cell 6:12 / conc 384 ringed');
lineChart('p9','9 · TTFT p95 vs concurrency (6:12)','KV holds ≤1.5 s p95 to conc 384; RR knees at ~96',D.d72ttft,v=>v>=10?v.toFixed(0)+'s':v.toFixed(1)+'s','TTFT p95 (s, log)',{logy:true,ref5:true,ring:384});
panel('p10','10 · Throughput vs concurrency by P:D split','Ordinal blues = prefill workers 3→15; dashed gray = RR at 6:12',()=>{
  const RAMP={'3:15':'o0','6:12':'o1','9:9':'o2','12:6':'o3','15:3':'o4'};
  const all=Object.values(D.d72pd).flatMap(s=>Object.values(s).flat());
  if(!all.length)return '';
  const concs=[...new Set(all.map(p=>p.conc))].sort((a,b)=>a-b);
  const xmin=Math.log(concs[0]),xmax=Math.log(concs[concs.length-1]);
  const X=c=>M.l+((Math.log(c)-xmin)/(xmax-xmin))*(PW-M.l-M.r);
  const ymax=Math.max(...all.map(p=>p.v))*1.1;
  const Y=v=>PH-M.b-(v/ymax)*(PH-M.t-M.b);
  let g=axes(X,Y,concs,[0,2000,4000,6000],'concurrency (log)','output tok/s (system)');
  for(const pd of Object.keys(RAMP)){
   const pts=(D.d72pd[pd]&&(D.d72pd[pd]['kv-t']||D.d72pd[pd]['kv-nvda']))||[];if(!pts.length)continue;
   g+=`<path d="${pts.map((p,i)=>`${i?'L':'M'}${X(p.conc)} ${Y(p.v)}`).join('')}" fill="none" stroke="var(--${RAMP[pd]})" stroke-width="2"/>`;
   for(const p of pts) g+=mark(X(p.conc),Y(p.v),RAMP[pd],JSON.stringify({s:'KV '+pd,conc:p.conc,v:p.v.toFixed(0)+' tok/s',ttft95:p.ttft95}),pd==='6:12'&&p.conc===384);
  }
  const rr=(D.d72pd['6:12']&&D.d72pd['6:12']['rr'])||[];
  g+=`<path d="${rr.map((p,i)=>`${i?'L':'M'}${X(p.conc)} ${Y(p.v)}`).join('')}" fill="none" stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 4"/>`;
  for(const p of rr) g+=mark(X(p.conc),Y(p.v),'ref',JSON.stringify({s:'RR 6:12',conc:p.conc,v:p.v.toFixed(0)+' tok/s',ttft95:p.ttft95}),false);
  return g;
 },[['KV 3:15','o0'],['KV 6:12','o1'],['KV 9:9','o2'],['KV 12:6','o3'],['KV 15:3','o4'],['RR 6:12 (dashed)','ref']]);
hullPanel('p11','11 · Efficiency frontier (72 GPU, 6:12, hull)','Hollow = TTFT p95 above 5 s',D.d72cloud,false);
panel('p12','12 · Cross-backend frontier — sglang vs trtllm (24 GPU, 3:3)',
 'Same simulator + trace, backend constants swapped: the KV-vs-RR gap is backend-independent; sglang shifts both frontiers toward higher interactivity',()=>{
  const sets=[['trtllm KV','ref',frontier(D.trtcloud['kv-t']||[])],['trtllm RR','o1',frontier(D.trtcloud['rr']||[])],
              ['sglang KV','s3',frontier(D.d24cloud['kv-t']||[])],['sglang RR','s1',frontier(D.d24cloud['rr']||[])]];
  const all=sets.flatMap(s=>s[2]);if(!all.length)return '';
  const xmax=Math.max(...all.map(p=>p.x))*1.08,ymax=Math.max(...all.map(p=>p.y))*1.1;
  const X=v=>M.l+(v/xmax)*(PW-M.l-M.r),Y=v=>PH-M.b-(v/ymax)*(PH-M.t-M.b);
  let g=axes(X,Y,[0,25,50,75,100],[0,40,80,120],'tok/s per user','tok/s per GPU',v=>v);
  for(const [name,c,fr] of sets){if(!fr.length)continue;
   const dash=name.includes('RR')?' stroke-dasharray="5 4"':'';
   g+=`<path d="${fr.map((p,i)=>`${i?'L':'M'}${X(p.x)} ${Y(p.y)}`).join('')}" fill="none" stroke="var(--${c})" stroke-width="2"${dash}/>`;
   for(const p of fr) g+=mark(X(p.x),Y(p.y),c,JSON.stringify({s:name,conc:p.conc,v:p.y.toFixed(1)+' tok/s/GPU',ttft95:p.ttft95}),false);}
  return g;
 },[['sglang KV','s3'],['sglang RR (dashed)','s1'],['trtllm KV','ref'],['trtllm RR (dashed)','o1']]);

const tip=document.getElementById('tip');
document.querySelectorAll('.pt').forEach(el=>{
 el.addEventListener('mousemove',e=>{const p=JSON.parse(el.dataset.tip);
  tip.innerHTML=`<b>${p.s}</b>${p.conc?` · conc ${p.conc}`:''}<br>${p.v}`+
   (p.ttft95!==undefined?`<br><span style="color:var(--ink-2)">TTFT p95 ${(+p.ttft95).toFixed(2)} s${p.ttft99?` · p99 ${(+p.ttft99).toFixed(2)} s`:''}</span>`:'')+
   (p.hit!==undefined?`<br><span style="color:var(--ink-2)">reuse ${(+p.hit).toFixed(1)}%</span>`:'');
  tip.style.left=Math.min(e.clientX+13,innerWidth-260)+'px';tip.style.top=(e.clientY+13)+'px';tip.style.opacity=1;});
 el.addEventListener('mouseleave',()=>tip.style.opacity=0);
});
</script>
"""

if __name__ == "__main__":
    main()
