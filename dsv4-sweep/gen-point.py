#!/usr/bin/env python3
"""Generate DSv4 point manifests (server + bench) from point3 as the template.
Usage: gen-point.py <id> <pnodes> <pworkers> <dtp> <dnodes> <conc> [pmaxreq] [pcgbs] [dmaxreq] [dcgbs] [dmemfrac]
All values come from the vendored srt-slurm recipes (see reference/)."""
import sys, re, os

pid, pnodes, pworkers, dtp, dnodes, conc = sys.argv[1:7]
pmaxreq, pcgbs, dmaxreq, dcgbs, dmemfrac = (sys.argv[7:12] + ['256','256','3072','256','0.94'])[:5]
# parity extras: deepgemm tokens/rank (decode), spec steps, draft tokens, context-len, swa ratio
dtokrank, spsteps, spdraft, ctxlen, swaratio = (sys.argv[12:17] + ['2048','3','4','16384','0.15'])[:5]
D = os.path.expanduser('~/dsr1-pareto/dsv4-sweep/manifests')
s = open(f'{D}/point3-1p1d-dep4-dep8-mtp.yaml').read()

s = s.replace('dsv4-p3-', f'dsv4-{pid}-')
s = s.replace('numNodes: 3', f'numNodes: {int(pnodes)+int(dnodes)}')
# prefill worker count
i = s.find(f'name: dsv4-{pid}-prefill')
j = s.find(f'name: dsv4-{pid}-decode')
pre = s[i:j].replace('replicas: 1', f'replicas: {pworkers}', 1)
pre = pre.replace('--max-running-requests 256', f'--max-running-requests {pmaxreq}')
pre = pre.replace('--cuda-graph-max-bs 256', f'--cuda-graph-max-bs {pcgbs}')
s = s[:i] + pre + s[j:]
# decode: TP/DP/EP + nodes + knobs
dec = s[j:]
for k in ['tensor-parallel-size', 'data-parallel-size', 'expert-parallel-size']:
    dec = dec.replace(f'--{k} 8', f'--{k} {dtp}')
dec = dec.replace('--nnodes 2', f'--nnodes {dnodes}')
dec = dec.replace('replicas: 2', f'replicas: {dnodes}', 1)
dec = dec.replace('--max-running-requests 3072', f'--max-running-requests {dmaxreq}')
dec = dec.replace('--cuda-graph-max-bs 256', f'--cuda-graph-max-bs {dcgbs}')
dec = dec.replace('--mem-fraction-static 0.94', f'--mem-fraction-static {dmemfrac}')
dec = dec.replace('--speculative-num-steps 3', f'--speculative-num-steps {spsteps}')
dec = dec.replace('--speculative-num-draft-tokens 4', f'--speculative-num-draft-tokens {spdraft}')
dec = dec.replace('--context-length 16384', f'--context-length {ctxlen}')
dec = dec.replace('--swa-full-tokens-ratio 0.15', f'--swa-full-tokens-ratio {swaratio}')
dec = dec.replace('''- name: SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK
          value: "2048"''', f'''- name: SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK
          value: "{dtokrank}"''')
s = s[:j] + dec
s = s.replace('# DSv4 point 3: disagg-mid-curve-1p1d-dep4-dep8-mtp (verbatim recipe flags).',
              f'# DSv4 {pid}: generated from recipe (see reference/). prefill {pworkers}w DEP4 x{pnodes}n, decode DEP{dtp} x{dnodes}n.')
open(f'{D}/{pid}.yaml', 'w').write(s)

# bench
b = open(f'{D}/bench-template.yaml').read()
total = (int(pnodes) + int(dnodes)) * 4
b = b.replace('JOBSUFFIX', pid).replace('FRONTSVC', f'{pid}-frontend')
b = b.replace('CONCLIST', conc).replace('RATE', 'inf')
b = b.replace('GPUTOTAL', str(total)).replace('PGPU', str(int(pnodes)*4)).replace('DGPU', str(int(dnodes)*4))
open(f'{D}/bench-{pid}.yaml', 'w').write(b)
print(f'{pid}: server+bench written (nodes {pnodes}P+{dnodes}D={int(pnodes)+int(dnodes)}, conc {conc})')
