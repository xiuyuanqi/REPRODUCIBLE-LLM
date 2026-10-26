from pathlib import Path
import json,csv,collections,math,statistics,hashlib
R=Path(__file__).resolve().parent/'source_data'
ALL=json.loads((R/'normalized_source_rows.json').read_text())
MODELS=['Qwen--Qwen3-8B','Qwen--Qwen3-14B','meta-llama--Llama-3.1-8B-Instruct','google--gemma-3-4b-it','google--gemma-3-12b-it']
SHORT=['Qwen3-8B','Qwen3-14B','Llama3.1-8B','Gemma3-4B','Gemma3-12B']
SCHEMES=['normal','corun_style','scheme1','scheme2_fixedL_capacity'];BATCHES=[1,2,4,8,16];HARDWARE=['A6000','A100','H100']
selected=[];excluded=collections.Counter();sources=set()
for r0 in ALL:
 s=r0['_source'];m=r0.get('model');hw=r0['_hardware']
 if hw is None:continue
 sources.add(s)
 r=dict(r0,hardware=hw)
 if r.get('dtype')!='BF16' or r.get('fc_mode')!='prompt' or r.get('scheme') not in SCHEMES or m not in MODELS:excluded['outside BF16/prompt/canonical models/schemes']+=1;continue
 if r.get('question_index') not in ('1','2','3','4'):excluded['outside q1-q4']+=1;continue
 if r.get('sdpa_policies')!='cuDNN_disabled' or r.get('timing_scope')!='question':excluded['backend or timing scope unverified']+=1;continue
 if r.get('completed_repeats')!='10' or r.get('pair_count')!='45':excluded['incomplete repetitions']+=1;continue
 required=['tok1_divergent_pairs','top20_logit_divergent_pairs','median_prefill_cuda_ms','median_decode_cuda_ms','mean_useful_prompt_tokens_per_second']
 if any(r.get(k) in ('',None) for k in required):excluded['missing required metric']+=1;continue
 for k in ('tok1_divergent_pairs','top20_logit_divergent_pairs'):
  assert 0<=int(r[k])<=45
 assert r.get('tok1_comparison_source')=='generated_token_ids'
 r['batch_size']=int(r['batch_size']);r['question_index']=int(r['question_index']);selected.append(r)
keys=lambda r:(r['hardware'],r['model'],r['dataset'],r['question_index'],r['batch_size'])
groups=collections.defaultdict(dict)
for r in selected:
 k=keys(r);assert r['scheme'] not in groups[k],('Duplicate chosen source',k,r['scheme']);groups[k][r['scheme']]=r
paired={k:g for k,g in groups.items() if set(g)==set(SCHEMES)}
paired_rows=[r for g in paired.values() for r in g.values()]
coverage=[];heat=[]
# Dataset-balanced mean: average per-question run-pair agreement within each dataset,
# then arithmetic mean over the available datasets. Identical matched cohort for all policies.
for hw in HARDWARE:
 for model in MODELS:
  for b in BATCHES:
   gg=[g for k,g in paired.items() if k[0]==hw and k[1]==model and k[4]==b]
   for scheme in SCHEMES:
    rows=[g[scheme] for g in gg]
    byds=collections.defaultdict(list)
    for r in rows:byds[r['dataset']].append(r)
    def rate(key):
     if not rows:return None
     return statistics.mean(100*(1-sum(int(r[key]) for r in rs)/(45*len(rs))) for rs in byds.values())
    d=dict(hardware=hw,model=model,batch_size=b,scheme=scheme,token_agreement=rate('tok1_divergent_pairs'),top20_agreement=rate('top20_logit_divergent_pairs'),datasets=len(byds),target_conditions=len(rows),run_pairs=45*len(rows),questions=','.join(map(str,sorted({r['question_index'] for r in rows}))))
    heat.append(d)
   coverage.append(dict(hardware=hw,model=model,batch_size=b,datasets=len({g['normal']['dataset'] for g in gg}),target_conditions=len(gg)))
# Performance comparisons are normalized within each hardware/model, not across model sizes.
PMODELS={'A6000':MODELS[1],'A100':MODELS[0],'H100':MODELS[1]}
datasets=sorted({k[2] for k in paired if k[2]!='BFCL_v3_multiple'})
common=[];perf_exclusions=[]
for dataset in datasets:
 for q in range(1,5):
  kk=[(hw,PMODELS[hw],dataset,q,b) for hw in HARDWARE for b in BATCHES]
  if not all(k in paired for k in kk):perf_exclusions.append(dict(dataset=dataset,q=q,reason='missing matched full hardware/B grid'));continue
  okay=True
  for k in kk:
   g=paired[k];modes={r.get('logit_capture_modes','unspecified') for r in g.values()}
   if len(modes)>1 or any(r.get('mixed_logit_capture_timing')=='True' for r in g.values()):okay=False
   real=[float(r['mean_real_prompt_tokens']) for r in g.values()]
   if max(real)-min(real)>1e-6:okay=False
  if not okay:perf_exclusions.append(dict(dataset=dataset,q=q,reason='capture-mode or input-token mismatch'));continue
  common.append((dataset,q))
assert len(common)==28, ('Expected seven tinyBenchmarks x four questions',common)
stacks=[]
for hw in HARDWARE:
 model=PMODELS[hw]
 for b in (8,):
  for scheme in SCHEMES:
   byds=collections.defaultdict(list)
   for dataset,q in common:
    g=paired[(hw,model,dataset,q,b)];r=g[scheme];base=g['normal'];den=float(base['median_prefill_cuda_ms'])+float(base['median_decode_cuda_ms'])
    byds[dataset].append((float(r['median_prefill_cuda_ms'])/den,float(r['median_decode_cuda_ms'])/den))
   pref=statistics.mean(statistics.mean(t[0] for t in v) for v in byds.values());dec=statistics.mean(statistics.mean(t[1] for t in v) for v in byds.values())
   stacks.append(dict(hardware=hw,model=model,scheme=scheme,batch_size=b,prefill_ratio=pref,decode_ratio=dec,total_ratio=pref+dec,datasets=len(byds),target_conditions=len(common)))
curves=[];perfpoints=[]
for hw in HARDWARE:
 model=PMODELS[hw]
 for b in BATCHES:
  for scheme in SCHEMES:
   byds=collections.defaultdict(list)
   for dataset,q in common:
    g=paired[(hw,model,dataset,q,b)];r=g[scheme];base=g['normal']
    throughput=float(r['mean_useful_prompt_tokens_per_second']);baseline=float(base['mean_useful_prompt_tokens_per_second']);ratio=throughput/baseline
    assert ratio>0 and math.isfinite(ratio)
    byds[dataset].append(math.log(ratio));perfpoints.append(dict(hardware=hw,model=model,dataset=dataset,question_index=q,batch_size=b,scheme=scheme,useful_input_tokens_s=throughput,padded_useful_input_tokens_s=baseline,ratio=ratio))
   curves.append(dict(hardware=hw,model=model,batch_size=b,scheme=scheme,throughput_ratio=math.exp(statistics.mean(statistics.mean(logs) for logs in byds.values())),models=1,datasets=len(byds),target_conditions=len(common)))
def dumpcsv(name,rows):
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def clean(r):return {k:v for k,v in r.items() if not k.startswith('_')}
report=dict(hardware=HARDWARE,models=MODELS,model_labels=SHORT,schemes=SCHEMES,batches=BATCHES,heatmaps=heat,stacks=stacks,curves=curves,coverage=coverage,performance_models=PMODELS,performance_cohort=[dict(dataset=d,question=q) for d,q in common],exclusions=dict(excluded),unpaired_conditions=len(groups)-len(paired),chosen_paired_conditions=len(paired),performance_exclusions=perf_exclusions,source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sorted(sources)})
(R/'plot_data.json').write_text(json.dumps(report,indent=2))
(R/'chosen_paired_rows.json').write_text(json.dumps(paired_rows))
dumpcsv('figure_heatmap_values.csv',heat);dumpcsv('figure_runtime_values.csv',stacks);dumpcsv('figure_prefill_efficiency.csv',curves);dumpcsv('paired_throughput_points.csv',perfpoints)
print('Paired conditions',len(paired),'performance conditions per hardware/B/method',len(common),'exclusions',dict(excluded),'unpaired',len(groups)-len(paired))
for x in stacks:print('STACK',x)
for x in curves:
 if x['batch_size'] in (1,8,16):print('CURVE',x)
for x in heat:
 if x['scheme']!='normal' and x['top20_agreement'] is not None and x['top20_agreement']<100:print('NONDETERMINISTIC',x)
