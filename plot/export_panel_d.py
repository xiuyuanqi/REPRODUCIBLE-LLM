"""Export the actual inputs and annotated comparison factors for panel d."""
from pathlib import Path
import json,csv,statistics,collections,math
R=Path(__file__).resolve().parent
D=json.loads((R/'plot_data.json').read_text())
base=R/'source_data'
rows=json.loads((base/'chosen_paired_rows.json').read_text())
NAMES={'normal':'STD','corun_style':'CoRun','scheme1':'BlockLock-Rows','scheme2_fixedL_capacity':'BlockLock-Packed'}
MODELS=D['performance_models'];cohort={(c['dataset'],c['question']) for c in D['performance_cohort']}
lookup={}
for r in rows:
 if r['model']==MODELS[r['hardware']] and r['batch_size']==8 and (r['dataset'],r['question_index']) in cohort:
  key=(r['hardware'],r['dataset'],r['question_index'],r['scheme']);assert key not in lookup;lookup[key]=r
assert len(lookup)==336
raw=[];summary=[]
for hw in D['hardware']:
 for scheme in D['schemes']:
  rr=[]
  for dataset,q in sorted(cohort):
   r=lookup[hw,dataset,q,scheme];normal=lookup[hw,dataset,q,'normal']
   p=float(r['median_prefill_cuda_ms']);d=float(r['median_decode_cuda_ms']);n=float(normal['median_prefill_cuda_ms'])+float(normal['median_decode_cuda_ms'])
   point=dict(hardware=hw,model=r['model'],dataset=dataset,question_index=q,batch_size=8,scheme=scheme,display_name=NAMES[scheme],completed_repeats=int(r['completed_repeats']),source_median_prefill_cuda_ms=p,source_median_decode_cuda_ms=d,prefill_plus_decode_ms=p+d,source_median_generate_cuda_ms=r.get('median_generate_cuda_ms',''),matched_STD_prefill_plus_decode_ms=n,plotted_prefill_ratio=p/n,plotted_decode_ratio=d/n,plotted_total_ratio=(p+d)/n,source_csv=r['_source'])
   raw.append(point);rr.append(point)
  bar=next(x for x in D['stacks'] if x['hardware']==hw and x['scheme']==scheme)
  assert len(rr)==28
  for k,bar_key in [('plotted_prefill_ratio','prefill_ratio'),('plotted_decode_ratio','decode_ratio'),('plotted_total_ratio','total_ratio')]:assert abs(statistics.mean(x[k] for x in rr)-bar[bar_key])<1e-10
  summary.append(dict(hardware=hw,model=MODELS[hw],batch_size=8,display_name=NAMES[scheme],scheme=scheme,datasets=7,question_conditions=28,mean_question_median_prefill_s=statistics.mean(x['source_median_prefill_cuda_ms'] for x in rr)/1000,mean_question_median_decode_s=statistics.mean(x['source_median_decode_cuda_ms'] for x in rr)/1000,mean_question_median_total_s=statistics.mean(x['prefill_plus_decode_ms'] for x in rr)/1000,plotted_prefill_ratio=bar['prefill_ratio'],plotted_decode_ratio=bar['decode_ratio'],plotted_total_ratio=bar['total_ratio']))
factors=[]
for hw in D['hardware']:
 group={r['scheme']:r for r in summary if r['hardware']==hw};ref=group['corun_style'];normal=group['normal']
 for scheme,r in group.items():
  r['ratio_of_mean_times_vs_STD']=r['mean_question_median_total_s']/normal['mean_question_median_total_s']
  r['CoRun_over_method_plotted_height']=ref['plotted_total_ratio']/r['plotted_total_ratio']
  r['CoRun_over_method_mean_time']=ref['mean_question_median_total_s']/r['mean_question_median_total_s']
  if scheme!='corun_style':factors.append(dict(hardware=hw,model=MODELS[hw],batch_size=8,reference='CoRun',compared_method=r['display_name'],scheme=scheme,CoRun_plotted_height=ref['plotted_total_ratio'],method_plotted_height=r['plotted_total_ratio'],annotated_CoRun_over_method=ref['plotted_total_ratio']/r['plotted_total_ratio'],ratio_of_absolute_mean_times=ref['mean_question_median_total_s']/r['mean_question_median_total_s'],annotation_basis='ratio of the two plotted heights, each a mean of matched STD-normalized question times'))
def savecsv(name,a):
 with (R/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
savecsv('panel_d_question_level.csv',raw);savecsv('panel_d_summary.csv',summary);savecsv('panel_d_corun_factors.csv',factors)
(R/'panel_d_data.json').write_text(json.dumps(dict(raw_rows=raw,summary=summary,factors=factors),indent=2))
for r in summary:print(r['hardware'],r['display_name'],f"prefill={r['mean_question_median_prefill_s']:.6f}s",f"decode={r['mean_question_median_decode_s']:.6f}s",f"total={r['mean_question_median_total_s']:.6f}s",f"plot={r['plotted_total_ratio']:.6f}")
for r in factors:print('FACTOR',r['hardware'],r['compared_method'],r['annotated_CoRun_over_method'])
