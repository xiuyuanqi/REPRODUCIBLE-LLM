from pathlib import Path
import json,csv,hashlib,math,statistics
R=Path(__file__).resolve().parent;D=json.loads((R/'plot_data.json').read_text())
base=R/'source_data';original=json.loads((base/'plot_data.json').read_text())
for k,v in original.items():assert D[k]==v,('Existing panel input altered',k)
for p,h in D['base_artifacts'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
for p,h in D['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
config=json.loads((R/'source_config.json').read_text())
source_root=(R/Path(config['root'])).resolve()
assert len(D['source_sha256'])==len(config['files'])==5
assert all(Path(p).is_relative_to(source_root) for p in D['source_sha256'])
source_records={}
identity=lambda r:tuple(str(r[k]) for k in ('dtype','model','dataset','fc_mode','scheme','batch_size','question_index'))
for path in D['source_sha256']:
 with Path(path).open(newline='',encoding='utf-8-sig') as f:
  for row in csv.DictReader(f):source_records[path,identity(row)]=row
selected=json.loads((base/'chosen_paired_rows.json').read_text())
assert len(selected)==4*D['chosen_paired_conditions']
for row in selected:
 source=source_records[row['_source'],identity(row)]
 for key in ('tok1_divergent_pairs','top20_logit_divergent_pairs','median_prefill_cuda_ms','median_decode_cuda_ms','mean_useful_prompt_tokens_per_second','pair_count','completed_repeats','sdpa_policies','timing_scope'):
  assert row[key]==source[key],(row['_source'],identity(row),key)
for point in D['heatmaps']:
 rows=[row for row in selected if all(row[k]==point[k] for k in ('hardware','model','scheme','batch_size'))]
 by_dataset={name:[r for r in rows if r['dataset']==name] for name in {r['dataset'] for r in rows}}
 for column,divergent in [('token_agreement','tok1_divergent_pairs'),('top20_agreement','top20_logit_divergent_pairs')]:
  if not rows:assert point[column] is None
  else:
   expected=statistics.mean(100*sum(int(r['pair_count'])-int(r[divergent]) for r in group)/sum(int(r['pair_count']) for r in group) for group in by_dataset.values())
   assert abs(expected-point[column])<1e-10
# Re-read the configured primary A6000 summaries independently of the snapshot.
raw=[]
for entry in config['files']:
 if entry['hardware']=='A6000':
  with (source_root/entry['path']).open(newline='',encoding='utf-8-sig') as f:raw.extend(csv.DictReader(f))
new=D['dataset_detail'];assert len(new)==32
for point in new:
 rr=[r for r in raw if r['model']==point['model'] and r['dataset']==point['dataset'] and r['scheme']==point['scheme'] and r['batch_size']=='8' and r['fc_mode']=='prompt' and r['dtype']=='BF16' and r['question_index'] in ('1','2','3','4')]
 assert len(rr)==4 and all(r['completed_repeats']=='10' and r['sdpa_policies']=='cuDNN_disabled' and r['timing_scope']=='question' for r in rr)
 equal=sum(int(r['pair_count'])-int(r['tok1_divergent_pairs']) for r in rr);pairs=sum(int(r['pair_count']) for r in rr)
 assert pairs==180 and point['run_pairs']==180 and point['token_equal_pairs']==equal
 assert abs(100*equal/pairs-point['token_agreement_pct'])<1e-10
 expected=sum(float(r['median_decode_cuda_ms']) for r in rr)/4000
 assert abs(expected-point['decode_time_s'])<1e-10
 assert 0<=point['token_agreement_pct']<=100 and point['decode_time_s']>0
 if point['scheme'] in ('scheme1','scheme2_fixedL_capacity','corun_style'):assert point['token_agreement_pct']==100
report=dict(passed=True,core_panel_inputs_match_current_sources=True,source_files_unchanged=True,all_sources_under_configured_root=True,primary_source_files=5,selected_policy_rows_checked=len(selected),heatmap_cells_checked=len(D['heatmaps']),new_dataset_policy_points=32,questions_per_point=4,run_pairs_per_point=180,independent_validation='Direct re-read of configured primary CSVs',agreement='actual generated token sequences; fixed B=8',decode_time='seconds; arithmetic mean of four question-level median CUDA decode times; log axis')
(R/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
