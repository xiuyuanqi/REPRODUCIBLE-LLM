from pathlib import Path
import json,csv,statistics,hashlib,collections
R=Path(__file__).resolve().parent
D=json.loads((R/'plot_data.json').read_text());P=json.loads((R/'panel_d_data.json').read_text())
assert len(P['raw_rows'])==336 and len(P['summary'])==12 and len(P['factors'])==9
cache={}
for row in P['raw_rows']:
 path=row['source_csv']
 if path not in cache:
  with Path(path).open(encoding='utf-8-sig',newline='') as f:sr=list(csv.DictReader(f))
  cache[path]={(r['model'],r['dataset'],int(r.get('question_index') or -1),int(r['batch_size']),r['scheme'],r['fc_mode'],r['dtype']):r for r in sr}
 r=cache[path][row['model'],row['dataset'],row['question_index'],8,row['scheme'],'prompt','BF16']
 assert r['completed_repeats']=='10' and r['sdpa_policies']=='cuDNN_disabled'
 assert float(r['median_prefill_cuda_ms'])==row['source_median_prefill_cuda_ms']
 assert float(r['median_decode_cuda_ms'])==row['source_median_decode_cuda_ms']
for item in P['summary']:
 raw=[r for r in P['raw_rows'] if r['hardware']==item['hardware'] and r['scheme']==item['scheme']]
 assert len(raw)==28
 for k,col in [('plotted_prefill_ratio','plotted_prefill_ratio'),('plotted_decode_ratio','plotted_decode_ratio'),('plotted_total_ratio','plotted_total_ratio')]:assert abs(statistics.mean(r[k] for r in raw)-item[col])<1e-10
 assert abs(statistics.mean(r['prefill_plus_decode_ms'] for r in raw)/1000-item['mean_question_median_total_s'])<1e-10
for item in P['factors']:
 num=next(r for r in P['summary'] if r['hardware']==item['hardware'] and r['scheme']=='corun_style')
 den=next(r for r in P['summary'] if r['hardware']==item['hardware'] and r['scheme']==item['scheme'])
 assert abs(item['annotated_CoRun_over_method']-num['plotted_total_ratio']/den['plotted_total_ratio'])<1e-10
 assert abs(item['ratio_of_absolute_mean_times']-num['mean_question_median_total_s']/den['mean_question_median_total_s'])<1e-10
for path,sha in D['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
(R/'panel_d_verification.json').write_text(json.dumps(dict(passed=True,question_rows=336,summary_rows=12,factors=9,raw_rows_checked_against_original_csv=True,source_files_unchanged=True,annotated_factor_basis='CoRun plotted height divided by method plotted height',absolute_mean_time_ratios_exported_separately=True),indent=2))
print('Verified 336 raw-source matches, 12 aggregates and 9 factor labels.')
