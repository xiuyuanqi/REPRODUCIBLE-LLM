from pathlib import Path
import json,csv,collections,statistics,hashlib,subprocess,sys
R=Path(__file__).resolve().parent
BASE=R/'source_data'
subprocess.run([sys.executable,str(R/'load_sources.py')],check=True)
subprocess.run([sys.executable,str(R/'build_core_data.py')],check=True)
D=json.loads((BASE/'plot_data.json').read_text());rows=json.loads((BASE/'chosen_paired_rows.json').read_text())
DATASETS=[('tinyAI2_arc','AI2 ARC'),('tinyAlpacaEval','AlpacaEval'),('tinyGSM8k','GSM8K'),('tinyHellaswag','HellaSwag'),('tinyMMLU','MMLU'),('tinyTruthfulQA','TruthfulQA'),('tinyWinogrande','WinoGrande'),('BFCL_v3_multiple','BFCL v3')]
chosen=[r for r in rows if r['hardware']=='A6000' and r['model']=='Qwen--Qwen3-14B' and r['batch_size']==8]
assert len(chosen)==128
result=[]
for dataset,label in DATASETS:
 for scheme in D['schemes']:
  rr=[r for r in chosen if r['dataset']==dataset and r['scheme']==scheme]
  assert len(rr)==4 and {r['question_index'] for r in rr}=={1,2,3,4}
  pairs=sum(int(r['pair_count']) for r in rr);div=sum(int(r['tok1_divergent_pairs']) for r in rr)
  result.append(dict(hardware='A6000',model='Qwen--Qwen3-14B',dataset=dataset,dataset_label=label,batch_size=8,scheme=scheme,questions=4,repeats_per_question=10,run_pairs=pairs,token_equal_pairs=pairs-div,token_agreement_pct=100*(pairs-div)/pairs,decode_time_s=statistics.mean(float(r['median_decode_cuda_ms']) for r in rr)/1000,decode_statistic='mean of four question-level ten-repeat medians',decode_work='one target' if scheme=='scheme2_fixedL_capacity' else 'B requests sequential full generations' if scheme=='corun_style' else 'B requests batched'))
D['dataset_detail']=result;D['dataset_detail_order']=[dict(dataset=d,label=l) for d,l in DATASETS]
D['base_artifacts']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'plot_data.json',BASE/'chosen_paired_rows.json')}
(R/'plot_data.json').write_text(json.dumps(D,indent=2))
with (R/'figure_a6000_dataset_values.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
(R/'dataset_detail_source_rows.json').write_text(json.dumps(chosen))
print('Added',len(result),'dataset-policy points, based on',len(chosen),'question-policy conditions.')

import subprocess,sys
subprocess.run([sys.executable,str(R/"export_panel_d.py")],check=True)
