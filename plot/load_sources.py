"""Read one primary summary per server; do not double-count question exports."""
from pathlib import Path
import csv,json,hashlib,collections

R=Path(__file__).resolve().parent
config=json.loads((R/'source_config.json').read_text())
# Relative roots are anchored to source_config.json, independent of the working directory.
root=(R/Path(config['root'])).resolve()
out=R/'source_data';out.mkdir(exist_ok=True)
rows=[];audit=[]
for entry in config['files']:
    path=root/entry['path']
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    header=next(i for i,line in enumerate(lines) if line.startswith('dtype,'))
    data=list(csv.DictReader(lines[header:]))
    keys=('dtype','model','dataset','fc_mode','scheme','batch_size','question_index')
    identifiers=[tuple(row.get(key,'') for key in keys) for row in data]
    assert len(identifiers)==len(set(identifiers)),f'Duplicate records in {path}'
    audit.append(dict(path=str(path),hardware=entry['hardware'],rows=len(data),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      models=dict(collections.Counter(row.get('model','') for row in data))))
    rows.extend(dict(row,_source=str(path),_hardware=entry['hardware']) for row in data)
(out/'normalized_source_rows.json').write_text(json.dumps(rows))
(out/'source_audit.json').write_text(json.dumps(audit,indent=2))
print(f'Read {len(rows)} records from {len(audit)} primary CSVs under {root}')
