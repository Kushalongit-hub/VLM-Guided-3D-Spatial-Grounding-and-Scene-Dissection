import json
from pathlib import Path

prior = set()
for f in Path('results').glob('sunrgbd_*_selection.json'):
    if 'phase1' in f.name:
        continue
    data = json.loads(f.read_text())
    for s in data.get('scenes', []):
        prior.add(s['metadata_index'])

for fname in ['sunrgbd_phase1_development20_selection.json', 'sunrgbd_phase1_final40_selection.json']:
    data = json.loads((Path('results') / fname).read_text())
    overlap = [s['metadata_index'] for s in data['scenes'] if s['metadata_index'] in prior]
    print(f'{fname}: {len(data["scenes"])} scenes, overlap with prior: {len(overlap)}')
