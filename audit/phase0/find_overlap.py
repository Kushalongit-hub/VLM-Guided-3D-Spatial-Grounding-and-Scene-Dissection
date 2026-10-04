import json
from pathlib import Path

final = json.loads(Path('results/sunrgbd_phase1_final40_selection.json').read_text())
final_indices = {s['metadata_index'] for s in final['scenes']}

for f in Path('results').glob('sunrgbd_*_selection.json'):
    if 'phase1' in f.name:
        continue
    data = json.loads(f.read_text())
    prior_indices = {s['metadata_index'] for s in data.get('scenes', [])}
    overlap = final_indices & prior_indices
    if overlap:
        print(f'{f.name}: {len(overlap)} overlap: {sorted(overlap)}')
        for idx in overlap:
            scene = next(s for s in final['scenes'] if s['metadata_index'] == idx)
            print(f'  Scene: {scene}')
