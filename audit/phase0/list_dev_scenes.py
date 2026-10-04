import json
from pathlib import Path

inventory = json.loads(Path('results/phase1_development20_annotation_inventory.json').read_text())
for scene in inventory['scenes']:
    print(f"{scene['scene_id']}: {scene['candidate_child_class']} + {scene['candidate_parent_class']}")
