import json
from pathlib import Path

inventory = json.loads(Path('results/phase1_development20_annotation_inventory.json').read_text())

annotations = {
    "split": "development",
    "annotation_pass": "first_pass_rgb_only",
    "instruction": "Record support labels from RGB image only. Do not view depth.npy, reference.json, or 3D geometry. Mark uncertain cases as 'uncertain'. All labels are provisional pending human review.",
    "scenes": []
}

for scene in inventory['scenes']:
    scene_annotation = {
        "scene_id": scene['scene_id'],
        "metadata_index": scene['metadata_index'],
        "candidate_child_class": scene['candidate_child_class'],
        "candidate_parent_class": scene['candidate_parent_class'],
        "rgb_first_pass_support_label": None,
        "rgb_first_pass_confidence": None,
        "rgb_first_pass_notes": "",
        "provisional": True,
        "human_reviewed": False,
        "final_support_label": None,
        "abstained": False,
        "abstention_reason": ""
    }
    annotations['scenes'].append(scene_annotation)

Path('results/phase1_development20_first_pass_annotations.json').write_text(
    json.dumps(annotations, indent=2)
)
print(f"Created template for {len(annotations['scenes'])} scenes")
print("Next step: annotate each scene from RGB only, then commit the first-pass labels.")
