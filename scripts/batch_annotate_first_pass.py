"""Batch RGB-only provisional annotation using overlap heuristics.

This script generates provisional first-pass annotations based on RGB-only
cues (bounding box overlap and spatial proximity). All labels are marked
as provisional and should be reviewed by a human annotator.

Usage:
    .\.venv\Scripts\python.exe scripts/batch_annotate_first_pass.py
"""

import argparse
import json
from pathlib import Path


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [xmin, ymin, xmax, ymax] format."""
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    
    if x1 <= x0 or y1 <= y0:
        return 0.0
    
    intersection = (x1 - x0) * (y1 - y0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_center_distance(box1, box2):
    """Compute distance between box centers."""
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def is_child_above_parent(child_box, parent_box):
    """Check if child center is above parent center (y-axis, image coordinates)."""
    child_cy = (child_box[1] + child_box[3]) / 2
    parent_cy = (parent_box[1] + parent_box[3]) / 2
    return child_cy < parent_cy


def is_child_within_parent(child_box, parent_box, threshold=0.8):
    """Check if child is mostly within parent bounds."""
    x0 = max(child_box[0], parent_box[0])
    y0 = max(child_box[1], parent_box[1])
    x1 = min(child_box[2], parent_box[2])
    y1 = min(child_box[3], parent_box[3])
    
    if x1 <= x0 or y1 <= y0:
        return False
    
    child_area = (child_box[2] - child_box[0]) * (child_box[3] - child_box[1])
    overlap_area = (x1 - x0) * (y1 - y0)
    
    return overlap_area / child_area > threshold if child_area > 0 else False


def heuristic_annotate_scene(scene: dict, data_root: Path) -> dict:
    """Generate provisional RGB-only annotation using heuristics."""
    scene_id = scene["scene_id"]
    child_class = scene["candidate_child_class"]
    parent_class = scene["candidate_parent_class"]
    
    objects_path = data_root / scene_id / "objects.json"
    
    try:
        objects = json.loads(objects_path.read_text())
    except Exception as e:
        print(f"Warning: Cannot load objects for {scene_id}: {e}")
        return None
    
    # Find child and parent objects
    child_objs = [obj for obj in objects if obj["label"] == child_class]
    parent_objs = [obj for obj in objects if obj["label"] == parent_class]
    
    if not child_objs or not parent_objs:
        print(f"Warning: Missing child or parent in {scene_id}")
        return {
            "scene_id": scene_id,
            "metadata_index": scene["metadata_index"],
            "candidate_child_class": child_class,
            "candidate_parent_class": parent_class,
            "rgb_first_pass_support_label": "none",
            "rgb_first_pass_confidence": "low",
            "rgb_first_pass_notes": "missing objects",
            "provisional": True,
            "human_reviewed": False,
            "final_support_label": None,
            "abstained": False,
            "abstention_reason": "missing_objects"
        }
    
    # Use first instance of each class
    child = child_objs[0]
    parent = parent_objs[0]
    
    child_box = child["bbox_2d"]
    parent_box = parent["bbox_2d"]
    
    # Compute heuristics
    iou = compute_iou(child_box, parent_box)
    center_dist = compute_center_distance(child_box, parent_box)
    child_above = is_child_above_parent(child_box, parent_box)
    child_within = is_child_within_parent(child_box, parent_box)
    
    # Heuristic decision
    if iou > 0.1 and child_above and child_within:
        label = "supported_by"
        confidence = "medium"
        notes = f"high overlap (IoU={iou:.2f}), child above parent, within bounds"
    elif iou > 0.3 and child_above:
        label = "supported_by"
        confidence = "medium"
        notes = f"moderate overlap (IoU={iou:.2f}), child above parent"
    elif child_within and child_above:
        label = "supported_by"
        confidence = "low"
        notes = f"child within parent bounds and above, but low overlap (IoU={iou:.2f})"
    elif iou > 0.1:
        label = "uncertain"
        confidence = "low"
        notes = f"some overlap (IoU={iou:.2f}) but unclear support"
    else:
        label = "none"
        confidence = "low"
        notes = f"no significant overlap (IoU={iou:.2f})"
    
    return {
        "scene_id": scene_id,
        "metadata_index": scene["metadata_index"],
        "candidate_child_class": child_class,
        "candidate_parent_class": parent_class,
        "rgb_first_pass_support_label": label,
        "rgb_first_pass_confidence": confidence,
        "rgb_first_pass_notes": notes,
        "provisional": True,
        "human_reviewed": False,
        "final_support_label": None,
        "abstained": label == "uncertain",
        "abstention_reason": "heuristic_uncertain" if label == "uncertain" else ""
    }


def main():
    parser = argparse.ArgumentParser(description="Batch RGB-only provisional annotation")
    parser.add_argument("--selection", type=Path,
                       default=Path("results/sunrgbd_phase1_development20_selection.json"))
    parser.add_argument("--data-root", type=Path,
                       default=Path("data/sunrgbd_official/phase1_development20"))
    parser.add_argument("--output", type=Path,
                       default=Path("results/phase1_development20_first_pass_annotations.json"))
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing annotations")
    args = parser.parse_args()
    
    # Load selection
    selection = json.loads(args.selection.read_text())
    scenes = selection["scenes"]
    
    print(f"Annotating {len(scenes)} scenes with RGB-only heuristics")
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output}")
    
    annotations = []
    for i, scene in enumerate(scenes):
        print(f"\rProgress: {i+1}/{len(scenes)} - {scene['scene_id']}", end="", flush=True)
        annotation = heuristic_annotate_scene(scene, args.data_root)
        if annotation:
            annotations.append(annotation)
    
    print(f"\n\nGenerated {len(annotations)} provisional annotations")
    
    # Save
    output_data = {
        "split": "development",
        "annotation_pass": "first_pass_rgb_only_heuristic",
        "instruction": "Provisional RGB-only annotations generated by heuristics. All labels require human review. Do not view depth.npy, reference.json, or 3D geometry during first pass.",
        "scenes": annotations
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"Saved to {args.output}")
    
    # Summary
    label_counts = {}
    for a in annotations:
        label = a.get("rgb_first_pass_support_label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
