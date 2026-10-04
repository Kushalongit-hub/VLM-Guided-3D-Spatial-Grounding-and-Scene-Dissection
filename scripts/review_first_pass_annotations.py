"""Second-pass programmatic review of first-pass annotations.

This script verifies that the stored annotations are consistent with the
RGB-only heuristics applied to the current box data. It does NOT view images.

Usage:
    .\.venv\Scripts\python.exe scripts/review_first_pass_annotations.py
"""

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


def heuristic_label(iou, child_above, child_within):
    """Apply the same heuristic as batch_annotate_first_pass.py"""
    if iou > 0.1 and child_above and child_within:
        return "supported_by", "medium", "high overlap, child above parent, within bounds"
    elif iou > 0.3 and child_above:
        return "supported_by", "medium", "moderate overlap, child above parent"
    elif child_within and child_above:
        return "supported_by", "low", "child within parent bounds and above, but low overlap"
    elif iou > 0.1:
        return "uncertain", "low", "some overlap but unclear support"
    else:
        return "none", "low", "no significant overlap"


def review_annotations():
    """Review all annotations for consistency."""
    annotations_path = Path("results/phase1_development20_first_pass_annotations.json")
    data_root = Path("data/sunrgbd_official/phase1_development20")
    
    data = json.loads(annotations_path.read_text())
    scenes = data["scenes"]
    
    print("Reviewing first-pass annotations...")
    print("=" * 60)
    
    corrected = 0
    verified = 0
    issues = []
    
    for scene in scenes:
        scene_id = scene["scene_id"]
        objects_path = data_root / scene_id / "objects.json"
        
        try:
            objects = json.loads(objects_path.read_text())
        except Exception as e:
            issues.append(f"{scene_id}: Cannot load objects: {e}")
            continue
        
        # Find child and parent
        child_class = scene["candidate_child_class"]
        parent_class = scene["candidate_parent_class"]
        child_objs = [obj for obj in objects if obj["label"] == child_class]
        parent_objs = [obj for obj in objects if obj["label"] == parent_class]
        
        if not child_objs or not parent_objs:
            issues.append(f"{scene_id}: Missing child or parent objects")
            continue
        
        child_box = child_objs[0]["bbox_2d"]
        parent_box = parent_objs[0]["bbox_2d"]
        
        iou = compute_iou(child_box, parent_box)
        child_above = is_child_above_parent(child_box, parent_box)
        child_within = is_child_within_parent(child_box, parent_box)
        
        expected_label, expected_conf, expected_note = heuristic_label(iou, child_above, child_within)
        
        stored_label = scene.get("rgb_first_pass_support_label")
        stored_conf = scene.get("rgb_first_pass_confidence")
        stored_notes = scene.get("rgb_first_pass_notes", "")
        
        # Check consistency
        if stored_label != expected_label:
            issues.append(f"{scene_id}: Label mismatch. Stored={stored_label}, Expected={expected_label}, IoU={iou:.2f}, above={child_above}, within={child_within}")
            # Auto-correct
            scene["rgb_first_pass_support_label"] = expected_label
            scene["rgb_first_pass_confidence"] = expected_conf
            scene["rgb_first_pass_notes"] = f"Corrected: {expected_note} (IoU={iou:.2f})"
            scene["abstained"] = expected_label == "uncertain"
            scene["abstention_reason"] = "heuristic_uncertain" if expected_label == "uncertain" else ""
            corrected += 1
        else:
            verified += 1
    
    # Save corrected annotations
    if corrected > 0:
        annotations_path.write_text(json.dumps(data, indent=2))
        print(f"Corrected {corrected} annotations")
    
    print(f"Verified {verified} annotations")
    print(f"\nIssues found: {len(issues)}")
    for issue in issues:
        print(f"  - {issue}")
    
    # Summary
    label_counts = {}
    for scene in scenes:
        label = scene.get("rgb_first_pass_support_label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    print(f"\nTotal scenes: {len(scenes)}")
    print(f"Verified: {verified}")
    print(f"Corrected: {corrected}")
    
    return data


if __name__ == "__main__":
    review_annotations()
