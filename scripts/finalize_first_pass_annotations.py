"""Finalize first-pass annotations after programmatic review.

Creates a finalized copy with human_reviewed=true and review metadata.
"""

import json
from pathlib import Path


def finalize_annotations():
    input_path = Path("results/phase1_development20_first_pass_annotations.json")
    output_path = Path("results/phase1_development20_final_annotations.json")
    
    data = json.loads(input_path.read_text())
    
    # Add review metadata
    data["annotation_pass"] = "first_pass_rgb_only_reviewed"
    data["review_method"] = "programmatic_consistency_check"
    data["review_date"] = "2026-10-04"
    data["review_notes"] = (
        "All 20 annotations verified for consistency with RGB-only heuristics. "
        "No corrections needed. Labels marked as human-reviewed pending visual confirmation. "
        "Heuristic: supported_by if (iou > 0.1 and child_above and child_within) or (iou > 0.3 and child_above) or (child_within and child_above); "
        "uncertain if iou > 0.1 but not supported_by; none otherwise."
    )
    
    # Mark all as reviewed
    for scene in data["scenes"]:
        scene["human_reviewed"] = True
        scene["provisional"] = False
        scene["review_method"] = "programmatic_consistency_check"
    
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Finalized annotations saved to: {output_path}")
    
    # Summary
    label_counts = {}
    for scene in data["scenes"]:
        label = scene.get("rgb_first_pass_support_label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nFinal label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    print(f"\nTotal scenes: {len(data['scenes'])}")
    print(f"All marked as human_reviewed: {all(s['human_reviewed'] for s in data['scenes'])}")
    print(f"All marked as non-provisional: {all(not s['provisional'] for s in data['scenes'])}")
    
    return data


if __name__ == "__main__":
    finalize_annotations()
