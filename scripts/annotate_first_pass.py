"""Interactive RGB-only annotation tool for Phase 1 development scenes.

Usage:
    .\.venv\Scripts\python.exe scripts/annotate_first_pass.py

Instructions:
    1. Run this script to annotate each development scene.
    2. For each scene, view the RGB image and decide:
       - 'y': laptop/desk IS supported by desk/laptop (visible contact)
       - 'n': laptop/desk is NOT supported by desk/laptop
       - 'u': uncertain / ambiguous (mark as uncertain)
    3. The script saves annotations incrementally to a JSON file.
    4. Do NOT view depth.npy, reference.json, or 3D geometry during annotation.
"""

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def load_image(image_path: Path):
    """Load image using PIL or OpenCV."""
    if HAS_PIL:
        return Image.open(image_path).convert("RGB")
    elif HAS_CV2:
        img = cv2.imread(str(image_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
    return None


def annotate_scene(scene: dict, data_root: Path, output_path: Path) -> dict:
    """Annotate a single scene from RGB only."""
    scene_id = scene["scene_id"]
    child_class = scene["candidate_child_class"]
    parent_class = scene["candidate_parent_class"]
    
    image_path = data_root / scene_id / "image.jpg"
    objects_path = data_root / scene_id / "objects.json"
    
    print(f"\n{'='*60}")
    print(f"Scene: {scene_id}")
    print(f"Child: {child_class}, Parent: {parent_class}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")
    
    # Load and display image with bounding boxes
    image = load_image(image_path)
    if image is None:
        print(f"ERROR: Cannot load image {image_path}")
        return None
    
    # Load objects for bounding box visualization
    try:
        objects = json.loads(objects_path.read_text())
        target_classes = {child_class, parent_class}
        
        if HAS_PIL:
            draw = ImageDraw.Draw(image)
            for obj in objects:
                if obj["label"] not in target_classes:
                    continue
                x0, y0, x1, y1 = obj["bbox_2d"]
                color = "red" if obj["label"] == child_class else "blue"
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                draw.text((x0 + 3, y0 + 3), f"{obj['label']} {obj['instance_id']}", 
                         fill=color, stroke_width=2, stroke_fill="white")
        
        print(f"Objects found: {len(objects)}")
        for obj in objects:
            if obj["label"] in target_classes:
                print(f"  {obj['label']} ({obj['instance_id']}): {obj['bbox_2d']}")
    except Exception as e:
        print(f"Warning: Cannot load objects: {e}")
    
    # Try to display image
    try:
        if HAS_PIL:
            image.show()
            print("Image displayed. Please inspect the RGB image.")
        elif HAS_CV2:
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Scene: {scene_id}", img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Note: Cannot display image interactively ({e})")
        print(f"  Open manually: {image_path}")
    
    # Get annotation from user
    print(f"\nQuestion: Is the {child_class} physically supported by the {parent_class}?")
    print("  'y' = supported (visible contact)")
    print("  'n' = not supported")
    print("  'u' = uncertain/ambiguous")
    print("  's' = skip this scene")
    
    while True:
        response = input("Your annotation [y/n/u/s]: ").strip().lower()
        if response in ('y', 'n', 'u', 's'):
            break
        print("Invalid input. Please enter y, n, u, or s.")
    
    if response == 's':
        print(f"Skipped {scene_id}")
        return None
    
    # Map response to label
    label_map = {
        'y': 'supported_by',
        'n': 'none',
        'u': 'uncertain'
    }
    
    confidence_map = {
        'y': 'high',
        'n': 'high',
        'u': 'low'
    }
    
    annotation = {
        "scene_id": scene_id,
        "metadata_index": scene["metadata_index"],
        "candidate_child_class": child_class,
        "candidate_parent_class": parent_class,
        "rgb_first_pass_support_label": label_map[response],
        "rgb_first_pass_confidence": confidence_map[response],
        "rgb_first_pass_notes": "",
        "provisional": True,
        "human_reviewed": False,
        "final_support_label": None,
        "abstained": response == 'u',
        "abstention_reason": "uncertain_rgb_only" if response == 'u' else ""
    }
    
    print(f"Recorded: {label_map[response]}")
    return annotation


def main():
    parser = argparse.ArgumentParser(description="RGB-only first-pass annotation tool")
    parser.add_argument("--selection", type=Path, 
                       default=Path("results/sunrgbd_phase1_development20_selection.json"))
    parser.add_argument("--data-root", type=Path, 
                       default=Path("data/sunrgbd_official/phase1_development20"))
    parser.add_argument("--output", type=Path,
                       default=Path("results/phase1_development20_first_pass_annotations.json"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--scene-id", type=str, default=None,
                       help="Annotate only this scene")
    args = parser.parse_args()
    
    # Load selection
    selection = json.loads(args.selection.read_text())
    scenes = selection["scenes"]
    
    # Load existing annotations if output exists
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        annotated_ids = {s["scene_id"] for s in existing["scenes"] if s.get("rgb_first_pass_support_label")}
        annotations = existing["scenes"]
        print(f"Loaded {len(annotated_ids)} existing annotations")
    else:
        annotated_ids = set()
        annotations = []
    
    # Determine which scenes to annotate
    if args.scene_id:
        scenes_to_annotate = [s for s in scenes if s["scene_id"] == args.scene_id]
    else:
        scenes_to_annotate = scenes[args.start_index:]
    
    print(f"Annotating {len(scenes_to_annotate)} scenes from RGB only")
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output}")
    
    for scene in scenes_to_annotate:
        if scene["scene_id"] in annotated_ids:
            print(f"\nSkipping {scene['scene_id']} (already annotated)")
            continue
        
        annotation = annotate_scene(scene, args.data_root, args.output)
        
        if annotation is None:
            continue
        
        # Update or append annotation
        existing_idx = next((i for i, a in enumerate(annotations) 
                            if a["scene_id"] == scene["scene_id"]), None)
        if existing_idx is not None:
            annotations[existing_idx] = annotation
        else:
            annotations.append(annotation)
        
        # Save incrementally
        output_data = {
            "split": "development",
            "annotation_pass": "first_pass_rgb_only",
            "instruction": "Record support labels from RGB image only. Do not view depth.npy, reference.json, or 3D geometry. Mark uncertain cases as 'uncertain'. All labels are provisional pending human review.",
            "scenes": annotations
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"Saved to {args.output}")
    
    # Summary
    labeled = [a for a in annotations if a.get("rgb_first_pass_support_label")]
    print(f"\n{'='*60}")
    print(f"Annotation complete!")
    print(f"Total scenes: {len(scenes)}")
    print(f"Labeled: {len(labeled)}")
    print(f"Uncertain: {sum(1 for a in labeled if a['rgb_first_pass_support_label'] == 'uncertain')}")
    print(f"Supported: {sum(1 for a in labeled if a['rgb_first_pass_support_label'] == 'supported_by')}")
    print(f"Not supported: {sum(1 for a in labeled if a['rgb_first_pass_support_label'] == 'none')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
