"""Tune contact-surface and support-gate thresholds on the development set.

This script runs the full pipeline on all 20 development scenes and sweeps
min_contact_fraction to find the best threshold for relation detection.

Usage:
    .\.venv\Scripts\python.exe scripts/tune_development_thresholds.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time

import numpy as np

from src.pipeline.mask_contact import visible_contact_fraction
from src.pipeline.mask_support_gate import gate_support_proposals
from src.pipeline.generic_support_proposals import propose_labeled_support_relations
from src.pipeline.grounding_dino_detector import detect_grounded_objects, load_grounding_dino
from src.pipeline.sam_segmenter import load_sam, segment_detector_boxes
from src.pipeline.object_detector import suppress_duplicate_masks
from src.pipeline.depth_estimator import infer_metric_depth
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.sunrgbd import read_calibration
from PIL import Image


def compute_metrics(predicted_labels, reference_labels):
    """Compute precision, recall, F1 for relation detection."""
    tp = sum(1 for p, r in zip(predicted_labels, reference_labels)
             if p == "supported_by" and r == "supported_by")
    fp = sum(1 for p, r in zip(predicted_labels, reference_labels)
             if p == "supported_by" and r != "supported_by")
    fn = sum(1 for p, r in zip(predicted_labels, reference_labels)
             if p != "supported_by" and r == "supported_by")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def process_scene(scene_id, image_path, calib_path, objects_path, annotations, 
                  grounding_checkpoint, sam_checkpoint, depth_checkpoint, device="cpu"):
    """Process a single scene and return contact evidence."""
    print(f"  Processing {scene_id}...")
    start = time.time()
    
    # Load calibration
    rtilt, intrinsics = read_calibration(calib_path)
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    
    # Load objects
    objects = json.loads(objects_path.read_text())
    
    # Run Grounding DINO
    processor, model, grounding_device = load_grounding_dino(grounding_checkpoint, device)
    with Image.open(image_path) as source:
        objects = detect_grounded_objects(
            source.convert("RGB"), processor, model,
            labels=("laptop", "desk"),
            box_threshold=0.25,
            text_threshold=0.25
        )
    del model
    if grounding_device == "cuda":
        import torch
        torch.cuda.empty_cache()
    
    # Run SAM
    sam_processor, sam_model, sam_device = load_sam(sam_checkpoint, device)
    with Image.open(image_path) as source:
        masks, sam_scores = segment_detector_boxes(
            source.convert("RGB"), objects, sam_processor, sam_model)
    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox_2d"]
        mask = masks[obj["instance_id"]]
        if not mask[ymin:ymax, xmin:xmax].any():
            mask[ymin:ymax, xmin:xmax] = True
    objects, masks = suppress_duplicate_masks(objects, masks)
    del sam_model
    if sam_device == "cuda":
        import torch
        torch.cuda.empty_cache()
    
    # Run depth estimation
    depth, depth_device = infer_metric_depth(image_path, depth_checkpoint, device)
    
    # Build layout
    camera = Intrinsics(fx, fy, cx, cy)
    layout = build_layout(depth, objects, camera, masks=masks)
    
    # Propose support relations
    proposals = propose_labeled_support_relations(layout, {"laptop"})
    
    # Get reference label
    scene_annot = next(s for s in annotations["scenes"] if s["scene_id"] == scene_id)
    reference_label = scene_annot["rgb_first_pass_support_label"]
    
    # Compute contact evidence for each proposal
    contact_evidence = []
    for proposal in proposals:
        child_id = proposal["child_id"]
        parent_id = proposal["parent_id"]
        if child_id in masks and parent_id in masks:
            contact = visible_contact_fraction(masks[child_id], masks[parent_id])
            contact_evidence.append({
                "child_id": child_id,
                "parent_id": parent_id,
                "contact_fraction": contact["contact_fraction"],
                "contact_columns": contact["contact_columns"],
                "sampled_columns": contact["sampled_columns"]
            })
    
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    
    return {
        "scene_id": scene_id,
        "reference_label": reference_label,
        "contact_evidence": contact_evidence,
        "wall_seconds": elapsed
    }


def tune_thresholds():
    """Tune thresholds on the development set."""
    annotations_path = Path("results/phase1_development20_final_annotations.json")
    data_root = Path("data/sunrgbd_official/phase1_development20")
    cache_dir = Path("outputs/threshold_tuning_cache")
    output_dir = Path("outputs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = json.loads(annotations_path.read_text())
    scenes = annotations["scenes"]
    
    # Checkpoints
    grounding_checkpoint = Path("checkpoints/grounding-dino-tiny")
    sam_checkpoint = Path("checkpoints/sam-vit-base")
    depth_checkpoint = Path("checkpoints/depth_anything_v2_metric_hypersim_vits.pth")
    
    print("Threshold tuning on development set")
    print("=" * 60)
    print(f"Scenes: {len(scenes)}")
    print(f"Device: cpu")
    
    # Run pipeline on all scenes
    print("\nRunning pipeline on all scenes...")
    evidence = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        image_path = data_root / scene_id / "image.jpg"
        calib_path = data_root / scene_id / "calib.txt"
        objects_path = data_root / scene_id / "objects.json"
        
        result = process_scene(
            scene_id, image_path, calib_path, objects_path, annotations,
            grounding_checkpoint, sam_checkpoint, depth_checkpoint, device="cpu"
        )
        evidence.append(result)
    
    # Save raw evidence
    evidence_path = output_dir / "threshold_tuning_evidence.json"
    evidence_path.write_text(json.dumps({
        "scenes": evidence,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }, indent=2))
    print(f"\nEvidence saved to: {evidence_path}")
    
    # Thresholds to sweep
    contact_thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    
    results = []
    
    for threshold in contact_thresholds:
        print(f"\nTesting min_contact_fraction={threshold:.2f}")
        
        predicted_labels = []
        reference_labels = []
        
        for result in evidence:
            reference_labels.append(result["reference_label"])
            
            # Apply threshold
            if result["contact_evidence"]:
                max_contact = max(e["contact_fraction"] for e in result["contact_evidence"])
                predicted = "supported_by" if max_contact > threshold else "none"
            else:
                predicted = "none"
            
            predicted_labels.append(predicted)
        
        metrics = compute_metrics(predicted_labels, reference_labels)
        metrics["threshold"] = threshold
        results.append(metrics)
        
        print(f"  P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")
    
    # Find best threshold
    best = max(results, key=lambda r: r["f1"])
    print(f"\n{'=' * 60}")
    print(f"Best threshold: {best['threshold']:.2f} (F1={best['f1']:.3f})")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall: {best['recall']:.3f}")
    print(f"  TP={best['true_positives']}, FP={best['false_positives']}, FN={best['false_negatives']}")
    
    # Save results
    output_path = output_dir / "threshold_tuning_results.json"
    output_path.write_text(json.dumps({
        "tuning_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "development_set": annotations_path.name,
        "device": "cpu",
        "thresholds_tested": contact_thresholds,
        "results": results,
        "best_threshold": best["threshold"],
        "best_metrics": best
    }, indent=2))
    
    print(f"\nResults saved to: {output_path}")
    
    return best


if __name__ == "__main__":
    tune_thresholds()
