"""Frozen RGB-to-3D laptop/table category transfer evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.child_ray_support import correct_child_ray_support_layout
from src.pipeline.depth_estimator import infer_metric_depth
from src.pipeline.detection_evaluation import box_iou
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.grounding_dino_detector import detect_grounded_objects, load_grounding_dino
from src.pipeline.mask_support_gate import gate_support_proposals
from src.pipeline.object_detector import suppress_duplicate_masks
from src.pipeline.sam_segmenter import load_sam, segment_detector_boxes
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/sunrgbd_official/laptop_table6"
SELECTION = ROOT / "results/sunrgbd_laptop_table6_selection.json"
LABELS = ROOT / "results/sunrgbd_laptop_table6_rgb_labels.json"
OUTPUT = ROOT / "results/sunrgbd_laptop_table6_transfer.json"


def matched_detection(oracle: dict, detections: list[dict], label: str):
    candidates = [(box_iou(oracle["bbox_2d"], det["bbox_2d"]), det)
                  for det in detections if det["label"] == label]
    if not candidates:
        return None
    iou, detection = max(candidates, key=lambda item: item[0])
    return {"instance_id": detection["instance_id"], "box_iou": float(iou)} if iou >= 0.5 else None


def proposals(layout: dict) -> list[dict]:
    """Frozen box-foot/depth-gap rule generalized to laptop/table labels."""
    output = []
    for child in layout["objects"]:
        if child["object"] != "laptop":
            continue
        x0, _, x1, bottom = child["bbox_2d"]
        foot_x = (x0 + x1) / 2
        possible = []
        for parent in layout["objects"]:
            if parent["object"] != "table":
                continue
            left, top, right, lower = parent["bbox_2d"]
            if not (left <= foot_x < right and top <= bottom <= lower):
                continue
            gap = abs(child["center_point_3d"][2] - parent["center_point_3d"][2])
            if gap <= 0.75:
                possible.append((gap, parent["instance_id"]))
        if possible:
            gap, parent_id = min(possible)
            output.append({"child_id": child["instance_id"], "parent_id": parent_id,
                           "depth_gap_m": float(gap)})
    return output


def center_error(layout: dict, instance_id: str, truth: np.ndarray) -> float:
    center = next(np.asarray(obj["center_point_3d"], dtype=float) for obj in layout["objects"]
                  if obj["instance_id"] == instance_id)
    return float(np.linalg.norm(center - truth))


def main() -> None:
    import torch

    selected = json.loads(SELECTION.read_text(encoding="utf-8"))["scenes"]
    labels = {row["scene_id"]: row["supported_by_paired_table"] for row in
              json.loads(LABELS.read_text(encoding="utf-8"))["scenes"]}
    if set(labels) != {row["scene_id"] for row in selected}:
        raise ValueError("Selection and frozen RGB labels differ")
    detector_processor, detector, device = load_grounding_dino(ROOT / "checkpoints/grounding-dino-tiny", "cuda")
    detected = {}
    for scene in selected:
        scene_id = scene["scene_id"]
        with Image.open(DATA / scene_id / "image.jpg") as raw:
            detected[scene_id] = detect_grounded_objects(
                raw.convert("RGB"), detector_processor, detector, labels=("laptop", "table"),
                box_threshold=0.25, text_threshold=0.25)
        print(f"{scene_id}: raw detections={len(detected[scene_id])}", flush=True)
    del detector
    torch.cuda.empty_cache()
    processor, sam, _ = load_sam(ROOT / "checkpoints/sam-vit-base", "cuda")
    rows = []
    for scene in selected:
        scene_id = scene["scene_id"]
        folder = DATA / scene_id
        objects = detected[scene_id]
        if objects:
            with Image.open(folder / "image.jpg") as raw:
                masks, sam_scores = segment_detector_boxes(raw.convert("RGB"), objects, processor, sam)
            for obj in objects:
                x0, y0, x1, y1 = obj["bbox_2d"]
                if not masks[obj["instance_id"]][y0:y1, x0:x1].any():
                    masks[obj["instance_id"]][y0:y1, x0:x1] = True
            objects, masks = suppress_duplicate_masks(objects, masks)
        else:
            masks, sam_scores = {}, {}
        row = {"scene_id": scene_id, "visible_relation": labels[scene_id],
               "detections": objects, "sam_iou_estimates": sam_scores}
        rows.append((row, masks))
    del sam
    torch.cuda.empty_cache()
    results = []
    for row, masks in rows:
        scene_id = row["scene_id"]
        folder = DATA / scene_id
        manifest = next(item for item in json.loads((DATA / "manifest.json").read_text())["scenes"]
                        if item["scene_id"] == scene_id)
        depth, _ = infer_metric_depth(folder / "image.jpg",
                                      ROOT / "checkpoints/depth_anything_v2_metric_hypersim_vits.pth", "cuda")
        camera = Intrinsics(**manifest["intrinsics"])
        box_layout = build_layout(depth, row["detections"], camera)
        mask_layout = build_layout(depth, row["detections"], camera, masks=masks)
        before = proposals(mask_layout)
        after, evidence = gate_support_proposals(before, masks)
        oracle = json.loads((folder / "objects.json").read_text())
        reference = {obj["instance_id"]: np.asarray(obj["center_point_3d"], dtype=float)
                     for obj in json.loads((folder / "reference.json").read_text())["objects"]}
        matches = {}
        errors = {}
        for label in ("laptop", "table"):
            gt = next(obj for obj in oracle if obj["label"] == label)
            match = matched_detection(gt, row["detections"], label)
            matches[label] = match
            if match:
                errors[label] = {"box_m": center_error(box_layout, match["instance_id"], reference[gt["instance_id"]]),
                                 "mask_m": center_error(mask_layout, match["instance_id"], reference[gt["instance_id"]])}
        if len(after) == 1:
            rtilt, _ = read_calibration(folder / "calib.txt")
            pair = after[0]
            corrected = correct_child_ray_support_layout(
                mask_layout, pair["child_id"], pair["parent_id"], camera_up_from_rtilt(rtilt),
                offset_m=0.4, child_sigma_m=0.4, relation_sigma_m=0.2, max_shift_m=0.5)
            row["correction"] = corrected["child_ray_support_correction"]
            if matches["laptop"] and matches["laptop"]["instance_id"] == pair["child_id"]:
                truth_id = next(obj["instance_id"] for obj in oracle if obj["label"] == "laptop")
                errors["laptop"]["corrected_mask_m"] = center_error(
                    corrected, pair["child_id"], reference[truth_id])
        else:
            row["correction_status"] = "abstained_no_unique_contact_proposal"
        row.update({"matches": matches, "errors": errors, "proposals_before_gate": before,
                    "proposals_after_gate": after, "contact_evidence": evidence})
        if matches["laptop"] and matches["table"]:
            target = (matches["laptop"]["instance_id"], matches["table"]["instance_id"])
            row["paired_candidate_before_gate"] = any(
                (p["child_id"], p["parent_id"]) == target for p in before)
            row["paired_candidate_after_gate"] = any(
                (p["child_id"], p["parent_id"]) == target for p in after)
        else:
            row["paired_candidate_before_gate"] = False
            row["paired_candidate_after_gate"] = False
        results.append(row)
        print(f"{scene_id}: matched={bool(matches['laptop'] and matches['table'])} "
              f"pair={row['paired_candidate_before_gate']}/{row['paired_candidate_after_gate']}", flush=True)
    summary = {
        "scene_count": len(results),
        "visible_positive_count": sum(r["visible_relation"] is True for r in results),
        "visible_negative_count": sum(r["visible_relation"] is False for r in results),
        "paired_detection_coverage": sum(bool(r["matches"]["laptop"] and r["matches"]["table"]) for r in results),
        "positive_pair_candidates_before_gate": sum(r["visible_relation"] is True and r["paired_candidate_before_gate"] for r in results),
        "positive_pair_candidates_after_gate": sum(r["visible_relation"] is True and r["paired_candidate_after_gate"] for r in results),
        "negative_pair_candidates_before_gate": sum(r["visible_relation"] is False and r["paired_candidate_before_gate"] for r in results),
        "negative_pair_candidates_after_gate": sum(r["visible_relation"] is False and r["paired_candidate_after_gate"] for r in results),
        "negative_scenes_with_any_proposal_before_gate": sum(
            r["visible_relation"] is False and bool(r["proposals_before_gate"]) for r in results),
        "negative_scenes_with_any_proposal_after_gate": sum(
            r["visible_relation"] is False and bool(r["proposals_after_gate"]) for r in results),
        "corrected_matched_laptop_count": sum("corrected_mask_m" in r["errors"].get("laptop", {}) for r in results),
        "positive_matched_laptop_error_mean_m": {
            method: float(np.mean([r["errors"]["laptop"][method] for r in results
                                   if r["visible_relation"] is True
                                   and method in r["errors"].get("laptop", {})]))
            for method in ("box_m", "mask_m", "corrected_mask_m")},
        "positive_matched_table_error_mean_m": {
            method: float(np.mean([r["errors"]["table"][method] for r in results
                                   if r["visible_relation"] is True
                                   and method in r["errors"].get("table", {})]))
            for method in ("box_m", "mask_m")},
    }
    report = {"setting": "frozen laptop/table category transfer", "summary": summary, "scenes": results}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
