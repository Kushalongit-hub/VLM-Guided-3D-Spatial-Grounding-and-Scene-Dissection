"""Predict metric depth and object-level 3D layout from one RGB image.

Optional geometric support proposals and an advisory local VLM relation check
are available. Full scene-graph reasoning and object orientation remain open.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.depth_estimator import infer_metric_depth
from src.pipeline.child_ray_support import correct_child_ray_support_layout
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.generic_support_proposals import propose_labeled_support_relations
from src.pipeline.grounding_dino_detector import detect_grounded_objects, load_grounding_dino
from src.pipeline.object_detector import detect_objects, segment_objects, suppress_duplicate_masks
from src.pipeline.radial_support import correct_radial_support_layout
from src.pipeline.sam_segmenter import load_sam, segment_detector_boxes
from src.pipeline.mask_support_gate import gate_support_proposals
from src.pipeline.support_constraints import correct_support_layout
from src.pipeline.support_constraints import camera_up_from_rtilt
from src.pipeline.support_proposals import propose_support_relations
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.vlm_support import PROMPT as VLM_SUPPORT_PROMPT
from src.pipeline.vlm_support import gate_vlm_proposals, judge_support_pair, load_vlm, overlay_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    region_source = parser.add_mutually_exclusive_group(required=True)
    region_source.add_argument("--objects", type=Path,
                               help="JSON list with label and pixel bbox_2d")
    region_source.add_argument("--detector-weights", type=Path,
                               help="Local YOLO11n detection weights")
    region_source.add_argument("--segmenter-weights", type=Path,
                               help="Local YOLO11n-seg instance segmentation weights")
    region_source.add_argument("--grounding-dino-checkpoint", type=Path,
                               help="Local Grounding DINO Tiny checkpoint directory")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Official Metric Hypersim Small .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-output", type=Path,
                        help="Optional saved HxW metric depth .npy")
    parser.add_argument("--detections-output", type=Path,
                        help="Optional saved detection/segmentation objects JSON")
    parser.add_argument("--masks-output", type=Path,
                        help="Optional NPZ of masks keyed by instance ID")
    parser.add_argument("--sam-checkpoint", type=Path,
                        help="Local SAM ViT-B checkpoint; segment detector or supplied boxes")
    parser.add_argument("--boxes-preview", type=Path,
                        help="Optional RGB PNG with object boxes")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--grounding-box-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-text-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-labels", nargs="+",
                        help="Grounding DINO object labels; default: cup mug desk table")
    parser.add_argument("--support-child-labels", nargs="+",
                        help="Child labels for configurable support proposals; default: cup mug")
    parser.add_argument("--support-constraints", type=Path,
                        help="JSON with relations, center_sigma_m, and explicit up_camera")
    parser.add_argument("--propose-support", action="store_true",
                        help="Add heuristic cup/mug support candidates; does not correct geometry")
    parser.add_argument("--mask-contact-gate", action="store_true",
                        help="Keep only proposed support pairs with visible child/parent mask adjacency; may miss occluded contacts")
    parser.add_argument("--vlm-checkpoint", type=Path,
                        help="Local SmolVLM2 checkpoint for advisory RGB support judgments")
    parser.add_argument("--vlm-gate", action="store_true",
                        help="Retain only VLM YES proposals; requires --vlm-checkpoint and proposal mode")
    radial = parser.add_mutually_exclusive_group()
    radial.add_argument("--radial-support-pair", nargs=2, metavar=("CHILD_ID", "PARENT_ID"),
                        help="Apply the frozen camera-ray correction to one supplied support pair")
    radial.add_argument("--radial-proposed-support", action="store_true",
                        help="Correct the single proposed support pair; requires exactly one proposal")
    radial_up = parser.add_mutually_exclusive_group()
    radial_up.add_argument("--up-camera", type=float, nargs=3, metavar=("X", "Y", "Z"),
                           help="Camera-frame up direction for radial correction")
    radial_up.add_argument("--sunrgbd-calibration", type=Path,
                           help="SUN RGB-D calib.txt used to derive camera-frame up")
    parser.add_argument("--support-method", choices=("paired-radial", "child-ray-soft"),
                        default="paired-radial",
                        help="Camera-ray correction; child-ray-soft is a post-hoc research alternative")
    parser.add_argument("--profile-output", type=Path,
                        help="Save stage wall time and CUDA peak allocated/reserved memory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=518)
    for key in ("fx", "fy", "cx", "cy"):
        parser.add_argument(f"--{key}", type=float, required=True)
    args = parser.parse_args()
    if (args.radial_support_pair or args.radial_proposed_support) and not (args.up_camera or args.sunrgbd_calibration):
        parser.error("Radial correction requires --up-camera or --sunrgbd-calibration")
    if args.support_constraints and (args.radial_support_pair or args.radial_proposed_support):
        parser.error("Choose either --support-constraints or radial correction")
    if args.support_method != "paired-radial" and not (args.radial_support_pair or args.radial_proposed_support):
        parser.error("--support-method requires a radial support pair or proposal")
    if args.sam_checkpoint and args.segmenter_weights:
        parser.error("--sam-checkpoint cannot be combined with --segmenter-weights")
    if args.grounding_labels and not args.grounding_dino_checkpoint:
        parser.error("--grounding-labels requires --grounding-dino-checkpoint")
    if args.support_child_labels and not (args.propose_support or args.radial_proposed_support):
        parser.error("--support-child-labels requires --propose-support or --radial-proposed-support")
    for labels, name in ((args.grounding_labels, "--grounding-labels"),
                         (args.support_child_labels, "--support-child-labels")):
        if labels and len(labels) != len(set(labels)):
            parser.error(f"{name} contains duplicate labels")
    if args.mask_contact_gate and not (args.propose_support or args.radial_proposed_support):
        parser.error("--mask-contact-gate requires --propose-support or --radial-proposed-support")
    if args.mask_contact_gate and not (args.sam_checkpoint or args.segmenter_weights):
        parser.error("--mask-contact-gate requires SAM or segmentation masks")
    if args.vlm_checkpoint and not (args.propose_support or args.radial_proposed_support):
        parser.error("--vlm-checkpoint requires --propose-support or --radial-proposed-support")
    if args.vlm_gate and not args.vlm_checkpoint:
        parser.error("--vlm-gate requires --vlm-checkpoint")
    if args.vlm_gate and args.radial_support_pair:
        parser.error("--vlm-gate only filters automatically proposed support pairs")
    profile = {"stages": {}}
    def stage_done(name: str, started: float) -> None:
        cuda = None
        if args.device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    cuda = torch
            except ImportError:
                pass
        profile["stages"][name] = {"wall_seconds": perf_counter() - started}
        if cuda is not None:
            profile["stages"][name]["cuda_peak_allocated_gb"] = cuda.cuda.max_memory_allocated() / 1e9
            profile["stages"][name]["cuda_peak_reserved_gb"] = cuda.cuda.max_memory_reserved() / 1e9
            cuda.cuda.reset_peak_memory_stats()
    if args.profile_output and args.device != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
    total_started = perf_counter()
    stage_started = perf_counter()
    masks = None
    if args.objects:
        objects = json.loads(args.objects.read_text(encoding="utf-8"))
        if not isinstance(objects, list):
            raise ValueError("Objects JSON must be a list")
    elif args.detector_weights:
        objects = detect_objects(args.image, args.detector_weights,
                                 args.confidence, args.device)
    elif args.grounding_dino_checkpoint:
        from PIL import Image
        processor, model, grounding_device = load_grounding_dino(
            args.grounding_dino_checkpoint, args.device)
        with Image.open(args.image) as source:
            objects = detect_grounded_objects(source.convert("RGB"), processor, model,
                                               labels=tuple(args.grounding_labels) if args.grounding_labels else ("cup", "mug", "desk", "table"),
                                               box_threshold=args.grounding_box_threshold,
                                               text_threshold=args.grounding_text_threshold)
        del model
        if grounding_device == "cuda":
            import torch
            torch.cuda.empty_cache()
    else:
        objects, masks = segment_objects(args.image, args.segmenter_weights,
                                         args.confidence, args.device)
    sam_scores = None
    sam_fallback_ids = []
    sam_suppressed_ids = []
    if args.sam_checkpoint and objects:
        from PIL import Image
        sam_processor, sam_model, sam_device = load_sam(args.sam_checkpoint, args.device)
        with Image.open(args.image) as source:
            masks, sam_scores = segment_detector_boxes(
                source.convert("RGB"), objects, sam_processor, sam_model)
        for obj in objects:
            xmin, ymin, xmax, ymax = obj["bbox_2d"]
            mask = masks[obj["instance_id"]]
            if not mask[ymin:ymax, xmin:xmax].any():
                mask[ymin:ymax, xmin:xmax] = True
                sam_fallback_ids.append(obj["instance_id"])
        prior_ids = {obj["instance_id"] for obj in objects}
        objects, masks = suppress_duplicate_masks(objects, masks)
        kept_ids = {obj["instance_id"] for obj in objects}
        sam_suppressed_ids = sorted(prior_ids - kept_ids)
        sam_scores = {key: score for key, score in sam_scores.items() if key in kept_ids}
        sam_fallback_ids = [key for key in sam_fallback_ids if key in kept_ids]
        del sam_model
        if sam_device == "cuda":
            import torch
            torch.cuda.empty_cache()
    stage_done("object_regions", stage_started)
    if args.detections_output:
        args.detections_output.parent.mkdir(parents=True, exist_ok=True)
        args.detections_output.write_text(json.dumps(objects, indent=2), encoding="utf-8")
    if args.masks_output:
        if masks is None:
            parser.error("--masks-output requires --segmenter-weights or --sam-checkpoint with objects")
        args.masks_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.masks_output, **masks)
    if args.boxes_preview:
        overlay = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if overlay is None:
            raise ValueError(f"Cannot read RGB image: {args.image}")
        if masks is not None:
            for mask in masks.values():
                overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array([0, 220, 0])).astype(np.uint8)
        for obj in objects:
            xmin, ymin, xmax, ymax = obj["bbox_2d"]
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
            caption = obj["label"]
            if "confidence" in obj:
                caption += f" {obj['confidence']:.2f}"
            cv2.putText(overlay, caption, (xmin, max(20, ymin - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        args.boxes_preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.boxes_preview), overlay):
            raise OSError(f"Could not write boxes preview: {args.boxes_preview}")
    camera = Intrinsics(args.fx, args.fy, args.cx, args.cy)
    stage_started = perf_counter()
    depth, device = infer_metric_depth(args.image, args.checkpoint,
                                       args.device, args.input_size)
    stage_done("metric_depth", stage_started)
    stage_started = perf_counter()
    layout = build_layout(depth, objects, camera, masks=masks)
    layout["source_image"] = str(args.image)
    layout["depth_model"] = "Depth-Anything-V2-Metric-Hypersim-Small"
    layout["depth_device"] = device
    layout["object_regions_source"] = ("external_json" if args.objects else
                                        "grounding_dino_tiny" if args.grounding_dino_checkpoint else
                                        "yolo11n_seg_coco" if args.segmenter_weights else
                                        "yolo11n_coco")
    if args.sam_checkpoint:
        layout["segmentation_model"] = "SAM-ViT-B" if objects else "not_run_no_objects"
        layout["sam_mask_iou_estimates"] = sam_scores or {}
        layout["sam_empty_mask_box_fallback_ids"] = sam_fallback_ids
        layout["sam_duplicate_suppressed_ids"] = sam_suppressed_ids
    if args.propose_support or args.radial_proposed_support:
        if args.support_child_labels:
            layout["proposed_support_relations"] = propose_labeled_support_relations(
                layout, set(args.support_child_labels))
            layout["support_proposal_rule"] = {
                "child_labels": args.support_child_labels,
                "parent_labels": "table_desk_counter_surfaces",
                "footpoint_inside_parent_box": True,
                "max_median_depth_gap_m": 0.75,
            }
        else:
            layout["proposed_support_relations"] = propose_support_relations(layout)
            layout["support_proposal_rule"] = "cup_or_mug_box_foot_inside_surface_box_and_depth_gap_at_most_0.75_m"
        if args.mask_contact_gate:
            layout["proposed_support_relations_before_mask_gate"] = layout["proposed_support_relations"]
            layout["proposed_support_relations"], layout["mask_contact_evidence"] = gate_support_proposals(
                layout["proposed_support_relations"], masks or {})
            layout["mask_contact_gate_rule"] = "visible_lower_contour_adjacency_fraction_gt_0_radius_8px"
    if args.vlm_checkpoint:
        stage_done("geometry_and_proposals", stage_started)
        source_path = args.vlm_checkpoint / "source.json"
        if source_path.is_file():
            layout["vlm_checkpoint_source"] = json.loads(source_path.read_text(encoding="utf-8"))
        layout["vlm_model"] = layout.get("vlm_checkpoint_source", {}).get(
            "repo_id", args.vlm_checkpoint.name)
        layout["vlm_support_prompt"] = VLM_SUPPORT_PROMPT
        layout["vlm_support_judgments"] = []
        layout["vlm_gate_enabled"] = bool(args.vlm_gate)
        if layout["proposed_support_relations"]:
            layout["vlm_stage_status"] = "inferred"
            stage_started = perf_counter()
            from PIL import Image
            import torch
            if device == "cuda":
                torch.cuda.empty_cache()
            processor, vlm_model, vlm_device = load_vlm(args.vlm_checkpoint, args.device)
            by_id = {obj["instance_id"]: obj for obj in layout["objects"]}
            with Image.open(args.image) as raw_image:
                source_image = raw_image.convert("RGB")
                for proposal in layout["proposed_support_relations"]:
                    marked = overlay_pair(source_image,
                                          by_id[proposal["child_id"]]["bbox_2d"],
                                          by_id[proposal["parent_id"]]["bbox_2d"])
                    judgment = judge_support_pair(marked, processor, vlm_model)
                    layout["vlm_support_judgments"].append({
                        "child_id": proposal["child_id"], "parent_id": proposal["parent_id"],
                        **judgment})
            del vlm_model
            if vlm_device == "cuda":
                torch.cuda.empty_cache()
            stage_done("vlm_relation_check", stage_started)
        else:
            layout["vlm_stage_status"] = "not_run_no_proposals"
        stage_started = perf_counter()
        if args.vlm_gate:
            layout["proposed_support_relations_before_vlm_gate"] = layout["proposed_support_relations"]
            layout["proposed_support_relations"] = gate_vlm_proposals(
                layout["proposed_support_relations"], layout["vlm_support_judgments"])
    apply_radial = bool(args.radial_support_pair or args.radial_proposed_support)
    if (args.radial_proposed_support and (args.mask_contact_gate or args.vlm_gate)
            and len(layout["proposed_support_relations"]) != 1):
        layout["support_correction_status"] = (
            "abstained_no_unique_vlm_accepted_proposal" if args.vlm_gate else
            "abstained_no_unique_visible_contact_proposal")
        apply_radial = False
    if apply_radial:
        if args.radial_proposed_support:
            proposals = layout["proposed_support_relations"]
            if len(proposals) != 1:
                raise ValueError(f"Radial correction requires exactly one proposal; found {len(proposals)}. Use --propose-support to inspect candidates and --radial-support-pair to select one.")
            child_id, parent_id = proposals[0]["child_id"], proposals[0]["parent_id"]
            relation_source = "image_depth_heuristic"
        else:
            child_id, parent_id = args.radial_support_pair
            relation_source = "supplied_pair"
        if args.sunrgbd_calibration:
            rtilt, _ = read_calibration(args.sunrgbd_calibration)
            up = camera_up_from_rtilt(rtilt)
        else:
            up = np.asarray(args.up_camera, dtype=float)
        if args.support_method == "paired-radial":
            layout = correct_radial_support_layout(
                layout, child_id, parent_id, up, offset_m=0.4, child_sigma_m=0.4,
                parent_sigma_m=0.2, relation_sigma_m=0.2, max_shift_m=0.5)
            key = "radial_support_correction"
        else:
            layout = correct_child_ray_support_layout(
                layout, child_id, parent_id, up, offset_m=0.4, child_sigma_m=0.4,
                relation_sigma_m=0.2, max_shift_m=0.5)
            key = "child_ray_support_correction"
        layout[key]["relation_source"] = relation_source
        layout[key]["up_camera"] = np.asarray(up, dtype=float).tolist()
    if args.support_constraints:
        support = json.loads(args.support_constraints.read_text(encoding="utf-8"))
        if "up_camera" not in support:
            raise ValueError("Support input requires up_camera in camera coordinates")
        layout = correct_support_layout(layout, support["relations"],
                                        support["center_sigma_m"],
                                        up=np.asarray(support["up_camera"], dtype=float))
        layout["support_relations_source"] = str(args.support_constraints)
    stage_done("support_correction" if args.vlm_checkpoint else "geometry_and_support", stage_started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    if args.depth_output:
        args.depth_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.depth_output, depth)
    profile["total_wall_seconds"] = perf_counter() - total_started
    profile["selected_device"] = device
    if args.profile_output:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote {len(layout['objects'])} objects to {args.output} using {device}")


if __name__ == "__main__":
    main()
