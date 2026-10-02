"""Exploratory SAM-mask effects on previously scored DINO support pairs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.child_ray_support import correct_child_ray_support_layout
from src.pipeline.detection_evaluation import box_iou
from src.pipeline.geometry import Intrinsics, build_layout
from src.pipeline.mask_contact import visible_contact_fraction
from src.pipeline.sam_segmenter import load_sam, segment_detector_boxes
from src.pipeline.sunrgbd import read_calibration
from src.pipeline.support_constraints import camera_up_from_rtilt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest = {row["scene_id"]: row for row in json.loads(
        (args.root / "manifest.json").read_text(encoding="utf-8"))["scenes"]}
    annotations = {row["scene_id"]: row for row in json.loads(
        args.labels.read_text(encoding="utf-8"))["scenes"]}
    detected = {row["scene_id"]: row["detections"] for row in json.loads(
        args.detections.read_text(encoding="utf-8"))["scenes"]}
    if manifest.keys() != annotations.keys() or manifest.keys() != detected.keys():
        raise ValueError("Scene IDs differ among inputs")
    processor, sam, device = load_sam(args.sam_checkpoint, args.device)
    rows = []
    for scene_id, scene in manifest.items():
        annotation = annotations[scene_id]
        objects = json.loads((args.root / scene["objects"]).read_text(encoding="utf-8"))
        oracle = {row["instance_id"]: row for row in objects}
        pair = {}
        for role, reference_id, allowed in (
                ("child", annotation["child_id"], {"cup", "mug"}),
                ("parent", annotation["parent_id"], {"table", "desk"})):
            candidates = [(box_iou(oracle[reference_id]["bbox_2d"], det["bbox_2d"]), det)
                          for det in detected[scene_id] if det["label"] in allowed]
            if candidates:
                iou, best = max(candidates, key=lambda item: item[0])
                if iou >= 0.5:
                    pair[role] = {"object": best, "box_iou": iou}
        row = {"scene_id": scene_id, "relation_label": annotation["supported_by_table"],
               "child_matched": "child" in pair, "parent_matched": "parent" in pair}
        if pair:
            selected = [item["object"] for item in pair.values()]
            with Image.open(args.root / scene_id / "image.jpg") as source:
                masks, scores = segment_detector_boxes(source.convert("RGB"), selected, processor, sam)
            depth = np.load(args.root / scene_id / "predicted_depth.npy", allow_pickle=False)
            camera = Intrinsics(**scene["intrinsics"])
            reference = {obj["instance_id"]: np.asarray(obj["center_point_3d"])
                         for obj in json.loads((args.root / scene["reference"]).read_text(encoding="utf-8"))["objects"]}
            result = {}
            for role, item in pair.items():
                obj = item["object"]
                instance_id = obj["instance_id"]
                xmin, ymin, xmax, ymax = obj["bbox_2d"]
                if not masks[instance_id][ymin:ymax, xmin:xmax].any():
                    result[role] = {"box_iou": item["box_iou"], "sam_iou_estimate": scores[instance_id],
                                    "empty_mask": True}
                    continue
                box_layout = build_layout(depth, [obj], camera)
                mask_layout = build_layout(depth, [obj], camera, masks={instance_id: masks[instance_id]})
                truth = reference[annotation[f"{role}_id"]]
                box_error = float(np.linalg.norm(np.asarray(box_layout["objects"][0]["center_point_3d"]) - truth))
                mask_error = float(np.linalg.norm(np.asarray(mask_layout["objects"][0]["center_point_3d"]) - truth))
                result[role] = {"box_iou": item["box_iou"], "sam_iou_estimate": scores[instance_id],
                                "empty_mask": False, "box_error_m": box_error,
                                "mask_error_m": mask_error, "mask_minus_box_error_m": mask_error - box_error}
            row["regions"] = result
            if len(pair) == 2 and not any(item["empty_mask"] for item in result.values()):
                row["mask_contact"] = visible_contact_fraction(
                    masks[pair["child"]["object"]["instance_id"]],
                    masks[pair["parent"]["object"]["instance_id"]])
            if (row["relation_label"] is True and len(pair) == 2 and
                    not any(item["empty_mask"] for item in result.values())):
                child_id = pair["child"]["object"]["instance_id"]
                parent_id = pair["parent"]["object"]["instance_id"]
                rtilt, _ = read_calibration(args.root / scene_id / "calib.txt")
                up = camera_up_from_rtilt(rtilt)
                pair_objects = [pair["child"]["object"], pair["parent"]["object"]]
                layouts = {"box": build_layout(depth, pair_objects, camera),
                           "mask": build_layout(depth, pair_objects, camera, masks=masks)}
                child_truth = reference[annotation["child_id"]]
                row["support_child_error_m"] = {}
                for region_type, layout in layouts.items():
                    corrected = correct_child_ray_support_layout(
                        layout, child_id, parent_id, up, offset_m=0.4,
                        child_sigma_m=0.4, relation_sigma_m=0.2, max_shift_m=0.5)
                    for method, output in (("uncorrected", layout), ("child_ray_soft", corrected)):
                        child_center = next(np.asarray(obj["center_point_3d"]) for obj in output["objects"]
                                            if obj["instance_id"] == child_id)
                        row["support_child_error_m"][f"{region_type}_{method}"] = float(
                            np.linalg.norm(child_center - child_truth))
        rows.append(row)
        print(f"{scene_id}: child={row['child_matched']} parent={row['parent_matched']}", flush=True)
    regions = [value for row in rows for value in row.get("regions", {}).values()
               if not value["empty_mask"]]
    positive_pairs = [row for row in rows if "support_child_error_m" in row]
    support_means = {key: float(np.mean([row["support_child_error_m"][key] for row in positive_pairs]))
                     for key in positive_pairs[0]["support_child_error_m"]} if positive_pairs else {}
    report = {
        "setting": "post-hoc SAM masks on frozen Grounding DINO detections for previously scored support pairs",
        "scene_count": len(rows), "matched_region_count": len(regions),
        "child_region_count": sum("child" in row.get("regions", {}) for row in rows),
        "parent_region_count": sum("parent" in row.get("regions", {}) for row in rows),
        "mean_mask_minus_box_error_m": float(np.mean([x["mask_minus_box_error_m"] for x in regions]))
        if regions else None,
        "child_mean_mask_minus_box_error_m": float(np.mean([
            row["regions"]["child"]["mask_minus_box_error_m"] for row in rows
            if "child" in row.get("regions", {}) and not row["regions"]["child"]["empty_mask"]])),
        "parent_mean_mask_minus_box_error_m": float(np.mean([
            row["regions"]["parent"]["mask_minus_box_error_m"] for row in rows
            if "parent" in row.get("regions", {}) and not row["regions"]["parent"]["empty_mask"]])),
        "positive_paired_count": len(positive_pairs), "positive_paired_support_child_error_m": support_means,
        "note": "Same previously saved DINO boxes, predicted depth and oracle matching for both arms. Previously inspected scenes; no independent test claim.",
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
