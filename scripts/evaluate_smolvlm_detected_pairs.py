"""Post-hoc VLM sensitivity to matched detector boxes versus oracle boxes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.vlm_support import PROMPT, judge_support_pair, load_vlm, overlay_pair


PROTOCOL = ROOT / "results/smolvlm_detected_pair_protocol.json"
CHECKPOINT = ROOT / "checkpoints/smolvlm2-2.2b-instruct"
OUTPUT = ROOT / "results/smolvlm_detected_pair9.json"
SETS = (("laptop", "laptop_table6"), ("bottle", "bottle_table6"))


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    prompt_path = ROOT / "results/smolvlm_relation_protocol.json"
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != protocol["oracle_prompt_protocol_sha256"]:
        raise ValueError("Original prompt protocol changed")
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))["prompt"]
    if prompt != PROMPT:
        raise ValueError("Prompt implementation differs from frozen text")
    oracle = {row["scene_id"]: row for row in json.loads(
        (ROOT / "results/smolvlm_laptop_bottle_relation12.json").read_text())["scenes"]}
    processor, model, _ = load_vlm(CHECKPOINT, "cuda")
    rows, missing = [], []
    for child_class, name in SETS:
        report_path = ROOT / "results" / f"sunrgbd_{name}_transfer.json"
        if hashlib.sha256(report_path.read_bytes()).hexdigest() != protocol["detector_reports_sha256"][name]:
            raise ValueError(f"Frozen detector report changed: {name}")
        transfer = json.loads(report_path.read_text(encoding="utf-8"))
        for scene in transfer["scenes"]:
            scene_id = scene["scene_id"]
            child_match, parent_match = scene["matches"][child_class], scene["matches"]["table"]
            if not (child_match and parent_match):
                missing.append({"scene_id": scene_id, "child_class": child_class,
                                "child_matched": bool(child_match), "parent_matched": bool(parent_match)})
                continue
            detections = {obj["instance_id"]: obj for obj in scene["detections"]}
            child = detections[child_match["instance_id"]]
            parent = detections[parent_match["instance_id"]]
            image_path = ROOT / "data/sunrgbd_official" / name / scene_id / "image.jpg"
            with Image.open(image_path) as source:
                marked = overlay_pair(source, child["bbox_2d"], parent["bbox_2d"])
            marked_path = ROOT / "outputs/vlm_detected_pair_marked" / f"{scene_id}.png"
            marked_path.parent.mkdir(parents=True, exist_ok=True)
            marked.save(marked_path)
            judgment = judge_support_pair(marked, processor, model, prompt=prompt, max_new_tokens=12)
            row = {"scene_id": scene_id, "child_class": child_class,
                   "rgb_relation": scene["visible_relation"],
                   "detector_child_iou": child_match["box_iou"],
                   "detector_parent_iou": parent_match["box_iou"],
                   "oracle_box_answer": oracle[scene_id]["parsed_answer"],
                   "marked_image": str(marked_path.relative_to(ROOT)), **judgment}
            rows.append(row)
            OUTPUT.write_text(json.dumps({"status": "partial", "matched": rows,
                                          "unmatched": missing}, indent=2) + "\n", encoding="utf-8")
            print(f"{scene_id}: {row['oracle_box_answer']} -> {row['parsed_answer']}", flush=True)
    result = {"status": "complete", "setting": "post-hoc matched detector box VLM judgment",
              "matched_pair_count": len(rows), "unmatched_pair_count": len(missing),
              "changed_answer_count": sum(row["oracle_box_answer"] != row["parsed_answer"] for row in rows),
              "matched": rows, "unmatched": missing}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"matched", "unmatched"}}, indent=2))


if __name__ == "__main__":
    main()
