"""Exploratory VLM support-pair judgments from RGB and paired 2D boxes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.vlm_support import PROMPT, judge_support_pair, overlay_pair, parse_answer

PROTOCOL = ROOT / "results/smolvlm_relation_protocol.json"
CHECKPOINT = ROOT / "checkpoints/smolvlm2-2.2b-instruct"
OUTPUT = ROOT / "results/smolvlm_laptop_bottle_relation12.json"
SETS = (("laptop", "laptop_table6"), ("bottle", "bottle_table6"))


def main() -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["prompt"] != PROMPT:
        raise ValueError("Frozen VLM prompt differs from implementation")
    for _, name in SETS:
        labels_path = ROOT / "results" / f"sunrgbd_{name}_rgb_labels.json"
        digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        if digest != protocol["labels_sha256"][name]:
            raise ValueError(f"Frozen RGB labels changed: {name}")
    source = json.loads((CHECKPOINT / "source.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for child_class, name in SETS:
        data = ROOT / "data/sunrgbd_official" / name
        labels = json.loads((ROOT / "results" / f"sunrgbd_{name}_rgb_labels.json").read_text())
        for annotation in labels["scenes"]:
            scene_id = annotation["scene_id"]
            folder = data / scene_id
            objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
            child = next(obj for obj in objects if obj["label"] == child_class)
            parent = next(obj for obj in objects if obj["label"] == "table")
            with Image.open(folder / "image.jpg") as source_image:
                marked = overlay_pair(source_image, child["bbox_2d"], parent["bbox_2d"])
            marked_path = ROOT / "outputs/vlm_relation_marked" / f"{scene_id}.png"
            marked_path.parent.mkdir(parents=True, exist_ok=True)
            marked.save(marked_path)
            start = perf_counter()
            judgment = judge_support_pair(marked, processor, model,
                                          prompt=protocol["prompt"], max_new_tokens=12)
            raw = judgment["raw_answer"]
            row = {"scene_id": scene_id, "child_class": child_class,
                   "rgb_relation": annotation["supported_by_paired_table"],
                   "marked_image": str(marked_path.relative_to(ROOT)),
                   **judgment,
                   "generation_wall_seconds": perf_counter() - start}
            rows.append(row)
            OUTPUT.write_text(json.dumps({"status": "partial", "model": source,
                                          "scenes": rows}, indent=2) + "\n", encoding="utf-8")
            print(f"{scene_id}: {raw!r} => {row['parsed_answer']}", flush=True)
    report = {"status": "complete", "setting": "exploratory oracle-box VLM relation check",
              "model": source, "protocol": str(PROTOCOL.relative_to(ROOT)),
              "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
              "positive_count": sum(row["rgb_relation"] is True for row in rows),
              "negative_count": sum(row["rgb_relation"] is False for row in rows),
              "answer_counts": {answer: sum(row["parsed_answer"] == answer for row in rows)
                                for answer in ("YES", "NO", "UNCERTAIN", "UNPARSEABLE")},
              "scenes": rows}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
