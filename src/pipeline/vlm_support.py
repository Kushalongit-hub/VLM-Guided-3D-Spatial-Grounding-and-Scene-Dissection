"""Optional local VLM judgment of a marked child/support image pair."""

from __future__ import annotations

from pathlib import Path
import re

from PIL import Image, ImageDraw


PROMPT = (
    "In this image, the red box marks object A and the green box marks object B. "
    "Is A physically resting on B? Answer exactly YES, NO, or UNCERTAIN. "
    "'On' means A is supported by B, not just nearby or visually overlapping. "
    "Ignore other surfaces."
)


def overlay_pair(image: Image.Image, child_box: list[int], parent_box: list[int]) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for box, color, letter in ((child_box, "red", "A"), (parent_box, "lime", "B")):
        draw.rectangle(tuple(box), outline=color, width=5)
        draw.text((box[0] + 3, max(0, box[1] - 16)), letter, fill=color, stroke_width=1,
                  stroke_fill="black")
    return result


def parse_answer(raw: str) -> str:
    match = re.match(r"^[\s\W]*(YES|NO|UNCERTAIN)\b", raw, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNPARSEABLE"


def gate_vlm_proposals(proposals: list[dict], judgments: list[dict]) -> list[dict]:
    """Keep only explicit YES judgments, requiring one answer per proposal."""
    expected = [(str(row["child_id"]), str(row["parent_id"])) for row in proposals]
    actual = [(str(row["child_id"]), str(row["parent_id"])) for row in judgments]
    if len(expected) != len(set(expected)) or len(actual) != len(set(actual)):
        raise ValueError("Duplicate support pair in proposal or judgment")
    if set(expected) != set(actual):
        raise ValueError("VLM judgments must match proposed support pairs")
    by_pair = {pair: row["parsed_answer"] for pair, row in zip(actual, judgments)}
    if any(answer not in {"YES", "NO", "UNCERTAIN", "UNPARSEABLE"}
           for answer in by_pair.values()):
        raise ValueError("Unexpected VLM answer")
    return [row for row, pair in zip(proposals, expected) if by_pair[pair] == "YES"]


def load_vlm(checkpoint: Path, device: str = "auto"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"VLM checkpoint missing: {checkpoint}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, local_files_only=True, dtype=dtype,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    return processor, model, device


def judge_support_pair(marked_image: Image.Image, processor, model,
                       prompt: str = PROMPT, max_new_tokens: int = 12) -> dict:
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    message = [{"role": "user", "content": [
        {"type": "image", "image": marked_image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        message, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    generated = output[0][inputs["input_ids"].shape[-1]:]
    raw = processor.decode(generated, skip_special_tokens=True).strip()
    return {"raw_answer": raw, "parsed_answer": parse_answer(raw)}
