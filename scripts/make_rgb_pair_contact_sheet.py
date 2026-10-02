"""Create RGB-only paired-object contact sheets for blinded visual annotation."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    scenes = json.loads(args.selection.read_text(encoding="utf-8"))["scenes"]
    canvas_width, canvas_height = 1280, 2040
    for page in range((len(scenes) + 7) // 8):
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        for local_index, selected in enumerate(scenes[page * 8:(page + 1) * 8]):
            scene_id = selected["scene_id"]
            image = Image.open(args.root / scene_id / "image.jpg").convert("RGB")
            objects = json.loads((args.root / scene_id / "objects.json").read_text(encoding="utf-8"))
            target_classes = {selected["candidate_child_class"], selected["candidate_parent_class"]}
            draw = ImageDraw.Draw(image)
            for obj in objects:
                if obj["label"] not in target_classes:
                    continue
                color = "red" if obj["label"] in {"cup", "mug"} else "blue"
                box = obj["bbox_2d"]
                draw.rectangle(box, outline=color, width=4)
                draw.text((box[0], max(0, box[1] - 16)),
                          f"{obj['label']} {obj['instance_id']}", fill=color,
                          stroke_width=2, stroke_fill="white")
            image.thumbnail((630, 490))
            x = (local_index % 2) * 640
            y = (local_index // 2) * 510
            canvas.paste(image, (x, y + 20))
            ImageDraw.Draw(canvas).text((x + 5, y + 2), scene_id, fill="black")
        output = args.output_prefix.with_name(f"{args.output_prefix.name}_{page + 1}.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        print(output)


if __name__ == "__main__":
    main()
