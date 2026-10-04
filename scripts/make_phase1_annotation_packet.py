"""Build an RGB-only development annotation sheet and candidate ID inventory."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("results/sunrgbd_phase1_development20_selection.json"))
    parser.add_argument("--data", type=Path, default=Path("data/sunrgbd_official/phase1_development20"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1_development20_rgb_sheet.png"))
    parser.add_argument("--inventory", type=Path, default=Path("results/phase1_development20_annotation_inventory.json"))
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    panels, inventory = [], []
    for row in selection["scenes"]:
        scene_id = row["scene_id"]
        folder = args.data / scene_id
        image_path = folder / "image.jpg"
        objects_path = folder / "objects.json"
        if not image_path.is_file() or not objects_path.is_file():
            raise FileNotFoundError(f"Scene is incomplete: {folder}")
        image = Image.open(image_path).convert("RGB")
        objects = json.loads(objects_path.read_text(encoding="utf-8"))
        candidates = [obj for obj in objects if obj["label"] in
                      {row["candidate_child_class"], row["candidate_parent_class"]}]
        inventory.append({"scene_id": scene_id, "metadata_index": row["metadata_index"],
                          "candidate_child_class": row["candidate_child_class"],
                          "candidate_parent_class": row["candidate_parent_class"],
                          "candidates": candidates})
        canvas = Image.new("RGB", (image.width, image.height + 32), "white")
        canvas.paste(image, (0, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 8), scene_id, fill="black")
        for obj in candidates:
            x0, y0, x1, y1 = obj["bbox_2d"]
            color = "red" if obj["label"] == row["candidate_child_class"] else "lime"
            draw.rectangle((x0, y0 + 32, x1, y1 + 32), outline=color, width=3)
            draw.text((x0 + 3, y0 + 35), f'{obj["instance_id"]} {obj["label"]}', fill=color)
        panels.append(canvas)
    tile_width, tile_height = 480, 400
    sheet = Image.new("RGB", (tile_width * 4, tile_height * 5), "white")
    for i, panel in enumerate(panels):
        panel.thumbnail((tile_width - 8, tile_height - 8))
        sheet.paste(panel, ((i % 4) * tile_width + 4, (i // 4) * tile_height + 4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    args.inventory.write_text(json.dumps({"split": "development", "scenes": inventory}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(panels)} RGB panels to {args.output}")


if __name__ == "__main__":
    main()
