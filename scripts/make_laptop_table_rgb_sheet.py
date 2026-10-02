"""Show RGB and annotated 2D laptop/table boxes without reading depth or 3D."""

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/sunrgbd_official/laptop_table6"
selection = json.loads((ROOT / "results/sunrgbd_laptop_table6_selection.json").read_text())
panels = []
for scene in selection["scenes"]:
    scene_id = scene["scene_id"]
    folder = DATA / scene_id
    image = Image.open(folder / "image.jpg").convert("RGB")
    objects = json.loads((folder / "objects.json").read_text())
    canvas = Image.new("RGB", (image.width, image.height + 35), "white")
    canvas.paste(image, (0, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 8), scene_id, fill="black")
    for obj in objects:
        if obj["label"] not in {"laptop", "table"}:
            continue
        x0, y0, x1, y1 = obj["bbox_2d"]
        color = "red" if obj["label"] == "laptop" else "lime"
        draw.rectangle((x0, y0 + 35, x1, y1 + 35), outline=color, width=3)
        draw.text((x0 + 3, y0 + 38), obj["label"], fill=color)
    panels.append(canvas)
width = max(panel.width for panel in panels)
height = max(panel.height for panel in panels)
sheet = Image.new("RGB", (width * 2, height * 3), "white")
for i, panel in enumerate(panels):
    sheet.paste(panel, ((i % 2) * width, (i // 2) * height))
output = ROOT / "outputs/sunrgbd_laptop_table6_rgb_sheet.png"
output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output)
print(output)
