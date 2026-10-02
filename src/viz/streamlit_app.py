"""Inspect a saved RGB-to-3D layout, masks, depth and support correction.

Run from the repository root with:
    streamlit run src/viz/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = REPO / "outputs/grounded_sam_auto_desk_layout.json"
DEFAULT_DEPTH = REPO / "outputs/grounded_sam_auto_desk_depth.npy"
DEFAULT_MASKS = REPO / "outputs/grounded_sam_auto_desk_masks.npz"


def load_scene(layout_path: Path, depth_path: Path | None = None,
               masks_path: Path | None = None) -> tuple[dict, np.ndarray, np.ndarray | None, dict]:
    """Load a saved scene with shape checks; paths come from local user input."""
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    image_path = Path(layout["source_image"])
    if not image_path.is_absolute():
        image_path = REPO / image_path
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not load source image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = None
    if depth_path is not None:
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all():
            raise ValueError("Depth must be a finite HxW array matching the image")
    masks = {}
    if masks_path is not None:
        with np.load(masks_path, allow_pickle=False) as saved:
            for key in saved.files:
                mask = saved[key].astype(bool)
                if mask.shape != rgb.shape[:2]:
                    raise ValueError(f"Mask {key} does not match the image")
                masks[key] = mask
    return layout, rgb, depth, masks


def image_overlay(rgb: np.ndarray, layout: dict, masks: dict) -> np.ndarray:
    """Draw visible object masks/boxes and selected support edges."""
    output = rgb.copy()
    for obj in layout["objects"]:
        mask = masks.get(obj["instance_id"])
        if mask is not None:
            output[mask] = (0.6 * output[mask] + 0.4 * np.array([0, 220, 0])).astype(np.uint8)
    for obj in layout["objects"]:
        xmin, ymin, xmax, ymax = map(int, obj["bbox_2d"])
        cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
        label = f"{obj['object']} {obj['instance_id']}"
        cv2.putText(output, label, (xmin, max(20, ymin - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    correction = layout.get("child_ray_support_correction") or layout.get("radial_support_correction")
    if correction:
        indexed = {obj["instance_id"]: obj for obj in layout["objects"]}
        child = indexed.get(correction["child_id"])
        parent = indexed.get(correction["parent_id"])
        if child and parent:
            centers = []
            for obj in (child, parent):
                x0, y0, x1, y1 = obj["bbox_2d"]
                centers.append((int((x0 + x1) / 2), int((y0 + y1) / 2)))
            cv2.arrowedLine(output, centers[0], centers[1], (255, 70, 70), 2, tipLength=0.08)
    return output


def depth_preview(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Color valid metric depths using scene percentiles for display only."""
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        raise ValueError("Depth contains no valid positive pixels")
    lower, upper = map(float, np.percentile(valid, [2, 98]))
    if upper <= lower:
        upper = lower + 1e-6
    scaled = np.uint8(np.clip((depth - lower) / (upper - lower), 0, 1) * 255)
    bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~(np.isfinite(depth) & (depth > 0))] = 0
    return rgb, lower, upper


def scene_figure(layout: dict, depth: np.ndarray | None = None,
                 max_points: int = 6000, elevation: int = 20, azimuth: int = -70):
    """Plot camera-frame object centers and a decimated visible point cloud."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if depth is not None:
        height, width = depth.shape
        stride = max(1, int(np.ceil(np.sqrt(height * width / max_points))))
        yy, xx = np.mgrid[0:height:stride, 0:width:stride]
        z = depth[::stride, ::stride]
        valid = np.isfinite(z) & (z > 0)
        camera = layout["camera_intrinsics"]
        x = (xx[valid] - camera["cx"]) * z[valid] / camera["fx"]
        y = (yy[valid] - camera["cy"]) * z[valid] / camera["fy"]
        ax.scatter(x, z[valid], -y, c=z[valid], cmap="turbo", s=1, alpha=0.13,
                   depthshade=False, label="visible depth")
    for obj in layout["objects"]:
        x, y, z = obj["center_point_3d"]
        ax.scatter([x], [z], [-y], s=70, label=f"{obj['object']} ({obj['instance_id']})")
        ax.text(x, z, -y, obj["object"], fontsize=8)
        old = obj.get("uncorrected_center_point_3d")
        if old is not None:
            ox, oy, oz = old
            ax.scatter([ox], [oz], [-oy], marker="x", c="black", s=45)
            ax.plot([ox, x], [oz, z], [-oy, -y], c="black", linewidth=1.4)
    ax.set_xlabel("X right (m)")
    ax.set_ylabel("Z forward (m)")
    ax.set_zlabel("Height relative to camera (m)")
    ax.view_init(elev=elevation, azim=azimuth)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="RGB to 3D scene viewer", layout="wide")
    st.title("RGB → 3D scene viewer")
    st.caption("Inspect saved predictions. Distances are camera-frame metres; visible extents are not full object dimensions.")
    layout_text = st.sidebar.text_input("Layout JSON", str(DEFAULT_LAYOUT))
    depth_text = st.sidebar.text_input("Metric depth NPY (optional)",
                                       str(DEFAULT_DEPTH) if DEFAULT_DEPTH.exists() else "")
    masks_text = st.sidebar.text_input("Instance masks NPZ (optional)",
                                       str(DEFAULT_MASKS) if DEFAULT_MASKS.exists() else "")
    elevation = st.sidebar.slider("3D elevation", -10, 80, 20)
    azimuth = st.sidebar.slider("3D azimuth", -180, 180, -70)
    try:
        layout, rgb, depth, masks = load_scene(
            Path(layout_text), Path(depth_text) if depth_text.strip() else None,
            Path(masks_text) if masks_text.strip() else None)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        st.error(str(error))
        st.stop()
    correction = layout.get("child_ray_support_correction") or layout.get("radial_support_correction")
    st.write(f"**Objects:** {len(layout['objects'])} · **Detector:** {layout.get('object_regions_source', 'unknown')} "
             f"· **Depth device:** {layout.get('depth_device', 'unknown')}")
    if correction:
        st.info(f"Support correction: {correction['child_id']} → {correction['parent_id']}; "
                f"residual {correction['residual_m']:+.3f} m. "
                "This is a candidate relation, not confirmed physical contact.")
    left, right = st.columns(2)
    with left:
        st.subheader("Image, masks and boxes")
        st.image(image_overlay(rgb, layout, masks), width="stretch")
    with right:
        st.subheader("Predicted metric depth")
        if depth is None:
            st.info("Provide a depth NPY to view depth and the point cloud.")
        else:
            preview, lower, upper = depth_preview(depth)
            st.image(preview, width="stretch")
            st.caption(f"Display colors span {lower:.2f}–{upper:.2f} m (2nd–98th percentile); the NPY holds actual metres.")
    st.subheader("Camera-frame 3D layout")
    fig = scene_figure(layout, depth, elevation=elevation, azimuth=azimuth)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.subheader("Object data")
    st.dataframe([{"id": obj["instance_id"], "class": obj["object"],
                   "center_x_m": obj["center_point_3d"][0],
                   "center_y_m": obj["center_point_3d"][1],
                   "center_z_m": obj["center_point_3d"][2],
                   "region": obj.get("region_type", "box")}
                  for obj in layout["objects"]], hide_index=True)
    st.download_button("Download layout JSON", json.dumps(layout, indent=2),
                       file_name="layout.json", mime="application/json")


if __name__ == "__main__":
    main()
