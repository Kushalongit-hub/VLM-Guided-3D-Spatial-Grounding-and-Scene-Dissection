"""Fetch selected calibrated SUN RGB-D scenes from the official ZIP by HTTP range."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import zlib

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.sunrgbd import convert_labels, decode_depth_png, read_calibration
from src.utils.remote_zip import HTTPRangeReader


ARCHIVE_URL = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
SCENE_INDICES = (0, 2, 3)  # Zero-based index in the official MATLAB metadata.


def archive_member(path: str) -> str:
    start = path.find("SUNRGBD/")
    if start < 0:
        raise ValueError(f"Not a SUN RGB-D archive path: {path}")
    return "/".join(part for part in path[start:].split("/") if part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/sunrgbd_official"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sunrgbd_official/subset"))
    parser.add_argument("--selection", type=Path,
                        help="Frozen selection JSON from select_sunrgbd_subset.py")
    args = parser.parse_args()
    meta3_path = args.metadata_dir / "SUNRGBDMeta3DBB_v2.mat"
    meta2_path = args.metadata_dir / "SUNRGBDMeta2DBB_v2.mat"
    indices = SCENE_INDICES
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        for path in (meta3_path, meta2_path):
            if hashlib.sha256(path.read_bytes()).hexdigest() != selection["source_sha256"][path.name]:
                raise ValueError(f"Metadata checksum differs from frozen selection: {path}")
        indices = tuple(int(scene["metadata_index"]) for scene in selection["scenes"])
        if len(indices) != len(set(indices)):
            raise ValueError("Selection contains duplicate metadata indices")
    meta3 = loadmat(meta3_path, squeeze_me=True, struct_as_record=False)["SUNRGBDMeta"]
    meta2 = loadmat(meta2_path, squeeze_me=True, struct_as_record=False)["SUNRGBDMeta2DBB"]
    reader = HTTPRangeReader(ARCHIVE_URL)
    scenes, provenance = [], []
    with zipfile.ZipFile(reader) as archive:
        for index in indices:
            three, two = meta3[index], meta2[index]
            if three.sequenceName != two.sequenceName or not three.valid:
                raise ValueError(f"Metadata mismatch at scene {index}")
            scene_id = f"sunrgbd_{index + 1:06d}"
            scene_dir = args.output_dir / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            files = {}
            for key, original in (("image", three.rgbpath), ("depth", three.depthpath)):
                member = archive_member(str(original))
                suffix = Path(member).suffix
                target = scene_dir / f"{key}{suffix}"
                info = archive.getinfo(member)
                content = target.read_bytes() if target.is_file() else b""
                if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                    content = archive.read(member)
                    target.write_bytes(content)
                files[key] = {"member": member, "sha256": hashlib.sha256(content).hexdigest()}
            calib_path = scene_dir / "calib.txt"
            calib_path.write_text(
                " ".join(map(str, np.asarray(three.Rtilt).flatten(order="F"))) + "\n" +
                " ".join(map(str, np.asarray(three.K).flatten(order="F"))) + "\n",
                encoding="utf-8")
            labels = []
            boxes3 = np.atleast_1d(three.groundtruth3DBB)
            boxes2 = np.atleast_1d(two.groundtruth2DBB)
            paired2 = {}
            for obj2 in boxes2:
                if bool(obj2.has3dbox):
                    paired2.setdefault(str(obj2.classname), []).append(obj2)
            class_counts3 = {}
            for obj3 in boxes3:
                class_counts3[str(obj3.classname)] = class_counts3.get(str(obj3.classname), 0) + 1
            for obj3 in boxes3:
                name = str(obj3.classname)
                candidates = paired2.get(name, [])
                if class_counts3[name] != 1 or len(candidates) != 1:
                    continue
                obj2 = candidates[0]
                fields = [str(obj3.classname), *np.asarray(obj2.gtBb2D).reshape(-1),
                          *np.asarray(obj3.centroid).reshape(-1),
                          *np.abs(np.asarray(obj3.coeffs).reshape(-1)),
                          *np.asarray(obj3.orientation).reshape(-1)[:2]]
                labels.append(" ".join(map(str, fields)))
            if not labels:
                raise ValueError(f"No paired 2D/3D objects in {scene_id}")
            label_path = scene_dir / "label.txt"
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
            depth = decode_depth_png(scene_dir / f"depth{Path(str(three.depthpath)).suffix}")
            rtilt, intrinsics = read_calibration(calib_path)
            objects, reference = convert_labels(label_path, rtilt, depth.shape)
            np.save(scene_dir / "depth.npy", depth)
            (scene_dir / "objects.json").write_text(json.dumps(objects, indent=2), encoding="utf-8")
            (scene_dir / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
            scenes.append({"scene_id": scene_id, "depth": f"{scene_id}/depth.npy",
                           "objects": f"{scene_id}/objects.json",
                           "reference": f"{scene_id}/reference.json",
                           "intrinsics": intrinsics})
            provenance.append({"scene_id": scene_id, "metadata_index": index,
                               "sequence_name": str(three.sequenceName),
                               "paired_objects": len(objects), "files": files})
            print(f"Prepared {scene_id}: {len(objects)} paired objects")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps({"scenes": scenes}, indent=2), encoding="utf-8")
    (args.output_dir / "source_manifest.json").write_text(json.dumps({
        "archive_url": ARCHIVE_URL, "archive_bytes": reader.length,
        "metadata_sha256": {meta3_path.name: hashlib.sha256(meta3_path.read_bytes()).hexdigest(),
                            meta2_path.name: hashlib.sha256(meta2_path.read_bytes()).hexdigest()},
        "scenes": provenance}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
