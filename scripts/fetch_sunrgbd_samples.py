"""Fetch a small, fixed public SUN RGB-D sample for local depth evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET = "wyrx/SUNRGBD_seg"
REVISION = "3eb91204b39054a07891210e6f02918f60f1b397"


def download(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sunrgbd_sample"))
    args = parser.parse_args()
    if args.offset < 0 or not 1 <= args.length <= 100:
        parser.error("offset must be nonnegative and length must be 1..100")
    query = urlencode({"dataset": DATASET, "config": "default", "split": "test",
                       "offset": args.offset, "length": args.length})
    payload = json.loads(download(f"https://datasets-server.huggingface.co/rows?{query}"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in payload["rows"]:
        row_index = entry["row_idx"]
        row = entry["row"]
        record = {"row_idx": row_index}
        for key in ("image", "depth"):
            url = row[key]["src"]
            if f"/--/{REVISION}/--/" not in url:
                raise ValueError("Dataset revision changed; inspect and update the frozen sample")
            suffix = Path(url.split("?", 1)[0]).suffix
            name = f"{row_index:05d}_{key}{suffix}"
            content = download(url)
            (args.output_dir / name).write_bytes(content)
            record[key] = {"file": name, "sha256": hashlib.sha256(content).hexdigest()}
        records.append(record)
        print(f"Downloaded test row {row_index}")
    manifest = {"dataset": DATASET, "revision": REVISION, "split": "test",
                "records": records}
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
