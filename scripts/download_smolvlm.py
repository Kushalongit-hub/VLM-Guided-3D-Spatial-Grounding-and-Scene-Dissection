"""Download and verify a pinned local SmolVLM2 checkpoint."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download
import requests


REPO = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "checkpoints/smolvlm2-2.2b-instruct"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    info = HfApi().model_info(REPO, files_metadata=True)
    snapshot_download(
        repo_id=REPO, revision=info.sha, local_dir=DEST,
        allow_patterns=["*.json", "*.txt", "*.model"],
        max_workers=2,
    )
    for sibling in info.siblings:
        if not sibling.rfilename.endswith(".safetensors"):
            continue
        expected_sha = sibling.lfs.sha256
        expected_size = sibling.size
        target = DEST / sibling.rfilename
        part = target.with_suffix(target.suffix + ".part")
        if target.is_file():
            if target.stat().st_size == expected_size and sha256_file(target) == expected_sha:
                continue
            raise ValueError(f"Existing checkpoint shard is invalid: {target}")
        url = f"https://huggingface.co/{REPO}/resolve/{info.sha}/{sibling.rfilename}"
        for attempt in range(12):
            position = part.stat().st_size if part.is_file() else 0
            headers = {"Range": f"bytes={position}-"} if position else {}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if position and response.status_code != 206:
                        raise RuntimeError("Server did not honor the resume range")
                    with part.open("ab" if position else "wb") as output:
                        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if chunk:
                                output.write(chunk)
                                if output.tell() // (256 * 1024 * 1024) > position // (256 * 1024 * 1024):
                                    position = output.tell()
                                    print(f"{sibling.rfilename}: {position / 1e9:.2f}/{expected_size / 1e9:.2f} GB", flush=True)
                if part.stat().st_size == expected_size:
                    break
                raise RuntimeError("Incomplete response")
            except (requests.RequestException, RuntimeError) as error:
                if attempt == 11:
                    raise RuntimeError(f"Could not fetch {sibling.rfilename}") from error
                time.sleep(min(2 ** attempt, 15))
        if sha256_file(part) != expected_sha:
            raise ValueError(f"Checksum mismatch for {sibling.rfilename}")
        os.replace(part, target)
    (DEST / "source.json").write_text(json.dumps(
        {"repo_id": REPO, "revision": info.sha,
         "weight_sha256": {item.rfilename: item.lfs.sha256 for item in info.siblings
                           if item.rfilename.endswith(".safetensors")}}, indent=2) + "\n", encoding="utf-8")
    print(f"Cached {REPO}@{info.sha} in {DEST}")


if __name__ == "__main__":
    main()
