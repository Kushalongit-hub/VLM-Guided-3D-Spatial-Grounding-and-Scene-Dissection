"""Evaluate predicted object centers against annotated metric centers."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.evaluation import evaluate_centroids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_centroids(
        json.loads(args.prediction.read_text(encoding="utf-8")),
        json.loads(args.reference.read_text(encoding="utf-8")),
    )
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
