from __future__ import annotations

import argparse
from pathlib import Path

from yolo_runner.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO26 detect, segment, and pose on photos.",
    )
    parser.add_argument(
        "--photos",
        type=Path,
        default=Path("photos"),
        help="Folder containing input images (default: photos)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Folder for annotated outputs (default: results)",
    )
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default="s",
        help="YOLO26 model size: n=nano, s=small, m=medium, l=large, x=extra-large (default: s)",
    )
    args = parser.parse_args()

    raise SystemExit(
        run_pipeline(args.photos, args.output, model_size=args.model_size)
    )


if __name__ == "__main__":
    main()