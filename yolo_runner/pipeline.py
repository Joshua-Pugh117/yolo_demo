from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

TASKS = ("detect", "segment", "pose")


def model_weights(task: str, size: str) -> str:
    suffix = {"detect": "", "segment": "-seg", "pose": "-pose"}[task]
    return f"yolo26{size}{suffix}.pt"


def find_images(photos_dir: Path) -> list[Path]:
    return sorted(
        p for p in photos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_pipeline(
    photos_dir: Path,
    output_dir: Path,
    *,
    model_size: str = "s",
) -> int:
    """Run detect, segment, and pose on every image in photos_dir."""
    photos_dir = photos_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(photos_dir)
    if not images:
        print(f"No images found in {photos_dir}")
        print(f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        return 1

    models = {task: YOLO(model_weights(task, model_size)) for task in TASKS}

    print(f"Processing {len(images)} image(s) with YOLO26-{model_size}\n")

    for image in images:
        print(f"  {image.name}")

        for task, model in models.items():
            task_dir = output_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            out_path = task_dir / image.name

            results = model.predict(source=str(image), verbose=False)
            show_boxes = task == "detect"
            annotated = results[0].plot(boxes=show_boxes)
            cv2.imwrite(str(out_path), annotated)
            print(f"    {task:8} -> {out_path}")

    print(f"\nDone. Results saved to {output_dir}")
    return 0