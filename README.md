# YOLO Runner

Run [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) on every image in a folder — object detection, instance segmentation, and pose estimation — and save annotated results ready to share.

## Quick start

```bash
# 1. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Drop your images into photos/
# 4. Run
python run.py
```

Results are written to `results/`:

```
results/
├── detect/   # bounding boxes
├── segment/  # instance masks
└── pose/     # keypoints
```

## Usage

```bash
python run.py                          # default: photos/ → results/
python run.py --photos my_images       # custom input folder
python run.py --output my_results      # custom output folder
python run.py --model-size m           # larger model (n/s/m/l/x)
```

Or as a module / CLI entry point:

```bash
python -m yolo_runner
yolo-run
```

## Requirements

- Python 3.10+
- Images in JPG, PNG, WebP, BMP, or TIFF format

Models are downloaded automatically on first run.

## License

MIT