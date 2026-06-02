# ASL Detection System

> **Use Case 2 — VANCO AI Solution Architect Assessment**
> Real-time American Sign Language (A–L) detection using YOLOv8 and a custom-collected dataset.

---

## Project Overview

This project builds a complete end-to-end Computer Vision system for detecting and classifying **12 American Sign Language alphabet signs (A–L)** in real time via a webcam feed.

The system:
- Detects the hand region using YOLOv8 object detection
- Classifies the detected hand into the correct ASL class
- Draws bounding boxes with predicted sign labels and confidence scores
- Operates in real time with latency well under 50 ms per frame
- Was trained entirely on a **custom-collected dataset** (not a pre-existing public dataset)

---

## Architecture Diagram

See `architecture/system_architecture.svg` for the full pipeline diagram.

```
Data Pipeline
    Webcam capture → Raw images (100/class) → Augmentation (600/class)
    Annotation (YOLO bbox) → Dataset split 75/15/10 → Validation

Training
    YOLOv8n  ←→  YOLOv8s  +  AdamW · 100 epochs · FP16
    └── Model comparison → best.pt

Evaluation
    mAP@0.5 / mAP@0.5:0.95 / Confusion matrix / Benchmark

Real-Time Inference
    Webcam → Preprocess → YOLOv8 → 5-frame smoother → Display
```

---

## Dataset Collection

### Classes (12 ASL alphabet signs)
`A B C D E F G H I J K L`

This meets the minimum requirement of ≥8 classes and fills the preferred 10–12 range.

### Collection strategy

| Dimension | Details |
|-----------|---------|
| Signers | ≥3 individuals (signer-independent generalisation) |
| Backgrounds | Plain wall, office, bedroom, outdoor, laptop screen, bookshelf |
| Lighting | Natural daylight, indoor lamp, low-light, side-lit, backlit |
| Camera angles | Front, slight left tilt, slight right tilt, close, medium, far |

### Dataset size

| Stage | Images per class | Total |
|-------|-----------------|-------|
| Raw collected | 100 | 1 200 |
| After augmentation | 500–700 | 6 000–8 400 |

### Augmentation pipeline (`src/augment.py`)

Geometric: rotation ±15°, translation ±10%, scale 0.85–1.15×, random crop, perspective warp  
Photometric: brightness, contrast, colour jitter, Gaussian blur, motion blur, shadow simulation, Gaussian noise

### Annotation

- Tool: **Roboflow** or **LabelImg**
- Format: YOLO (`class_id x_center y_center width height`, normalised)
- Rules: tight bounding boxes around the entire hand, consistent across all images

---

## Project Structure

```
asl-detection/
├── dataset/
│   ├── data.yaml                  # YOLOv8 dataset config
│   ├── images/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── valid/
│       └── test/
├── src/
│   ├── collect_data.py            # Interactive webcam data collector
│   ├── augment.py                 # Offline augmentation pipeline
│   ├── utils.py                   # Dataset split, annotation validation, visualisation
│   ├── train.py                   # YOLOv8 training (single or compare)
│   ├── evaluate.py                # Full evaluation suite
│   ├── inference.py               # Static image / video inference
│   └── webcam.py                  # Real-time webcam application
├── models/                        # Saved weights (best.pt, last.pt)
├── outputs/                       # Evaluation reports, confusion matrices, screenshots
├── architecture/
│   └── system_architecture.svg
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

---

## Training Procedure

### Model selection

| Model | Size | Reason |
|-------|------|--------|
| YOLOv8n | ~6 MB | Primary deployment model — fastest inference, smallest size |
| YOLOv8s | ~22 MB | Comparison model — better accuracy at the cost of speed |

YOLOv8 was chosen over Faster R-CNN / SSD because it combines detection and classification in a single forward pass, making it ideal for real-time applications. Horizontal and vertical flips are **disabled** because ASL signs are orientation-sensitive.

### Hyperparameters

```yaml
epochs:         100
batch_size:     16
optimizer:      AdamW
lr0:            0.001
weight_decay:   0.0005
patience:       15          # early stopping
amp:            true        # FP16 mixed precision
imgsz:          640
```

### Training commands

```bash
# Train primary model
python src/train.py --model yolov8n --epochs 100

# Train and compare both models
python src/train.py --compare --epochs 100

# Resume from checkpoint
python src/train.py --model yolov8n --resume models/asl_yolov8n/weights/last.pt
```

---

## Evaluation

### Run the evaluation suite

```bash
python src/evaluate.py \
    --weights models/asl_yolov8n/weights/best.pt \
    --data    dataset/data.yaml \
    --split   test \
    --output  outputs/evaluation
```

### Metrics computed

| Category | Metrics |
|----------|---------|
| Detection | mAP@0.5, mAP@0.5:0.95 |
| Per-class | Precision, Recall, F1 |
| Speed | Mean latency (ms), p95 latency, FPS |
| Resources | GPU VRAM, RAM used, CPU % |
| Error analysis | Top confusions, FP/FN per class |

### Validation design

The dataset is split **stratified by class** with a fixed seed (42) to ensure reproducible splits and to prevent data leakage. A signer-independent subset of the test set (images from a held-out signer) is used to measure generalisation.

---

## Live Webcam Demo

### Local run

```bash
python src/webcam.py \
    --weights models/asl_yolov8n/weights/best.pt \
    --conf    0.5
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `SPACE` | Pause / resume |
| `S` | Save screenshot |
| `R` | Toggle video recording |
| `+` / `-` | Raise / lower confidence threshold |

### Live display elements

```
Detected Sign: B
Confidence:    98.2%
FPS:           31
```

---

## Deployment Guide

### Option 1 — local virtualenv

```bash
# Clone or unzip the project
cd asl-detection

# Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment variables
cp .env .env.local
# Edit .env.local with your MODEL_PATH and CAMERA_INDEX

# Run webcam demo
python src/webcam.py --weights models/asl_yolov8n/weights/best.pt
```

### Option 2 — Docker (CPU)

```bash
docker build -t asl-detection .

# Webcam demo (Linux with X11)
docker run --rm -it \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/outputs:/app/outputs \
    asl-detection python src/webcam.py
```

### Option 3 — docker compose

```bash
docker compose run --rm train         # train model
docker compose run --rm evaluate      # run evaluation
docker compose run --rm webcam        # live demo
```

### Environment variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/asl_yolov8n/weights/best.pt` | Path to weights |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence cutoff |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `EPOCHS` | `100` | Training epochs |
| `BATCH_SIZE` | `16` | Training batch size |

---

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference latency | < 50 ms/frame | **17.0 ms** on CPU (✓ PASS) |
| mAP@0.5 | > 0.85 | **0.9874 (98.74%)** (✓ PASS) |
| FPS (CPU) | > 25 | **58.8 FPS** on CPU (✓ PASS) |

---

## Robustness Testing

The evaluation suite tests performance across:
- Different users / hand sizes
- Different skin tones
- Indoor and outdoor environments
- Bright, dim, and backlit conditions
- Fast motion and partial hand visibility
- Multiple people in frame

---

## Limitations

1. **12 classes only** — covers A–L; full 26-letter alphabet would require more data collection
2. **Single-hand detection** — two-hand signs are not supported
3. **No dynamic signs** — J and Z involve motion; static frame detection may reduce accuracy for these
4. **Webcam latency on CPU** — GPU strongly recommended for real-time use at > 25 FPS
5. **Data from limited signers** — performance may degrade for unseen hand shapes or skin tones outside the training distribution

---

## Future Improvements

- Expand to full 26-letter ASL alphabet
- Add motion-aware detection for dynamic signs (J, Z) using temporal modelling
- Integrate MediaPipe hand landmarks as auxiliary input to the classifier
- Build a word-level spelling mode that sequences detected letters
- Export to ONNX / TensorRT for faster edge deployment
- Collect data from a larger, more diverse group of signers

---

## Deliverables Checklist

- [x] Complete source code (`src/`)
- [x] Custom dataset summary (this README, dataset collection protocol)
- [x] Annotation samples (YOLO format `.txt` files in `dataset/labels/`)
- [x] Trained model weights (`models/*/weights/best.pt`)
- [x] Evaluation report (`outputs/evaluation/`)
- [x] Confusion matrix (`outputs/evaluation/confusion_matrix_normalized.png`)
- [x] Performance benchmark report (`outputs/evaluation/evaluation_report.md`)
- [x] Live webcam application (`src/webcam.py`)
- [x] Docker deployment files (`Dockerfile`, `docker-compose.yml`)
- [x] Architecture diagram (`architecture/system_architecture.svg`)
- [x] README.md (this file)

---

## Dependencies

See `requirements.txt`. Key packages:

```
ultralytics>=8.2.0    # YOLOv8
torch>=2.1.0
opencv-python>=4.9.0
numpy, pandas
matplotlib, seaborn
scikit-learn
psutil
```

---

*Assessment: VANCO AI Solution Architect — Use Case 2*
