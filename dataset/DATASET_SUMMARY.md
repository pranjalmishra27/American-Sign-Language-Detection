# Dataset Summary — ASL Detection System

## Overview

| Property | Value |
|----------|-------|
| Task | ASL alphabet sign detection + classification |
| Classes | 12 (A, B, C, D, E, F, G, H, I, J, K, L) |
| Raw images collected | 1,200 (100 per class) |
| Final dataset after augmentation | ~7,200 (600 per class) |
| Annotation format | YOLO (class_id x_center y_center width height) |
| Split | 75% train / 15% valid / 10% test |

---

## Collection Protocol

### Equipment
- Webcam (1080p or 720p, minimum 30 FPS)
- `src/collect_data.py` — interactive capture tool

### Signer Diversity
| Signer | Gender | Hand Size | Notes |
|--------|--------|-----------|-------|
| Signer 1 | M | Medium | Primary collector |
| Signer 2 | F | Small | Additional diversity |
| Signer 3 | M | Large | Additional diversity |

### Background Diversity
| Background | Conditions |
|------------|------------|
| Plain white wall | Neutral, controlled |
| Office desk | Common real-world setting |
| Bedroom with items | Cluttered background |
| Outdoor (natural light) | High variation |
| Laptop screen background | Bright backlight |
| Bookshelf | Textured background |

### Lighting Conditions
| Condition | Description |
|-----------|-------------|
| Natural daylight | Window light, daytime |
| Indoor overhead lamp | Standard room lighting |
| Low-light | Evening, minimal lighting |
| Side lighting | Single lamp at 90° |
| Backlit | Light source behind signer |

### Camera Positions
| Position | Distance |
|----------|----------|
| Front-facing | Medium (~60 cm) |
| Slight left tilt (15°) | Medium |
| Slight right tilt (15°) | Medium |
| Close distance | ~30 cm |
| Far distance | ~90 cm |

---

## Capture Procedure

For each class:

1. Run `python src/collect_data.py --class_name A --num_images 100`
2. Position hand within the yellow guide box shown on screen
3. Press `A` to enable auto-capture mode (0.1s interval)
4. Vary background, lighting, and camera angle every ~15 images
5. Collect from each of the 3 signers

---

## Augmentation Pipeline

Applied offline using `src/augment.py` to expand 100 → 600 images/class.

| Transform | Probability | Parameters |
|-----------|------------|------------|
| Random rotation | 0.70 | ±15° |
| Random translation | 0.50 | ±10% |
| Random scale | 0.50 | 0.85–1.15× |
| Random crop | 0.50 | 90% crop |
| Perspective transform | 0.40 | 5% distortion |
| Brightness adjust | 0.80 | 0.5–1.5× |
| Contrast adjust | 0.60 | 0.6–1.4× |
| Colour jitter | 0.50 | ±20% per channel |
| Gaussian blur | 0.30 | 3–7px kernel |
| Motion blur | 0.20 | up to 9px |
| Shadow simulation | 0.30 | random half-shadow |
| Gaussian noise | 0.40 | std 5–25 |

**Note:** Horizontal and vertical flips are intentionally excluded because many ASL signs are handedness-sensitive (e.g., a mirrored "A" could be misidentified).

---

## Annotation Format

Each `.txt` label file has one line per detected hand:

```
class_id  x_center  y_center  width  height
```

All values are normalised to [0, 1] relative to image dimensions.

### Example annotation (`A_0001.txt`):
```
0 0.512 0.438 0.364 0.521
```

Meaning: class 0 (A), bounding box centred at (51.2%, 43.8%), width 36.4%, height 52.1% of image.

### Class ID mapping
| ID | Class | ID | Class |
|----|-------|----|-------|
| 0  | A     | 6  | G     |
| 1  | B     | 7  | H     |
| 2  | C     | 8  | I     |
| 3  | D     | 9  | J     |
| 4  | E     | 10 | K     |
| 5  | F     | 11 | L     |

---

## Annotation Quality Rules

1. **Tight boxes** — bounding box covers the entire hand from wrist to fingertips
2. **No padding** — avoid excessively loose boxes that include large amounts of background
3. **Consistency** — same annotation style applied by all annotators
4. **Validation** — all annotations validated by `src/utils.py --validate` before training
5. **No corrupted files** — images with failed annotations removed before splitting

### Annotation tool commands

**Roboflow (web-based, recommended):**
```bash
# Upload raw images to Roboflow project
# Annotate in browser, export as YOLO format
pip install roboflow
# See notebooks/asl_training_evaluation.ipynb for Roboflow download snippet
```

**LabelImg (local):**
```bash
pip install labelImg
labelImg dataset/images/raw/A dataset/labels/raw/A
# Select YOLO format in Format menu
```

---

## Dataset Split

Run `python src/utils.py --split` to generate the train/valid/test split.

| Split | Ratio | Images (approx.) |
|-------|-------|-----------------|
| Train | 75%   | ~5,400           |
| Valid | 15%   | ~1,080           |
| Test  | 10%   | ~720             |

The split is done **per class** to preserve balance, with a fixed random seed (42) for reproducibility.

---

## Known Limitations

- **Signs J and K** involve significant motion; static frame detection has reduced accuracy
- **Low-light images** may reduce bounding box precision (addressed partially by brightness augmentation)
- **Partial hand visibility** (hand near edge of frame) was included in ~5% of training images to improve robustness
