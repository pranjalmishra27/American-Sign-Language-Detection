# Annotation Samples

This file documents example YOLO annotations for each ASL class.
Each entry shows: filename, raw annotation, and a human-readable description.

---

## Format Reference

```
class_id  x_center  y_center  width  height
```
All coordinates are normalised to [0.0, 1.0] relative to image dimensions.

---

## Per-Class Sample Annotations

### Class 0 — A (closed fist, thumb to side)
**File:** `dataset/labels/train/A_0001.txt`
```
0 0.512 0.438 0.364 0.521
```
Hand centred slightly right-of-middle, occupying ~36% width × 52% height.

---

### Class 1 — B (flat hand, fingers together, thumb tucked)
**File:** `dataset/labels/train/B_0001.txt`
```
1 0.489 0.421 0.312 0.568
```
Tall narrow box — B is a vertically extended sign.

---

### Class 2 — C (curved hand, C-shape)
**File:** `dataset/labels/train/C_0001.txt`
```
2 0.502 0.445 0.381 0.492
```
Roughly square bounding box — C is compact with curved fingers.

---

### Class 3 — D (index finger up, others curved)
**File:** `dataset/labels/train/D_0001.txt`
```
3 0.476 0.418 0.298 0.543
```
Taller box due to extended index finger.

---

### Class 4 — E (all fingers bent, curved)
**File:** `dataset/labels/train/E_0001.txt`
```
4 0.521 0.459 0.342 0.478
```
Compact, nearly square box.

---

### Class 5 — F (thumb and index touching, others extended)
**File:** `dataset/labels/train/F_0001.txt`
```
5 0.498 0.432 0.367 0.514
```
Medium box, slight horizontal extension from pinky.

---

### Class 6 — G (index pointing sideways, thumb parallel)
**File:** `dataset/labels/train/G_0001.txt`
```
6 0.514 0.448 0.412 0.421
```
Wider box — G points horizontally, requiring more horizontal space.

---

### Class 7 — H (two fingers extended sideways)
**File:** `dataset/labels/train/H_0001.txt`
```
7 0.508 0.451 0.421 0.418
```
Wide, flat box — H is a horizontal sign like G.

---

### Class 8 — I (pinky extended upward)
**File:** `dataset/labels/train/I_0001.txt`
```
8 0.495 0.411 0.287 0.559
```
Tall narrow box — I has only the pinky extended vertically.

---

### Class 9 — J (pinky trace J-shape — static frame shows I-position)
**File:** `dataset/labels/train/J_0001.txt`
```
9 0.502 0.418 0.294 0.551
```
Similar to I at starting frame; temporal context needed for full J.

---

### Class 10 — K (index and middle extended, V-shape with thumb)
**File:** `dataset/labels/train/K_0001.txt`
```
10 0.487 0.428 0.354 0.532
```
Tall box with moderate width due to two extended fingers.

---

### Class 11 — L (index up, thumb out — L-shape)
**File:** `dataset/labels/train/L_0001.txt`
```
11 0.481 0.436 0.398 0.528
```
Wider box — L requires both vertical (index) and horizontal (thumb) extent.

---

## Multi-Detection Example

Some images contain a hand near the frame border, testing robustness:

**File:** `dataset/labels/train/A_0087.txt` (hand partially out of frame)
```
0 0.062 0.448 0.124 0.521
```
Hand near left edge — x_center at 6.2%, width 12.4% (partially visible).

---

## Annotation Statistics

| Class | Avg x_center | Avg y_center | Avg width | Avg height | Notes |
|-------|-------------|-------------|-----------|-----------|-------|
| A | 0.50 | 0.44 | 0.36 | 0.52 | Compact fist |
| B | 0.50 | 0.42 | 0.31 | 0.57 | Tall, narrow |
| C | 0.50 | 0.45 | 0.38 | 0.49 | Near-square |
| D | 0.50 | 0.42 | 0.30 | 0.54 | Tall (index up) |
| E | 0.50 | 0.46 | 0.34 | 0.48 | Compact |
| F | 0.50 | 0.43 | 0.37 | 0.51 | Medium |
| G | 0.50 | 0.45 | 0.41 | 0.42 | Wide (horizontal) |
| H | 0.50 | 0.45 | 0.42 | 0.42 | Wide (horizontal) |
| I | 0.50 | 0.41 | 0.29 | 0.56 | Tall, narrow |
| J | 0.50 | 0.42 | 0.29 | 0.55 | Tall (dynamic) |
| K | 0.50 | 0.43 | 0.35 | 0.53 | Tall-medium |
| L | 0.50 | 0.44 | 0.40 | 0.53 | Wide-tall |
