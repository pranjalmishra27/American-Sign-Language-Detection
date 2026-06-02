"""
utils.py
--------
Shared utilities for the ASL Detection project:
  - Dataset split (train/valid/test)
  - YOLO annotation validation
  - Logging helpers
  - Metrics formatting
  - Visualization helpers

Usage (dataset split):
    python src/utils.py --split \
        --images_dir dataset/images/augmented \
        --labels_dir dataset/labels/raw \
        --output_dir dataset
"""

import os
import cv2
import shutil
import random
import argparse
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ASL_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# ── dataset split ─────────────────────────────────────────────────────────────

def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.75,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Split annotated images into train / valid / test sets.

    Expects:
        images_dir/<class>/<image>.jpg
        labels_dir/<class>/<image>.txt  (YOLO format)

    Outputs:
        output_dir/images/{train,valid,test}/<image>.jpg
        output_dir/labels/{train,valid,test}/<image>.txt
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(seed)
    counts: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}

    for split in ["train", "valid", "test"]:
        (Path(output_dir) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "labels" / split).mkdir(parents=True, exist_ok=True)

    for cls in ASL_CLASSES:
        img_cls_dir = Path(images_dir) / cls
        lbl_cls_dir = Path(labels_dir) / cls

        if not img_cls_dir.exists():
            logger.warning(f"No images found for class {cls}, skipping.")
            continue

        images = sorted(img_cls_dir.glob("*.jpg")) + sorted(img_cls_dir.glob("*.png"))
        paired = []
        for img_path in images:
            lbl_path = lbl_cls_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                paired.append((img_path, lbl_path))
            else:
                logger.debug(f"No label for {img_path.name}, skipping.")

        random.shuffle(paired)
        n = len(paired)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        splits = {
            "train": paired[:n_train],
            "valid": paired[n_train:n_train + n_valid],
            "test":  paired[n_train + n_valid:],
        }

        for split_name, pairs in splits.items():
            for img_path, lbl_path in pairs:
                shutil.copy2(img_path, Path(output_dir) / "images" / split_name / img_path.name)
                shutil.copy2(lbl_path, Path(output_dir) / "labels" / split_name / lbl_path.name)
            counts[split_name] += len(pairs)
            logger.info(f"  {cls}  {split_name}: {len(pairs)} samples")

    logger.info(f"\nFinal split -> train: {counts['train']}  "
                f"valid: {counts['valid']}  test: {counts['test']}")
    return counts


# ── YOLO annotation validation ───────────────────────────────────────────────

def validate_annotations(labels_dir: str) -> Tuple[int, int]:
    """
    Validate YOLO annotation files.

    Checks:
      - Each line has exactly 5 fields
      - class_id is integer within [0, len(ASL_CLASSES))
      - x_center, y_center, width, height are in [0, 1]

    Returns:
        (valid_count, invalid_count)
    """
    valid = 0
    invalid = 0
    for lbl_path in Path(labels_dir).rglob("*.txt"):
        with open(lbl_path) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                logger.warning(f"Bad annotation (fields): {lbl_path}")
                invalid += 1
                continue
            try:
                cls_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                assert 0 <= cls_id < len(ASL_CLASSES)
                assert all(0.0 <= c <= 1.0 for c in coords)
                valid += 1
            except (ValueError, AssertionError):
                logger.warning(f"Bad annotation (values): {lbl_path}  ->  {line.strip()}")
                invalid += 1
    logger.info(f"Annotation validation: {valid} valid, {invalid} invalid")
    return valid, invalid


# ── dataset statistics ────────────────────────────────────────────────────────

def dataset_statistics(images_dir: str) -> Dict[str, int]:
    """Count images per class per split."""
    stats: Dict[str, int] = {}
    for split in ["train", "valid", "test"]:
        split_dir = Path(images_dir) / split
        if split_dir.exists():
            stats[split] = len(list(split_dir.glob("*.jpg"))) + \
                           len(list(split_dir.glob("*.png")))
    return stats


# ── visualization helpers ─────────────────────────────────────────────────────

def draw_detection(
    frame: np.ndarray,
    boxes: List[Tuple],
    labels: List[str],
    confidences: List[float],
    fps: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and confidence scores on a frame.

    Args:
        frame: BGR image
        boxes: list of (x1, y1, x2, y2) in pixel coords
        labels: predicted class names
        confidences: confidence scores [0, 1]
        fps: optional FPS to display
        color: BGR box color

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    for (x1, y1, x2, y2), label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        tag = f"{label}: {conf:.1%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

        # Label text
        cv2.putText(annotated, tag, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if fps is not None:
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    return annotated


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "outputs/confusion_matrix.png",
    normalize: bool = True,
):
    """Save a confusion matrix heatmap."""
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_training_curves(results_csv: str, output_path: str = "outputs/training_curves.png"):
    """Plot training and validation loss/metrics from YOLOv8 results.csv."""
    import pandas as pd
    df = pd.read_csv(results_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("val/box_loss",   "Val Box Loss"),
        ("val/cls_loss",   "Val Cls Loss"),
        ("metrics/precision(B)", "Precision"),
    ]
    for ax, (col, title) in zip(axes.flatten(), metrics):
        if col in df.columns:
            ax.plot(df["epoch"], df[col])
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)

    plt.suptitle("YOLOv8 Training Curves", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ASL Detection utility scripts")
    parser.add_argument("--split", action="store_true", help="Run dataset split")
    parser.add_argument("--validate", action="store_true", help="Validate annotations")
    parser.add_argument("--images_dir", type=str, default="dataset/images/augmented")
    parser.add_argument("--labels_dir", type=str, default="dataset/labels/raw")
    parser.add_argument("--output_dir", type=str, default="dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.split:
        split_dataset(args.images_dir, args.labels_dir, args.output_dir)
    if args.validate:
        validate_annotations(args.labels_dir)
