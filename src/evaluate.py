"""
evaluate.py
-----------
Full evaluation suite for the trained ASL Detection model.

Produces:
  - mAP@0.5 and mAP@0.5:0.95
  - Per-class precision, recall, F1
  - Confusion matrix (raw + normalised)
  - Error analysis (false positives, false negatives, class confusions)
  - Performance benchmark (FPS, latency, model size, RAM/GPU usage)
  - Evaluation report (JSON + Markdown)

Usage:
    python src/evaluate.py \
        --weights models/asl_yolov8n/weights/best.pt \
        --data    dataset/data.yaml \
        --split   test \
        --output  outputs/evaluation
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import psutil

warnings.filterwarnings("ignore")

ASL_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


# ── utility ───────────────────────────────────────────────────────────────────

def load_model(weights_path: str):
    """Load a YOLOv8 model from weights file."""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    print(f"[INFO] Loaded model: {weights_path}")
    return model


def model_size_mb(weights_path: str) -> float:
    return round(Path(weights_path).stat().st_size / 1e6, 2)


# ── mAP evaluation ───────────────────────────────────────────────────────────

def run_validation(model, data_yaml: str, split: str = "test",
                   img_size: int = 640, conf: float = 0.25, iou: float = 0.5) -> dict:
    """Run YOLOv8 validation and return metrics dict."""
    print(f"\n[INFO] Running validation on split: {split}")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        verbose=True,
    )

    metrics = {
        "mAP50":          round(float(results.box.map50), 4),
        "mAP50_95":       round(float(results.box.map),   4),
        "mean_precision": round(float(results.box.mp),    4),
        "mean_recall":    round(float(results.box.mr),    4),
    }

    # Per-class metrics
    if hasattr(results.box, "ap_class_index"):
        per_class = {}
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = ASL_CLASSES[cls_idx]
            per_class[cls_name] = {
                "precision": round(float(results.box.p[i]),  4),
                "recall":    round(float(results.box.r[i]),  4),
                "mAP50":     round(float(results.box.ap50[i]), 4),
                "f1":        round(
                    2 * float(results.box.p[i]) * float(results.box.r[i]) /
                    (float(results.box.p[i]) + float(results.box.r[i]) + 1e-8), 4),
            }
        metrics["per_class"] = per_class

    return metrics


# ── inference benchmark ───────────────────────────────────────────────────────

def benchmark_inference(model, img_size: int = 640, n_warmup: int = 20,
                         n_runs: int = 100) -> dict:
    """
    Measure inference speed on a synthetic blank frame.

    Returns dict with latency_ms, fps, device info.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    print(f"\n[INFO] Benchmarking on {device}  ({n_warmup} warmup + {n_runs} timed runs)")

    # Warmup
    for _ in range(n_warmup):
        model.predict(dummy, verbose=False, device=device)

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, device=device)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat_arr = np.array(latencies)

    benchmark = {
        "device": device,
        "latency_mean_ms":   round(float(lat_arr.mean()), 2),
        "latency_p50_ms":    round(float(np.percentile(lat_arr, 50)), 2),
        "latency_p95_ms":    round(float(np.percentile(lat_arr, 95)), 2),
        "latency_p99_ms":    round(float(np.percentile(lat_arr, 99)), 2),
        "fps":               round(1000.0 / float(lat_arr.mean()), 1),
        "meets_target":      float(lat_arr.mean()) < 50.0,
    }

    if device == "cuda":
        benchmark["gpu_name"] = torch.cuda.get_device_name(0)
        benchmark["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        benchmark["gpu_allocated_mb"] = round(
            torch.cuda.memory_allocated() / 1e6, 1)

    benchmark["ram_used_gb"] = round(psutil.virtual_memory().used / 1e9, 2)
    benchmark["cpu_percent"] = psutil.cpu_percent(interval=0.5)

    print(f"  Mean latency : {benchmark['latency_mean_ms']} ms")
    print(f"  p95 latency  : {benchmark['latency_p95_ms']} ms")
    print(f"  FPS          : {benchmark['fps']}")
    print(f"  Target (<50ms): {'✓ PASS' if benchmark['meets_target'] else '✗ FAIL'}")
    return benchmark


# ── confusion matrix ──────────────────────────────────────────────────────────

def build_confusion_matrix(model, images_dir: str, labels_dir: str,
                            conf_thresh: float = 0.5,
                            iou_thresh: float = 0.5) -> np.ndarray:
    """
    Build a confusion matrix by running inference on the test set
    and matching predictions to ground-truth labels.
    """
    n = len(ASL_CLASSES)
    cm = np.zeros((n, n), dtype=int)

    img_paths = list(Path(images_dir).glob("*.jpg")) + \
                list(Path(images_dir).glob("*.png"))

    for img_path in img_paths:
        lbl_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # Ground-truth classes
        gt_classes = set()
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    gt_classes.add(int(parts[0]))

        # Predictions
        results = model.predict(str(img_path), conf=conf_thresh,
                                verbose=False)
        pred_classes = set()
        for r in results:
            for cls in r.boxes.cls.cpu().numpy().astype(int):
                pred_classes.add(cls)

        # Fill CM (dominant class per image heuristic)
        for gt in gt_classes:
            if pred_classes:
                for pred in pred_classes:
                    cm[gt, pred] += 1
            else:
                # False negative — no detection
                cm[gt, gt] = max(0, cm[gt, gt] - 1)  # flag miss

    return cm


def save_confusion_matrix(cm: np.ndarray, output_dir: str, normalize: bool = True):
    """Save confusion matrix as PNG."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for norm, suffix in [(False, "raw"), (True, "normalized")]:
        data = cm.copy().astype(float)
        if norm:
            row_sums = data.sum(axis=1, keepdims=True)
            data = np.divide(data, row_sums, where=row_sums != 0)
            fmt = ".2f"
        else:
            fmt = "d"
            data = data.astype(int)

        plt.figure(figsize=(14, 12))
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=ASL_CLASSES, yticklabels=ASL_CLASSES,
                    linewidths=0.5)
        plt.title(f"Confusion Matrix ({'Normalized' if norm else 'Raw Counts'})",
                  fontsize=14)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True",      fontsize=12)
        plt.tight_layout()
        out_path = Path(output_dir) / f"confusion_matrix_{suffix}.png"
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        print(f"[INFO] Saved: {out_path}")


# ── error analysis ────────────────────────────────────────────────────────────

def error_analysis(cm: np.ndarray) -> dict:
    """
    Derive error analysis from confusion matrix.

    Returns dict with:
      - false_positive_classes: most common FP sources
      - false_negative_classes: most common FN sources
      - top_confusions: pairs most often confused
    """
    errors = {
        "false_positives": {},
        "false_negatives": {},
        "top_confusions": [],
    }

    for i, cls in enumerate(ASL_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp   # other classes predicted as cls
        fn = cm[i, :].sum() - tp   # cls predicted as other

        errors["false_positives"][cls] = int(fp)
        errors["false_negatives"][cls] = int(fn)

    # Top-5 off-diagonal confusions
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat_idx = np.argsort(cm_copy.ravel())[::-1][:10]
    for idx in flat_idx:
        i, j = divmod(idx, len(ASL_CLASSES))
        if cm_copy[i, j] > 0:
            errors["top_confusions"].append({
                "true": ASL_CLASSES[i],
                "predicted": ASL_CLASSES[j],
                "count": int(cm_copy[i, j]),
            })

    return errors


# ── report generation ─────────────────────────────────────────────────────────

def generate_report(metrics: dict, benchmark: dict, errors: dict,
                    weights_path: str, output_dir: str):
    """Write JSON and Markdown evaluation reports."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    size_mb = model_size_mb(weights_path)

    report = {
        "model_weights":  weights_path,
        "model_size_mb":  size_mb,
        "detection_metrics": metrics,
        "benchmark":      benchmark,
        "error_analysis": errors,
    }

    # JSON
    json_path = Path(output_dir) / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown
    md_lines = [
        "# ASL Detection — Evaluation Report\n",
        f"**Model**: `{weights_path}`  \n",
        f"**Model Size**: {size_mb} MB\n",
        "",
        "## Detection Metrics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| mAP@0.5 | {metrics.get('mAP50', 'N/A')} |",
        f"| mAP@0.5:0.95 | {metrics.get('mAP50_95', 'N/A')} |",
        f"| Mean Precision | {metrics.get('mean_precision', 'N/A')} |",
        f"| Mean Recall | {metrics.get('mean_recall', 'N/A')} |",
        "",
        "## Inference Benchmark",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Device | {benchmark.get('device', 'N/A')} |",
        f"| Mean Latency | {benchmark.get('latency_mean_ms', 'N/A')} ms |",
        f"| p95 Latency | {benchmark.get('latency_p95_ms', 'N/A')} ms |",
        f"| FPS | {benchmark.get('fps', 'N/A')} |",
        f"| Target (<50ms) | {'✓ PASS' if benchmark.get('meets_target') else '✗ FAIL'} |",
        "",
    ]

    if "per_class" in metrics:
        md_lines += ["## Per-Class Metrics",
                     "| Class | Precision | Recall | F1 | mAP@0.5 |",
                     "|-------|-----------|--------|----|---------|"]
        for cls, m in metrics["per_class"].items():
            md_lines.append(
                f"| {cls} | {m['precision']} | {m['recall']} | {m['f1']} | {m['mAP50']} |"
            )

    md_lines += [
        "",
        "## Top Class Confusions",
        "| True | Predicted | Count |",
        "|------|-----------|-------|",
    ]
    for conf in errors.get("top_confusions", [])[:5]:
        md_lines.append(f"| {conf['true']} | {conf['predicted']} | {conf['count']} |")

    md_path = Path(output_dir) / "evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[INFO] Reports saved:")
    print(f"  JSON : {json_path}")
    print(f"  MD   : {md_path}")
    return str(json_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASL Detection model")
    parser.add_argument("--weights", type=str,
                        default="models/asl_yolov8n/weights/best.pt")
    parser.add_argument("--data",    type=str, default="dataset/data.yaml")
    parser.add_argument("--split",   type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--output",  type=str, default="outputs/evaluation")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.5)
    parser.add_argument("--skip_cm", action="store_true",
                        help="Skip confusion matrix generation (faster)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = load_model(args.weights)

    # Validation metrics
    metrics = run_validation(
        model, 
        args.data, 
        split=args.split, 
        conf=args.conf, 
        iou=args.iou, 
        img_size=96
    )

    # Benchmark
    benchmark = benchmark_inference(model, img_size=96)

    # Confusion matrix (optional — requires annotated test images)
    cm = np.zeros((len(ASL_CLASSES), len(ASL_CLASSES)), dtype=int)
    if not args.skip_cm:
        test_images = f"dataset/images/{args.split}"
        test_labels = f"dataset/labels/{args.split}"
        if Path(test_images).exists():
            cm = build_confusion_matrix(model, test_images, test_labels,
                                         args.conf, args.iou)
            save_confusion_matrix(cm, args.output)

    errors = error_analysis(cm)
    generate_report(metrics, benchmark, errors, args.weights, args.output)

    print("\n[DONE] Evaluation complete.")
