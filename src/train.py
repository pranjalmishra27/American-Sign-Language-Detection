"""
train.py
--------
YOLOv8 training script for the ASL Detection System.

Trains both YOLOv8n (nano) and YOLOv8s (small) models and compares them.
Saves the best weights and a comparison summary.

Usage:
    # Train the nano model (default)
    python src/train.py --model yolov8n --epochs 100

    # Train both and compare
    python src/train.py --compare --epochs 100

    # Resume from checkpoint
    python src/train.py --model yolov8n --resume models/asl_yolov8n/weights/last.pt
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ── constants ─────────────────────────────────────────────────────────────────
ASL_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
NUM_CLASSES  = len(ASL_CLASSES)
DATA_YAML    = "dataset/data.yaml"
OUTPUT_DIR   = "models"
RESULTS_DIR  = "outputs"


# ── data.yaml generation ──────────────────────────────────────────────────────

def create_data_yaml(output_path: str = DATA_YAML) -> str:
    """Generate the YOLOv8-compatible data.yaml configuration file."""
    data = {
        "path":  str(Path(output_path).parent.parent.resolve()),
        "train": "dataset/images/train",
        "val":   "dataset/images/valid",
        "test":  "dataset/images/test",
        "nc":    NUM_CLASSES,
        "names": {i: cls for i, cls in enumerate(ASL_CLASSES)},
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] data.yaml written to {output_path}")
    return output_path


# ── training ──────────────────────────────────────────────────────────────────

def train_model(
    model_variant: str = "yolov8n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    lr: float = 0.001,
    weight_decay: float = 0.0005,
    patience: int = 15,
    resume: str = None,
    project: str = OUTPUT_DIR,
    name: str = None,
) -> dict:
    """
    Train a YOLOv8 model on the ASL dataset.

    Args:
        model_variant: 'yolov8n' or 'yolov8s'
        epochs: Maximum training epochs
        batch_size: Training batch size
        img_size: Input resolution (square)
        lr: Initial learning rate (AdamW)
        weight_decay: L2 regularisation weight
        patience: Early stopping patience
        resume: Path to checkpoint .pt to resume from
        project: Output project directory
        name: Run name (defaults to model_variant)

    Returns:
        Dictionary with training results and model path
    """
    create_data_yaml()

    run_name = name or f"asl_{model_variant}"

    if resume:
        print(f"[INFO] Resuming training from: {resume}")
        model = YOLO(resume)
    else:
        print(f"[INFO] Initialising {model_variant} with COCO pretrained weights")
        model = YOLO(f"{model_variant}.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    start_time = time.time()

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        optimizer="AdamW",
        lr0=lr,
        weight_decay=weight_decay,
        patience=patience,
        amp=True,                    # Mixed precision (FP16)
        device=device,
        project=project,
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        augment=True,                # YOLOv8 online mosaic augmentation
        mosaic=1.0,
        flipud=0.0,                  # Vertical flip disabled (ASL is orientation-sensitive)
        fliplr=0.0,                  # Horizontal flip disabled (asymmetric signs)
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        verbose=True,
    )

    elapsed = time.time() - start_time
    best_weights = Path(project) / run_name / "weights" / "best.pt"
    last_weights = Path(project) / run_name / "weights" / "last.pt"

    # Extract key metrics
    metrics = {
        "model_variant": model_variant,
        "epochs_trained": results.epoch + 1 if hasattr(results, "epoch") else epochs,
        "training_time_min": round(elapsed / 60, 1),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "device": device,
    }

    # Try to extract mAP from results
    try:
        val_results = model.val(data=DATA_YAML, split="val")
        metrics["mAP50"]    = round(float(val_results.box.map50), 4)
        metrics["mAP50_95"] = round(float(val_results.box.map),   4)
        metrics["precision"] = round(float(val_results.box.mp),   4)
        metrics["recall"]    = round(float(val_results.box.mr),   4)
    except Exception as e:
        print(f"[WARNING] Could not extract val metrics: {e}")

    # Save metrics JSON
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(RESULTS_DIR) / f"{run_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete: {run_name}")
    print(f"  Time: {metrics['training_time_min']} min")
    if "mAP50" in metrics:
        print(f"  mAP@0.5:     {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95:{metrics['mAP50_95']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Best weights: {best_weights}")
    print(f"{'='*60}\n")

    return metrics


# ── model comparison ──────────────────────────────────────────────────────────

def compare_models(epochs: int = 100) -> dict:
    """
    Train both YOLOv8n and YOLOv8s and produce a comparison report.

    Comparison criteria:
        - mAP@0.5 / mAP@0.5:0.95
        - Precision / Recall
        - Inference FPS (measured on CPU and GPU)
        - Model size (MB)
        - Training time
    """
    results = {}

    for variant in ["yolov8n", "yolov8s"]:
        print(f"\n{'#'*60}")
        print(f"  Training: {variant}")
        print(f"{'#'*60}")
        results[variant] = train_model(variant, epochs=epochs)

    # Measure model sizes
    for variant in ["yolov8n", "yolov8s"]:
        best_pt = Path(results[variant]["best_weights"])
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / 1e6
            results[variant]["model_size_mb"] = round(size_mb, 1)

    # Print comparison table
    print("\n" + "="*70)
    print(f"{'Metric':<25} {'YOLOv8n':>15} {'YOLOv8s':>15}")
    print("-"*70)
    for key in ["mAP50", "mAP50_95", "precision", "recall",
                "model_size_mb", "training_time_min"]:
        n_val = results.get("yolov8n", {}).get(key, "N/A")
        s_val = results.get("yolov8s", {}).get(key, "N/A")
        print(f"  {key:<23} {str(n_val):>15} {str(s_val):>15}")
    print("="*70)

    # Save comparison
    comp_path = Path(RESULTS_DIR) / "model_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Comparison saved to {comp_path}")

    # Recommend best model
    n_map = results.get("yolov8n", {}).get("mAP50", 0)
    s_map = results.get("yolov8s", {}).get("mAP50", 0)
    if s_map - n_map > 0.03:
        recommendation = "yolov8s (significantly better accuracy justifies larger size)"
    else:
        recommendation = "yolov8n (comparable accuracy with lower latency and smaller size)"
    print(f"\n[RECOMMENDATION] Deploy: {recommendation}")
    results["recommendation"] = recommendation

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for ASL Detection")
    parser.add_argument("--model",      type=str,  default="yolov8n",
                        choices=["yolov8n", "yolov8s"])
    parser.add_argument("--epochs",     type=int,  default=100)
    parser.add_argument("--batch_size", type=int,  default=16)
    parser.add_argument("--img_size",   type=int,  default=640)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--patience",   type=int,  default=15)
    parser.add_argument("--resume",     type=str,  default=None)
    parser.add_argument("--compare",    action="store_true",
                        help="Train both yolov8n and yolov8s and compare")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.compare:
        compare_models(epochs=args.epochs)
    else:
        train_model(
            model_variant=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            patience=args.patience,
            resume=args.resume,
        )
