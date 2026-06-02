"""
inference.py
------------
Run ASL detection inference on single images, directories, or video files.
Outputs annotated images/video and a JSON results file.

Usage:
    # Single image
    python src/inference.py --source path/to/image.jpg

    # Directory of images
    python src/inference.py --source path/to/images/

    # Video file
    python src/inference.py --source path/to/video.mp4

    # Custom weights and threshold
    python src/inference.py \
        --source  dataset/images/test \
        --weights models/asl_yolov8n/weights/best.pt \
        --conf    0.5 \
        --output  outputs/inference_results
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

ASL_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


def annotate_frame(
    frame: np.ndarray,
    boxes: list,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding boxes and labels on a single frame."""
    out = frame.copy()
    for (x1, y1, x2, y2, cls_name, conf) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{cls_name}: {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, tag, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out


def predict_frame(model, frame: np.ndarray, conf: float, iou: float,
                   img_size: int) -> List[tuple]:
    """Run inference on one frame and return list of detections."""
    results = model.predict(frame, conf=conf, iou=iou,
                             imgsz=img_size, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            c = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())
            cls_name = ASL_CLASSES[cls_id] if cls_id < len(ASL_CLASSES) else "?"
            detections.append((x1, y1, x2, y2, cls_name, c))
    return detections


# ── image inference ───────────────────────────────────────────────────────────

def infer_images(model, image_paths: List[Path], conf: float, iou: float,
                  img_size: int, output_dir: Path) -> dict:
    """Run inference on a list of images and save annotated outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    total_latency = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARNING] Cannot read: {img_path}")
            continue

        t0 = time.perf_counter()
        detections = predict_frame(model, frame, conf, iou, img_size)
        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency.append(latency_ms)

        annotated = annotate_frame(frame, detections)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        all_results[img_path.name] = {
            "latency_ms": round(latency_ms, 2),
            "detections": [
                {"bbox": [x1, y1, x2, y2], "class": cls, "confidence": round(c, 4)}
                for (x1, y1, x2, y2, cls, c) in detections
            ],
        }
        print(f"  {img_path.name:40s}  "
              f"detections: {len(detections)}  "
              f"latency: {latency_ms:.1f} ms")

    summary = {
        "total_images": len(image_paths),
        "mean_latency_ms": round(np.mean(total_latency), 2) if total_latency else 0,
        "mean_fps": round(1000 / np.mean(total_latency), 1) if total_latency else 0,
        "results": all_results,
    }
    json_path = output_dir / "inference_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Results saved to {json_path}")
    return summary


# ── video inference ───────────────────────────────────────────────────────────

def infer_video(model, video_path: Path, conf: float, iou: float,
                 img_size: int, output_dir: Path) -> dict:
    """Run inference on a video file and save annotated output."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = output_dir / (video_path.stem + "_annotated.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w, h))

    latencies = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detections = predict_frame(model, frame, conf, iou, img_size)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        annotated = annotate_frame(frame, detections)
        fps_display = 1000 / latency_ms
        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{n_frames}  "
                  f"mean latency: {np.mean(latencies):.1f} ms")

    cap.release()
    writer.release()

    summary = {
        "input_video": str(video_path),
        "output_video": str(out_path),
        "total_frames": frame_idx,
        "mean_latency_ms": round(np.mean(latencies), 2),
        "mean_fps": round(1000 / np.mean(latencies), 1),
    }
    print(f"\n[INFO] Annotated video saved to: {out_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ASL Detection inference")
    parser.add_argument("--source",  type=str, required=True,
                        help="Image file, directory, or video file")
    parser.add_argument("--weights", type=str,
                        default="models/asl_yolov8n/weights/best.pt")
    parser.add_argument("--conf",    type=float, default=0.5)
    parser.add_argument("--iou",     type=float, default=0.5)
    parser.add_argument("--img_size",type=int,   default=640)
    parser.add_argument("--output",  type=str,   default="outputs/inference_results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.weights)
    source = Path(args.source)
    output_dir = Path(args.output)

    if source.is_dir():
        img_paths = [p for p in source.iterdir()
                     if p.suffix.lower() in IMAGE_EXTS]
        print(f"[INFO] Found {len(img_paths)} images in {source}")
        infer_images(model, img_paths, args.conf, args.iou,
                      args.img_size, output_dir)

    elif source.suffix.lower() in IMAGE_EXTS:
        infer_images(model, [source], args.conf, args.iou,
                      args.img_size, output_dir)

    elif source.suffix.lower() in VIDEO_EXTS:
        infer_video(model, source, args.conf, args.iou,
                     args.img_size, output_dir)

    else:
        print(f"[ERROR] Unsupported source: {source}")
