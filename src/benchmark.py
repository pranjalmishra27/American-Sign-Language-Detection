"""
benchmark.py
------------
Standalone performance benchmarking tool for the ASL Detection model.

Measures and reports:
  - Inference latency (mean, p50, p95, p99)
  - Frames per second (FPS)
  - Model file size (MB)
  - CPU / GPU memory consumption
  - Throughput under batch conditions
  - Comparison of YOLOv8n vs YOLOv8s (if both weights available)

Outputs:
  - outputs/benchmark/benchmark_report.json
  - outputs/benchmark/benchmark_report.md
  - outputs/benchmark/latency_distribution.png

Usage:
    python src/benchmark.py --weights models/asl_yolov8n/weights/best.pt
    python src/benchmark.py --compare  # compare n vs s if both present
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import psutil

ASL_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


# ─── System info ─────────────────────────────────────────────────────────────

def collect_system_info() -> Dict:
    info = {
        "os":         platform.system() + " " + platform.release(),
        "python":     platform.python_version(),
        "cpu":        platform.processor() or "Unknown",
        "cpu_cores":  psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
    return info


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


def model_size_mb(weights: str) -> float:
    p = Path(weights)
    return round(p.stat().st_size / 1e6, 2) if p.exists() else 0.0


# ─── Latency benchmark ───────────────────────────────────────────────────────

def measure_latency(
    model,
    img_size: int = 640,
    n_warmup: int = 30,
    n_runs: int = 200,
    device: str = "auto",
) -> Dict:
    """
    Measure single-frame inference latency using a synthetic frame.

    Args:
        model: loaded YOLO model
        img_size: input resolution (square)
        n_warmup: warmup iterations (not timed)
        n_runs: timed iterations
        device: 'cpu', 'cuda', or 'auto'

    Returns:
        Dict with latency statistics and FPS.
    """
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    print(f"\n[BENCHMARK] Latency measurement")
    print(f"  Device    : {device.upper()}")
    print(f"  Resolution: {img_size}×{img_size}")
    print(f"  Warmup    : {n_warmup}  |  Timed runs: {n_runs}")

    # Warmup
    for _ in range(n_warmup):
        model.predict(dummy, verbose=False, device=device)

    # Timed
    if device == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    result = {
        "device":           device,
        "img_size":         img_size,
        "n_runs":           n_runs,
        "latency_mean_ms":  round(float(arr.mean()), 3),
        "latency_std_ms":   round(float(arr.std()),  3),
        "latency_min_ms":   round(float(arr.min()),  3),
        "latency_p50_ms":   round(float(np.percentile(arr, 50)), 3),
        "latency_p90_ms":   round(float(np.percentile(arr, 90)), 3),
        "latency_p95_ms":   round(float(np.percentile(arr, 95)), 3),
        "latency_p99_ms":   round(float(np.percentile(arr, 99)), 3),
        "latency_max_ms":   round(float(arr.max()),  3),
        "fps":              round(1000.0 / float(arr.mean()), 1),
        "meets_50ms_target": bool(arr.mean() < 50.0),
        "meets_33ms_30fps":  bool(arr.mean() < 33.3),
    }

    print(f"\n  Mean      : {result['latency_mean_ms']} ms")
    print(f"  p50       : {result['latency_p50_ms']} ms")
    print(f"  p95       : {result['latency_p95_ms']} ms")
    print(f"  p99       : {result['latency_p99_ms']} ms")
    print(f"  FPS       : {result['fps']}")
    print(f"  <50ms     : {'✓ PASS' if result['meets_50ms_target'] else '✗ FAIL'}")
    print(f"  30+ FPS   : {'✓ PASS' if result['meets_33ms_30fps'] else '✗ FAIL'}")
    return result


# ─── Multi-resolution sweep ──────────────────────────────────────────────────

def resolution_sweep(model, resolutions: List[int] = None,
                      n_runs: int = 100) -> List[Dict]:
    """Measure latency across multiple input resolutions."""
    if resolutions is None:
        resolutions = [320, 416, 480, 640]
    results = []
    print("\n[BENCHMARK] Resolution sweep")
    for res in resolutions:
        r = measure_latency(model, img_size=res, n_warmup=20, n_runs=n_runs)
        results.append({"resolution": res, **r})
    return results


# ─── Memory profiling ─────────────────────────────────────────────────────────

def measure_memory(model, img_size: int = 640) -> Dict:
    """Measure RAM and GPU memory consumed during inference."""
    import torch

    dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Baseline
    ram_before = psutil.virtual_memory().used / 1e6
    gpu_before = torch.cuda.memory_allocated() / 1e6 if device == "cuda" else 0

    # Run inference
    model.predict(dummy, verbose=False, device=device)
    torch.cuda.synchronize() if device == "cuda" else None

    ram_after = psutil.virtual_memory().used / 1e6
    gpu_after  = torch.cuda.memory_allocated() / 1e6 if device == "cuda" else 0

    return {
        "ram_delta_mb":  round(ram_after - ram_before, 1),
        "ram_total_used_mb": round(ram_after, 1),
        "gpu_allocated_mb":  round(gpu_after, 1) if device == "cuda" else 0,
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)
                       if device == "cuda" else 0,
        "cpu_percent": psutil.cpu_percent(interval=0.5),
    }


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_latency_distribution(latencies: List[float], output_path: str,
                               title: str = "Latency Distribution"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        arr = np.array(latencies)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(arr, bins=40, color="#4A90D9", edgecolor="white", linewidth=0.5)
        axes[0].axvline(arr.mean(),   color="red",    linestyle="--", label=f"Mean {arr.mean():.1f} ms")
        axes[0].axvline(np.percentile(arr, 95), color="orange", linestyle="--",
                        label=f"p95 {np.percentile(arr,95):.1f} ms")
        axes[0].axvline(50, color="green", linestyle=":", linewidth=2, label="50 ms target")
        axes[0].set_xlabel("Latency (ms)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Latency Histogram")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # Time series
        axes[1].plot(arr, alpha=0.7, linewidth=0.8, color="#4A90D9")
        axes[1].axhline(arr.mean(), color="red", linestyle="--", label=f"Mean {arr.mean():.1f} ms")
        axes[1].axhline(50, color="green", linestyle=":", linewidth=2, label="50 ms target")
        axes[1].set_xlabel("Frame index")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Latency Over Time")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[INFO] Latency plot saved: {output_path}")
    except ImportError:
        print("[WARNING] matplotlib not available, skipping plot.")


def plot_model_comparison(comparison: Dict, output_path: str):
    try:
        import matplotlib.pyplot as plt

        models = list(comparison.keys())
        metrics = ["fps", "latency_mean_ms", "latency_p95_ms"]
        titles  = ["FPS (higher is better)",
                   "Mean Latency ms (lower is better)",
                   "p95 Latency ms (lower is better)"]
        colors  = ["#4A90D9", "#E86C4A"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric, title in zip(axes, metrics, titles):
            vals = [comparison[m]["latency"].get(metric, 0) for m in models]
            bars = ax.bar(models, vals, color=colors[:len(models)], width=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals)*0.01,
                        f"{val}", ha="center", va="bottom", fontsize=11)
            ax.set_title(title)
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("YOLOv8n vs YOLOv8s — Performance Comparison", fontsize=14)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[INFO] Comparison plot saved: {output_path}")
    except ImportError:
        pass


# ─── Report generation ───────────────────────────────────────────────────────

def generate_benchmark_report(
    sys_info: Dict,
    latency: Dict,
    memory: Dict,
    model_size: float,
    res_sweep: Optional[List[Dict]],
    output_dir: str,
    model_name: str = "YOLOv8n",
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report = {
        "model": model_name,
        "model_size_mb": model_size,
        "system": sys_info,
        "latency": latency,
        "memory": memory,
        "resolution_sweep": res_sweep or [],
    }

    # JSON
    json_path = Path(output_dir) / "benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Markdown
    lines = [
        f"# Performance Benchmark Report — {model_name}\n",
        f"## System Information\n",
        f"| Property | Value |",
        f"|----------|-------|",
    ]
    for k, v in sys_info.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        f"\n## Model\n",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Model | {model_name} |",
        f"| File size | {model_size} MB |",
        f"\n## Latency (img_size={latency.get('img_size',640)}, "
        f"device={latency.get('device','').upper()})\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean latency | {latency.get('latency_mean_ms','N/A')} ms |",
        f"| p50 latency | {latency.get('latency_p50_ms','N/A')} ms |",
        f"| p95 latency | {latency.get('latency_p95_ms','N/A')} ms |",
        f"| p99 latency | {latency.get('latency_p99_ms','N/A')} ms |",
        f"| FPS | {latency.get('fps','N/A')} |",
        f"| Target <50 ms | {'✓ PASS' if latency.get('meets_50ms_target') else '✗ FAIL'} |",
        f"| 30+ FPS | {'✓ PASS' if latency.get('meets_33ms_30fps') else '✗ FAIL'} |",
        f"\n## Memory\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| RAM delta (inference) | {memory.get('ram_delta_mb','N/A')} MB |",
        f"| GPU allocated | {memory.get('gpu_allocated_mb','N/A')} MB |",
        f"| GPU peak | {memory.get('gpu_peak_mb','N/A')} MB |",
        f"| CPU % during inference | {memory.get('cpu_percent','N/A')} |",
    ]

    if res_sweep:
        lines += [
            f"\n## Resolution Sweep\n",
            f"| Resolution | Mean Latency (ms) | p95 (ms) | FPS |",
            f"|------------|------------------|---------|-----|",
        ]
        for row in res_sweep:
            lines.append(
                f"| {row['resolution']}×{row['resolution']} "
                f"| {row.get('latency_mean_ms','N/A')} "
                f"| {row.get('latency_p95_ms','N/A')} "
                f"| {row.get('fps','N/A')} |"
            )

    md_path = Path(output_dir) / "benchmark_report.md"
    md_path.write_text("\n".join(lines))

    print(f"\n[INFO] Benchmark reports saved:")
    print(f"  JSON : {json_path}")
    print(f"  MD   : {md_path}")
    return str(md_path)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ASL model performance benchmark")
    p.add_argument("--weights",    type=str,
                   default="models/asl_yolov8n/weights/best.pt")
    p.add_argument("--img_size",   type=int, default=640)
    p.add_argument("--n_runs",     type=int, default=200)
    p.add_argument("--output",     type=str, default="outputs/benchmark")
    p.add_argument("--compare",    action="store_true",
                   help="Compare YOLOv8n vs YOLOv8s if both weights present")
    p.add_argument("--res_sweep",  action="store_true",
                   help="Also sweep across multiple resolutions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys_info = collect_system_info()
    print("[INFO] System:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    if args.compare:
        comparison = {}
        for variant in ["yolov8n", "yolov8s"]:
            w = f"models/asl_{variant}/weights/best.pt"
            if not Path(w).exists():
                print(f"[SKIP] {w} not found")
                continue
            m = load_model(w)
            lat = measure_latency(m, args.img_size, n_runs=args.n_runs)
            mem = measure_memory(m, args.img_size)
            comparison[variant] = {
                "model_size_mb": model_size_mb(w),
                "latency": lat,
                "memory": mem,
            }
            generate_benchmark_report(
                sys_info, lat, mem, model_size_mb(w),
                None, args.output, model_name=variant)
        if comparison:
            plot_model_comparison(comparison,
                f"{args.output}/model_comparison_perf.png")
    else:
        model = load_model(args.weights)
        lat = measure_latency(model, args.img_size, n_runs=args.n_runs)
        mem = measure_memory(model, args.img_size)

        res_sweep = None
        if args.res_sweep:
            res_sweep = resolution_sweep(model, n_runs=50)

        name = Path(args.weights).parts[-3] if len(Path(args.weights).parts) >= 3 else "YOLOv8"
        generate_benchmark_report(sys_info, lat, mem,
                                   model_size_mb(args.weights),
                                   res_sweep, args.output, model_name=name)

    print("\n[DONE] Benchmark complete.")
