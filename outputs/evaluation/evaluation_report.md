# ASL Detection — Evaluation Report

**Model**: `models/asl_yolov8n/weights/best.pt`  

**Model Size**: 6.2 MB


## Detection Metrics
| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.9874 |
| mAP@0.5:0.95 | 0.9842 |
| Mean Precision | 0.9785 |
| Mean Recall | 0.9861 |

## Inference Benchmark
| Metric | Value |
|--------|-------|
| Device | cpu |
| Mean Latency | 17.0 ms |
| p95 Latency | 24.34 ms |
| FPS | 58.8 |
| Target (<50ms) | ✓ PASS |

## Per-Class Metrics
| Class | Precision | Recall | F1 | mAP@0.5 |
|-------|-----------|--------|----|---------|
| A | 0.9335 | 1.0 | 0.9656 | 0.995 |
| B | 0.8511 | 1.0 | 0.9195 | 0.9836 |
| C | 0.9578 | 1.0 | 0.9784 | 0.995 |
| D | 1.0 | 1.0 | 1.0 | 0.995 |
| E | 1.0 | 1.0 | 1.0 | 0.995 |
| F | 1.0 | 0.8526 | 0.9204 | 0.915 |
| G | 1.0 | 1.0 | 1.0 | 0.995 |
| H | 1.0 | 1.0 | 1.0 | 0.995 |
| I | 1.0 | 0.9801 | 0.99 | 0.995 |
| J | 1.0 | 1.0 | 1.0 | 0.995 |
| K | 1.0 | 1.0 | 1.0 | 0.995 |
| L | 1.0 | 1.0 | 1.0 | 0.995 |

## Top Class Confusions
| True | Predicted | Count |
|------|-----------|-------|
| F | B | 1 |
| I | A | 1 |