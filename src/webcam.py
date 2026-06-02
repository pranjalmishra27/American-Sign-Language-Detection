"""
webcam.py
---------
Real-time ASL Detection webcam application.

Pipeline:
    Webcam → Frame Capture → Preprocessing →
    YOLOv8 Detection → ASL Classification →
    Confidence Score → Visualization → User Display

Controls:
    Q        — quit
    S        — save screenshot
    R        — toggle recording
    +/-      — adjust confidence threshold
    SPACE    — pause / resume

Usage:
    python src/webcam.py --weights models/asl_yolov8n/weights/best.pt
"""

import argparse
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ASL_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# ── colour palette — one per class ───────────────────────────────────────────
CLASS_COLORS = {
    cls: (
        int(30  + i * 20),
        int(200 - i * 10),
        int(100 + i * 13),
    )
    for i, cls in enumerate(ASL_CLASSES)
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (9, 10), (10, 11), (11, 12),         # Middle
    (13, 14), (14, 15), (15, 16),         # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)            # Palm base
]


# ── FPS tracker ───────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ── prediction smoother (majority-vote over N frames) ────────────────────────
class PredictionSmoother:
    def __init__(self, window: int = 5):
        self._window = window
        self._history: deque = deque(maxlen=window)

    def update(self, label: str, conf: float) -> tuple:
        self._history.append((label, conf))
        if not self._history:
            return label, conf
        labels = [h[0] for h in self._history]
        majority = max(set(labels), key=labels.count)
        avg_conf = sum(h[1] for h in self._history if h[0] == majority) / \
                   max(1, labels.count(majority))
        return majority, avg_conf


# ── normalized geometric hand classifier ───────────────────────────────────────
class NormalizedHandClassifier:
    TEMPLATES = {
        "A": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # closed fist
        "B": np.array([0.0, 1.0, 1.0, 1.0, 1.0]),  # flat hand (thumb closed)
        "C": np.array([0.8, 0.4, 0.4, 0.4, 0.4]),  # curved hand
        "D": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),  # index up
        "E": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # curled fist (differentiated by sub-rule)
        "F": np.array([0.0, 0.0, 1.0, 1.0, 1.0]),  # okay sign (index touched, others open)
        "G": np.array([1.0, 1.0, 0.0, 0.0, 0.0]),  # horizontal point (differentiated by hand angle)
        "H": np.array([0.0, 1.0, 1.0, 0.0, 0.0]),  # index + middle extended horizontal
        "I": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),  # pinky extended
        "K": np.array([0.8, 1.0, 1.0, 0.0, 0.0]),  # V-shape (thumb open)
        "L": np.array([1.0, 1.0, 0.0, 0.0, 0.0]),  # L-shape (thumb + index open, upright)
    }

    last_features = np.zeros(5)
    last_scores = {}
    last_camera_angle = 0.0
    last_coords = []

    @staticmethod
    def normalize_landmarks(landmarks):
        """
        Normalize 21 hand landmarks to be scale-invariant and rotation-invariant.
        landmarks: MediaPipe Hand landmarks.
        Returns: list of 21 np.ndarray([x, y]), where Wrist is (0,0), Middle MCP (9) is (0, -1).
        """
        # Support both legacy and new MediaPipe formats
        lms = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks

        # 1. Translate wrist to (0,0)
        wrist = np.array([lms[0].x, lms[0].y])
        translated = []
        for lm in lms:
            translated.append(np.array([lm.x - wrist[0], lm.y - wrist[1]]))
            
        # 2. Compute angle of Middle MCP (landmark 9) relative to Wrist (0)
        mcp9 = translated[9]
        angle = np.arctan2(mcp9[1], mcp9[0]) # Angle in radians
        
        # 3. Rotate so landmark 9 points straight "up" (negative Y direction in image space)
        # Up in image coordinates is (0, -d). To align translated[9] with (0, -d),
        # the rotation angle should be -angle - np.pi/2.
        rot_angle = -angle - np.pi/2
        cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated = [R.dot(v) for v in translated]
        
        # 4. Scale so distance from wrist to Middle MCP is 1.0
        d = np.linalg.norm(rotated[9])
        if d < 1e-5:
            d = 1e-5
            
        normalized = [v / d for v in rotated]
        return normalized

    @classmethod
    def get_features(cls, lms_3d):
        # lms_3d: list of 21 np.ndarray([x, y, z])
        
        # Helper to compute distance ratio for a finger
        def finger_ratio(mcp, pip, dip, tip):
            d_tip_mcp = np.linalg.norm(lms_3d[tip] - lms_3d[mcp])
            d_segments = (np.linalg.norm(lms_3d[pip] - lms_3d[mcp]) + 
                          np.linalg.norm(lms_3d[dip] - lms_3d[pip]) + 
                          np.linalg.norm(lms_3d[tip] - lms_3d[dip]))
            if d_segments < 1e-5:
                return 0.0
            return d_tip_mcp / d_segments

        # 4 fingers straightness ratio
        r_index = finger_ratio(5, 6, 7, 8)
        r_middle = finger_ratio(9, 10, 11, 12)
        r_ring = finger_ratio(13, 14, 15, 16)
        r_pinky = finger_ratio(17, 18, 19, 20)

        # Thumb openness: distance from thumb tip (4) to index MCP (5)
        # Normalized by palm size (wrist 0 to middle MCP 9)
        palm_size = np.linalg.norm(lms_3d[9] - lms_3d[0])
        if palm_size < 1e-5:
            palm_size = 1e-5
        
        d_thumb_index = np.linalg.norm(lms_3d[4] - lms_3d[5]) / palm_size
        
        # Map values to [0, 1] range using optimized boundaries
        def clamp(v, lo=0.0, hi=1.0):
            return max(lo, min(hi, v))

        # Finger ratio ranges from ~0.4 (folded) to ~0.85 (extended)
        f_index = clamp((r_index - 0.4) / 0.45)
        f_middle = clamp((r_middle - 0.4) / 0.45)
        f_ring = clamp((r_ring - 0.4) / 0.45)
        f_pinky = clamp((r_pinky - 0.4) / 0.45)
        
        # Thumb distance ranges from ~0.38 (closed) to ~0.70 (open)
        f_thumb = clamp((d_thumb_index - 0.38) / 0.32)

        return np.array([f_thumb, f_index, f_middle, f_ring, f_pinky])

    @classmethod
    def classify(cls, landmarks, hand_label):
        """
        Classify ASL letters A-L based on 3D distance ratio mathematical features.
        landmarks: MediaPipe Hand landmarks.
        hand_label: "Left" or "Right" hand.
        Returns: (predicted_letter, confidence_score, explanation_str)
        """
        lms = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        lms_3d = [np.array([lm.x, lm.y, lm.z]) for lm in lms]
        
        features = cls.get_features(lms_3d)
        
        # Calculate matching scores
        scores = {}
        for cls_name, template in cls.TEMPLATES.items():
            dist = np.mean(np.abs(features - template))
            scores[cls_name] = 1.0 - dist
            
        # Get hand orientation in camera space
        wrist = lms_3d[0]
        mcp9 = lms_3d[9] - wrist
        camera_angle = np.degrees(np.arctan2(mcp9[1], mcp9[0]))
        is_upright = -140 < camera_angle < -40
        
        # Helper distances (normalized by palm size to be scale-invariant)
        palm_size = np.linalg.norm(lms_3d[9] - lms_3d[0])
        if palm_size < 1e-5:
            palm_size = 1e-5
            
        d_thumb_index_norm = np.linalg.norm(lms_3d[4] - lms_3d[8]) / palm_size
        d_index_middle_norm = np.linalg.norm(lms_3d[8] - lms_3d[12]) / palm_size
        
        # ── Refine overlapping classes via sub-rules ──────────────────────────
        
        # 1. Distinguish G and L (both have thumb + index open)
        if is_upright:
            scores["G"] *= 0.2
        else:
            scores["L"] *= 0.2
            
        # 2. Distinguish H and K (both have index + middle extended)
        if d_index_middle_norm > 0.45:
            scores["H"] *= 0.3
        else:
            scores["K"] *= 0.3
            
        # 3. Distinguish A and E (both have all closed)
        d_thumb_index_mcp = np.linalg.norm(lms_3d[4] - lms_3d[5]) / palm_size
        if d_thumb_index_mcp < 0.45:
            scores["A"] *= 0.4
        else:
            scores["E"] *= 0.4
            
        # 4. F requires index tip and thumb tip to touch
        if d_thumb_index_norm > 0.45:
            scores["F"] *= 0.2
            
        # Select best class
        best_cls = max(scores, key=scores.get)
        best_score = scores[best_cls]
        
        # Save to class variables for diagnostics
        cls.last_features = features
        cls.last_scores = scores
        cls.last_camera_angle = camera_angle
        cls.last_coords = [c.tolist() for c in lms_3d]
        
        descriptions = {
            "A": "Compact fist (thumb pressed against side)",
            "B": "Flat hand (fingers upright, thumb tucked)",
            "C": "C-shape (all fingers curved, thumb open)",
            "D": "Index finger pointing up, others folded",
            "E": "Curled fist (fingers bent tightly, thumb folded across)",
            "F": "Okay sign (thumb and index touching, others open)",
            "G": "Index and thumb pointing horizontally (G sign)",
            "H": "Index and middle extended horizontally together",
            "I": "Pinky finger extended, others folded",
            "K": "V-shape (index & middle spread, thumb upright)",
            "L": "L-shape (index pointing up, thumb pointing sideways)",
        }
        
        if best_score > 0.70:
            return best_cls, best_score, descriptions.get(best_cls, "ASL Sign")
            
        return None, 0.0, "Unknown hand pose"


# ── drawing helper ────────────────────────────────────────────────────────────
def draw_overlay(
    frame: np.ndarray,
    detections: list,          # list of (x1, y1, x2, y2, cls_name, conf)
    fps: float,
    conf_thresh: float,
    paused: bool,
    smoothed_label: str = None,
    smoothed_conf: float = None,
    backend_mode: str = "MediaPipe",
    hand_landmarks = None,
    hand_label: str = None,
    explanation: str = None,
) -> np.ndarray:
    """Draw all UI elements on the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # ── draw MediaPipe skeleton first (behind panels) ───────────────────────
    if backend_mode == "MediaPipe" and hand_landmarks:
        color = CLASS_COLORS.get(smoothed_label, (0, 255, 100)) if smoothed_label else (0, 255, 255)
        
        lms = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
        
        # Get coordinates in pixels
        pts = []
        for lm in lms:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pts.append((px, py))
            
        # Draw connections
        for connection in HAND_CONNECTIONS:
            pt1 = pts[connection[0]]
            pt2 = pts[connection[1]]
            cv2.line(out, pt1, pt2, color, 3, cv2.LINE_AA)
            
        # Draw joints
        for idx, pt in enumerate(pts):
            cv2.circle(out, pt, 8, color, -1, cv2.LINE_AA)
            cv2.circle(out, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(out, pt, 12, color, 2, cv2.LINE_AA)

    # ── YOLOv8 bounding boxes ──────────────────────────────────────────────────
    if backend_mode == "YOLOv8":
        for (x1, y1, x2, y2, cls_name, conf) in detections:
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            tag = f"{cls_name}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.rectangle(out, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
            cv2.putText(out, tag, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # ── HUD panel (top-left) ───────────────────────────────────────────────
    panel_h = 135
    panel_w = 345
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    lines = [
        (f"Mode: {backend_mode} [Press 'M' to Toggle]", (255, 200, 0)),
        (f"FPS: {fps:5.1f}",              (0, 255, 100)),
        (f"Conf Threshold: {conf_thresh:.0%}", (200, 200, 200)),
        (f"{'[PAUSED]' if paused else 'LIVE'}",
         (0, 80, 255) if paused else (100, 255, 100)),
    ]
    
    if backend_mode == "MediaPipe":
        lines.insert(2, (f"Hand: {hand_label if hand_label else 'Not Detected'}", (100, 200, 255)))
    else:
        lines.insert(2, (f"Detections: {len(detections)}", (200, 200, 200)))
        
    for i, (text, color_text) in enumerate(lines):
        cv2.putText(out, text, (10, 22 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 1, cv2.LINE_AA)

    # ── smoothed prediction panel (bottom-left) ────────────────────────────
    if smoothed_label:
        bh = 85
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h - bh), (520, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.65, out, 0.35, 0, out)
        cv2.putText(out, f"Sign: {smoothed_label}", (12, h - bh + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95,
                    CLASS_COLORS.get(smoothed_label, (0, 255, 0)), 2, cv2.LINE_AA)
        cv2.putText(out, f"Confidence: {smoothed_conf:.1%}", (12, h - bh + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    
        if explanation and backend_mode == "MediaPipe":
            cv2.putText(out, f"Gesture: {explanation}", (12, h - bh + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Debug Info Panel (top-right) ───────────────────────────────────────
    if backend_mode == "MediaPipe" and hand_landmarks:
        dp_w = 260
        dp_h = 160
        overlay_dp = out.copy()
        cv2.rectangle(overlay_dp, (w - dp_w, 0), (w, dp_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_dp, 0.65, out, 0.35, 0, out)
        
        # Calculate straightness metrics in real-time
        lms = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
        lms_3d = [np.array([lm.x, lm.y, lm.z]) for lm in lms]
        features = NormalizedHandClassifier.get_features(lms_3d)
        
        db_lines = [
            ("DEBUG: FINGER STATE", (255, 200, 0)),
            (f"Thumb ext:  {features[0]:.2f} ({'YES' if features[0] > 0.5 else 'NO'})", (0, 255, 100) if features[0] > 0.5 else (200, 200, 200)),
            (f"Index ext:  {features[1]:.2f} ({'YES' if features[1] > 0.5 else 'NO'})", (0, 255, 100) if features[1] > 0.5 else (200, 200, 200)),
            (f"Middle ext: {features[2]:.2f} ({'YES' if features[2] > 0.5 else 'NO'})", (0, 255, 100) if features[2] > 0.5 else (200, 200, 200)),
            (f"Ring ext:   {features[3]:.2f} ({'YES' if features[3] > 0.5 else 'NO'})", (0, 255, 100) if features[3] > 0.5 else (200, 200, 200)),
            (f"Pinky ext:  {features[4]:.2f} ({'YES' if features[4] > 0.5 else 'NO'})", (0, 255, 100) if features[4] > 0.5 else (200, 200, 200)),
        ]
        
        for i, (text, color_text) in enumerate(db_lines):
            cv2.putText(out, text, (w - dp_w + 10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1, cv2.LINE_AA)

    # ── controls reminder (bottom-right) ──────────────────────────────────
    controls = ["Q:quit  S:screenshot  R:record  SPACE:pause  M:mode  +/-:conf"]
    for i, c in enumerate(controls):
        cv2.putText(out, c, (w - 530, h - 10 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    return out


# ── main webcam loop ──────────────────────────────────────────────────────────
def run_webcam(
    weights: str,
    camera_index: int = 0,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    img_size: int = 640,
    smooth_window: int = 5,
    output_dir: str = "outputs",
):
    """Main real-time inference loop."""
    from ultralytics import YOLO

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model from: {weights}")
    model = YOLO(weights)
    print(f"[INFO] Running on: {device}")

    # Initialize MediaPipe HandLandmarker
    model_path = "models/hand_landmarker.task"
    if not os.path.exists(model_path):
        import urllib.request
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[INFO] Downloading hand landmarker model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("[INFO] Download complete!")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands_detector = vision.HandLandmarker.create_from_options(options)

    # Camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")

    fps_counter = FPSCounter(window=30)
    smoother    = PredictionSmoother(window=smooth_window)

    paused    = False
    recording = False
    writer    = None
    backend_mode = "MediaPipe" # Highly accurate mode default
    diagnostic_history = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Starting webcam loop. Controls:")
    print("         Q         quit")
    print("         S         save screenshot")
    print("         R         toggle video recording")
    print("         SPACE     pause / resume")
    print("         M         toggle between MediaPipe & YOLOv8 backends")
    print("         +  /  -   raise / lower confidence threshold\n")

    last_frame = None
    smooth_label, smooth_conf = None, None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame read failed, retrying...")
                time.sleep(0.02)
                continue
            last_frame = frame.copy()
        else:
            frame = last_frame.copy() if last_frame is not None else \
                    np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        box_size = 320
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        detections = []
        best_conf, best_label = 0.0, None
        explanation = None
        hand_landmarks = None
        hand_label = None

        if backend_mode == "YOLOv8":
            crop = frame[y1:y2, x1:x2]

            # YOLOv8 inference
            results = model.predict(
                crop,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=img_size,
                device=device,
                verbose=False,
            )

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu())
                    cls_id = int(box.cls[0].cpu())
                    cls_name = ASL_CLASSES[cls_id] if cls_id < len(ASL_CLASSES) else "?"
                    
                    rx1 = x1 + cx1
                    ry1 = y1 + cy1
                    rx2 = x1 + cx2
                    ry2 = y1 + cy2
                    
                    detections.append((rx1, ry1, rx2, ry2, cls_name, conf))
                    if conf > best_conf:
                        best_conf, best_label = conf, cls_name
        else:
            # MediaPipe inference using the task-based HandLandmarker
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            mp_results = hands_detector.detect(mp_image)
            
            if mp_results.hand_landmarks:
                hand_landmarks = mp_results.hand_landmarks[0]
                hand_label = mp_results.handedness[0][0].category_name
                
                # Predict ASL class
                pred_label, pred_conf, pred_exp = NormalizedHandClassifier.classify(
                    hand_landmarks, hand_label
                )
                if pred_label:
                    best_label = pred_label
                    best_conf = pred_conf
                    explanation = pred_exp
                
                # Append diagnostic info for analysis
                try:
                    diagnostic_history.append({
                        "time": time.time(),
                        "hand_label": hand_label,
                        "features": NormalizedHandClassifier.last_features.tolist(),
                        "scores": {k: float(v) for k, v in NormalizedHandClassifier.last_scores.items()},
                        "camera_angle": float(NormalizedHandClassifier.last_camera_angle),
                        "pred_label": pred_label,
                        "pred_conf": float(pred_conf) if pred_conf else 0.0
                    })
                except Exception as e:
                    pass

        # ── temporal smoothing ─────────────────────────────────────────────
        if best_label:
            smooth_label, smooth_conf = smoother.update(best_label, best_conf)
        else:
            smooth_label, smooth_conf = None, None

        fps = fps_counter.tick()

        # ── draw overlay ───────────────────────────────────────────────────
        display = draw_overlay(
            frame, detections, fps, conf_thresh, paused,
            smooth_label, smooth_conf, backend_mode,
            hand_landmarks, hand_label, explanation
        )
        
        # Guide box in YOLOv8 mode
        if backend_mode == "YOLOv8":
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(display, "Place Hand Here", (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # ── recording ──────────────────────────────────────────────────────
        if recording:
            if writer is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                vid_path = str(Path(output_dir) / f"recording_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(vid_path, fourcc, 25,
                                          (display.shape[1], display.shape[0]))
                print(f"[INFO] Recording started: {vid_path}")
            writer.write(display)
            cv2.circle(display, (display.shape[1] - 20, 20), 8, (0, 0, 255), -1)

        cv2.imshow("ASL Detection — Real Time", display)

        # ── keyboard control ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting.")
            break
        elif key == ord(" "):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shot_path = Path(output_dir) / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(shot_path), display)
            print(f"[INFO] Screenshot saved: {shot_path}")
        elif key == ord("r"):
            recording = not recording
            if not recording and writer is not None:
                writer.release()
                writer = None
                print("[INFO] Recording stopped.")
        elif key == ord("m"):
            backend_mode = "YOLOv8" if backend_mode == "MediaPipe" else "MediaPipe"
            print(f"[INFO] Switched backend to: {backend_mode}")
            smoother._history.clear()
        elif key == ord("+") or key == ord("="):
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"[INFO] Confidence threshold: {conf_thresh:.0%}")
        elif key == ord("-"):
            conf_thresh = max(0.05, conf_thresh - 0.05)
            print(f"[INFO] Confidence threshold: {conf_thresh:.0%}")

    # Save diagnostics at exit
    if diagnostic_history:
        import json
        diag_path = Path(output_dir) / "diagnostic_summary.json"
        try:
            with open(diag_path, "w") as f:
                json.dump(diagnostic_history, f, indent=4)
            print(f"[INFO] Saved {len(diagnostic_history)} diagnostic frames to {diag_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save diagnostics: {e}")

    # Cleanup
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam application closed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time ASL Detection webcam application")
    parser.add_argument("--weights",      type=str,   default="models/asl_yolov8n/weights/best.pt")
    parser.add_argument("--camera_index", type=int,   default=0)
    parser.add_argument("--conf",         type=float, default=0.5)
    parser.add_argument("--iou",          type=float, default=0.5)
    parser.add_argument("--img_size",     type=int,   default=640)
    parser.add_argument("--smooth",       type=int,   default=5,
                        help="Temporal smoothing window (frames)")
    parser.add_argument("--output_dir",   type=str,   default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam(
        weights=args.weights,
        camera_index=args.camera_index,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        img_size=args.img_size,
        smooth_window=args.smooth,
        output_dir=args.output_dir,
    )
