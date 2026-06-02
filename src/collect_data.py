"""
collect_data.py
---------------
Interactive webcam-based data collection tool for ASL dataset creation.

Usage:
    python src/collect_data.py --class_name A --num_images 100 --output_dir dataset/images/raw
"""

import cv2
import os
import argparse
import time
from pathlib import Path

ASL_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect ASL dataset images via webcam")
    parser.add_argument("--class_name", type=str, required=True,
                        choices=ASL_CLASSES, help="ASL class to collect")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Number of images to collect per class")
    parser.add_argument("--output_dir", type=str, default="dataset/images/raw",
                        help="Output directory for raw images")
    parser.add_argument("--camera_index", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between captures (seconds)")
    return parser.parse_args()


def collect_images(class_name: str, num_images: int, output_dir: str,
                   camera_index: int = 0, delay: float = 0.1):
    """
    Collect images for a single ASL class using webcam.

    Args:
        class_name: ASL sign label (e.g., 'A')
        num_images: Number of images to capture
        output_dir: Root output directory
        camera_index: Webcam index
        delay: Seconds between auto-captures
    """
    save_dir = Path(output_dir) / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    collected = 0
    auto_capture = False
    print(f"\n[INFO] Collecting class: {class_name}")
    print("[INFO] Controls:")
    print("         SPACE  - capture one image manually")
    print("         A      - toggle auto-capture mode")
    print("         Q      - quit / finish class")

    while collected < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame, retrying...")
            continue

        display = frame.copy()

        # HUD overlay
        status = "AUTO" if auto_capture else "MANUAL"
        cv2.putText(display, f"Class: {class_name}  [{collected}/{num_images}]  Mode: {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=capture | A=auto | Q=quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw guide box
        h, w = frame.shape[:2]
        box_x1, box_y1 = w // 4, h // 4
        box_x2, box_y2 = 3 * w // 4, 3 * h // 4
        cv2.rectangle(display, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)
        cv2.putText(display, "Place hand here", (box_x1 + 5, box_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow(f"ASL Data Collector - {class_name}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print(f"[INFO] Stopping early. Collected {collected} images.")
            break
        elif key == ord("a"):
            auto_capture = not auto_capture
            print(f"[INFO] Auto-capture: {'ON' if auto_capture else 'OFF'}")
        elif key == ord(" ") or auto_capture:
            if key == ord(" ") or auto_capture:
                timestamp = int(time.time() * 1000)
                filename = save_dir / f"{class_name}_{timestamp}_{collected:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                collected += 1
                print(f"[INFO] Saved: {filename.name}  ({collected}/{num_images})")
                if auto_capture:
                    time.sleep(delay)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Collected {collected} images for class '{class_name}' -> {save_dir}")
    return collected


def collect_all_classes(num_images: int = 100, output_dir: str = "dataset/images/raw",
                        camera_index: int = 0):
    """Collect images for all 12 ASL classes sequentially."""
    summary = {}
    for cls in ASL_CLASSES:
        print(f"\n{'='*50}")
        print(f"  Ready to collect class: {cls}")
        print(f"  Press ENTER when ready, or type 'skip' to skip.")
        user_input = input("  > ").strip().lower()
        if user_input == "skip":
            print(f"[SKIP] Skipping class {cls}")
            continue
        count = collect_images(cls, num_images, output_dir, camera_index)
        summary[cls] = count

    print("\n[SUMMARY]")
    for cls, count in summary.items():
        print(f"  {cls}: {count} images")
    return summary


if __name__ == "__main__":
    args = parse_args()
    collect_images(
        class_name=args.class_name,
        num_images=args.num_images,
        output_dir=args.output_dir,
        camera_index=args.camera_index,
        delay=args.delay,
    )
