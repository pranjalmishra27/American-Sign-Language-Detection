"""
augment.py
----------
Offline data augmentation pipeline to expand the raw dataset
from ~100 images/class to 500-700 images/class.

Applies a combination of geometric and photometric augmentations
to ensure robustness under varying lighting, backgrounds, and
camera positions.

Usage:
    python src/augment.py \
        --input_dir  dataset/images/raw \
        --output_dir dataset/images/augmented \
        --target_per_class 600
"""

import cv2
import numpy as np
import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ASL_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


# ─────────────────────────────────────────────────────────────────────────────
# Individual augmentation functions
# Each function accepts and returns a BGR numpy array (H, W, 3).
# ─────────────────────────────────────────────────────────────────────────────

def random_rotation(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate image by a random angle within ±max_angle degrees."""
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_translation(image: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Translate image by up to max_shift * image_dimension pixels."""
    h, w = image.shape[:2]
    tx = random.uniform(-max_shift * w, max_shift * w)
    ty = random.uniform(-max_shift * h, max_shift * h)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_scale(image: np.ndarray,
                 scale_range: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """Scale image then crop/pad to original size."""
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    if scale > 1.0:
        x_start = (new_w - w) // 2
        y_start = (new_h - h) // 2
        return resized[y_start:y_start + h, x_start:x_start + w]
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas


def random_crop(image: np.ndarray, crop_frac: float = 0.9) -> np.ndarray:
    """Randomly crop a fraction of the image then resize back."""
    h, w = image.shape[:2]
    crop_h = int(h * crop_frac)
    crop_w = int(w * crop_frac)
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    cropped = image[y:y + crop_h, x:x + crop_w]
    return cv2.resize(cropped, (w, h))


def perspective_transform(image: np.ndarray, distortion: float = 0.05) -> np.ndarray:
    """Apply a subtle random perspective warp."""
    h, w = image.shape[:2]
    d = distortion
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = src + np.float32([
        [random.uniform(-d * w, d * w), random.uniform(-d * h, d * h)],
        [random.uniform(-d * w, d * w), random.uniform(-d * h, d * h)],
        [random.uniform(-d * w, d * w), random.uniform(-d * h, d * h)],
        [random.uniform(-d * w, d * w), random.uniform(-d * h, d * h)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def brightness_adjust(image: np.ndarray,
                       factor_range: Tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
    """Randomly adjust brightness via HSV value channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(*factor_range)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def contrast_adjust(image: np.ndarray,
                     alpha_range: Tuple[float, float] = (0.6, 1.4)) -> np.ndarray:
    """Randomly adjust contrast: output = alpha * (input - mean) + mean."""
    alpha = random.uniform(*alpha_range)
    mean = image.mean()
    adjusted = alpha * (image.astype(np.float32) - mean) + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray,
                  kernel_range: Tuple[int, int] = (3, 7)) -> np.ndarray:
    """Apply Gaussian blur with a random odd kernel size."""
    k = random.choice([k for k in range(kernel_range[0], kernel_range[1] + 1, 2)])
    return cv2.GaussianBlur(image, (k, k), 0)


def motion_blur(image: np.ndarray, max_kernel: int = 9) -> np.ndarray:
    """Simulate motion blur with a random directional kernel."""
    k = random.choice(range(3, max_kernel + 1, 2))
    kernel = np.zeros((k, k))
    if random.random() > 0.5:
        kernel[k // 2, :] = 1.0  # horizontal
    else:
        kernel[:, k // 2] = 1.0  # vertical
    kernel /= k
    return cv2.filter2D(image, -1, kernel)


def color_jitter(image: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Randomly shift each BGR channel independently."""
    result = image.astype(np.float32)
    for c in range(3):
        shift = random.uniform(-strength * 127, strength * 127)
        result[:, :, c] = np.clip(result[:, :, c] + shift, 0, 255)
    return result.astype(np.uint8)


def shadow_simulation(image: np.ndarray) -> np.ndarray:
    """Overlay a semi-transparent triangular shadow on a random half."""
    h, w = image.shape[:2]
    shadow_mask = np.zeros_like(image, dtype=np.float32)
    x1, x2 = sorted(random.sample(range(w), 2))
    poly = np.array([[0, 0], [x1, 0], [x2, h], [0, h]])
    cv2.fillPoly(shadow_mask, [poly], (1, 1, 1))
    factor = random.uniform(0.4, 0.7)
    blended = image.astype(np.float32) * (1 - shadow_mask * (1 - factor))
    return np.clip(blended, 0, 255).astype(np.uint8)


def gaussian_noise(image: np.ndarray, std_range: Tuple[float, float] = (5, 25)) -> np.ndarray:
    """Add Gaussian noise."""
    std = random.uniform(*std_range)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Mirror image horizontally (use with caution for ASL — some signs are handed)."""
    return cv2.flip(image, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Probability each transform is applied
AUGMENTATION_PIPELINE = [
    (random_rotation,       0.7),
    (random_translation,    0.5),
    (random_scale,          0.5),
    (random_crop,           0.5),
    (perspective_transform, 0.4),
    (brightness_adjust,     0.8),
    (contrast_adjust,       0.6),
    (color_jitter,          0.5),
    (gaussian_blur,         0.3),
    (motion_blur,           0.2),
    (shadow_simulation,     0.3),
    (gaussian_noise,        0.4),
]


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply a random subset of augmentations to a single image."""
    for fn, prob in AUGMENTATION_PIPELINE:
        if random.random() < prob:
            image = fn(image)
    return image


# ─────────────────────────────────────────────────────────────────────────────
# Main augmentation loop
# ─────────────────────────────────────────────────────────────────────────────

def augment_class(class_name: str, input_dir: str, output_dir: str,
                  target_count: int = 600) -> int:
    """
    Augment images for one ASL class until target_count images exist.

    Returns the number of augmented images generated.
    """
    src_dir = Path(input_dir) / class_name
    dst_dir = Path(output_dir) / class_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    source_images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    if not source_images:
        print(f"[WARNING] No images found for class {class_name} in {src_dir}")
        return 0

    # Copy originals first
    for img_path in source_images:
        dest = dst_dir / img_path.name
        if not dest.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(dest), img)

    existing = list(dst_dir.glob("*.jpg")) + list(dst_dir.glob("*.png"))
    augmented_count = 0

    while len(existing) + augmented_count < target_count:
        src_path = random.choice(source_images)
        img = cv2.imread(str(src_path))
        if img is None:
            continue
        aug_img = augment_image(img)
        out_name = dst_dir / f"{class_name}_aug_{augmented_count:05d}.jpg"
        cv2.imwrite(str(out_name), aug_img)
        augmented_count += 1

    total = len(list(dst_dir.glob("*.jpg"))) + len(list(dst_dir.glob("*.png")))
    print(f"[INFO] {class_name}: {len(source_images)} originals -> "
          f"{augmented_count} augmented -> {total} total")
    return augmented_count


def augment_all_classes(input_dir: str, output_dir: str, target_per_class: int = 600):
    """Augment all 12 ASL classes."""
    print(f"\n[INFO] Augmenting dataset: target {target_per_class} images/class")
    summary = {}
    for cls in ASL_CLASSES:
        count = augment_class(cls, input_dir, output_dir, target_per_class)
        summary[cls] = count

    print("\n[SUMMARY] Augmentation complete:")
    total = 0
    for cls, count in summary.items():
        print(f"  {cls}: +{count} augmented images")
        total += count
    print(f"\n  Total augmented images generated: {total}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="ASL dataset augmentation pipeline")
    parser.add_argument("--input_dir", type=str, default="dataset/images/raw")
    parser.add_argument("--output_dir", type=str, default="dataset/images/augmented")
    parser.add_argument("--target_per_class", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment_all_classes(args.input_dir, args.output_dir, args.target_per_class)
