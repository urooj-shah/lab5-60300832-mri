#!/usr/bin/env python3
"""
Silver layer: Image feature extraction component.

Inputs:
    --input_data   : path to folder that contains yes/ and no/ subfolders
                     (Azure ML will mount the `tumor_images_raw` data asset here)
    --output_parquet : path where the Parquet file should be written

Output:
    One Parquet file with columns:
        image_id, label, f1, f2, ..., fN

Logs:
    - num_images
    - num_features
    - extraction_time_seconds
    - compute SKU (AZUREML_COMPUTE env var if present)
"""

import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from skimage import io, filters, color
from skimage.filters import rank
from skimage.morphology import disk
from skimage.feature import greycomatrix, greycoprops

# GLCM angles and properties
ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
PROPS = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

# Only process “real” image files
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# ---------- helpers ---------- #

def safe_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize arbitrary numeric image to uint8 [0,255] safely."""
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    # Avoid division by zero, NaNs, etc.
    if not np.isfinite(img_min):
        img_min = 0.0
    if not np.isfinite(img_max) or img_max <= img_min:
        return np.zeros_like(img, dtype=np.uint8)

    img = img - img_min
    img = img / (img_max - img_min)
    img = img * 255.0
    return img.astype(np.uint8)


def load_grayscale(path: Path) -> np.ndarray:
    """Load image as grayscale uint8 0-255 and handle RGB/RGBA."""
    img = io.imread(path)

    # Color / multi-channel image
    if img.ndim == 3:
        # If RGBA (4 channels), drop alpha via rgba2rgb
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)  # returns float rgb
        elif img.shape[2] != 3:
            # Some weird channel count: just take first 3 channels
            img = img[..., :3]

        # Now standard RGB -> grayscale float in [0, 1]
        img = color.rgb2gray(img)
        img = safe_to_uint8(img)

    # Already 2D or single-channel
    else:
        img = safe_to_uint8(img)

    return img


def compute_glcm_features(img: np.ndarray, prefix: str) -> dict:
    """
    Compute GLCM-based features for the given image.

    We force the image to uint8 here so greycomatrix never complains about float
    or other dtypes.
    """
    # ensure uint8 for greycomatrix
    if img.dtype != np.uint8:
        img = safe_to_uint8(img)

    glcm = greycomatrix(
        img,
        distances=[1],
        angles=ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )

    feats = {}
    for prop in PROPS:
        vals = greycoprops(glcm, prop)[0]  # shape (num_angles,)
        feats[f"{prefix}_glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"{prefix}_glcm_{prop}_std"] = float(np.std(vals))
    return feats


def compute_all_features(image_path: Path, label: str) -> dict:
    """Compute all required features for a single image."""
    try:
        gray = load_grayscale(image_path)

        features_dict = {
            "image_id": image_path.name,
            "label": label,
        }

        # --- raw image stats + GLCM ---
        features_dict["raw_mean"] = float(gray.mean())
        features_dict["raw_std"] = float(gray.std())
        features_dict.update(compute_glcm_features(gray, "raw"))

        # --- entropy (rank filter needs uint8) ---
        entropy_img = rank.entropy(gray, disk(3))
        features_dict["entropy_mean"] = float(entropy_img.mean())
        features_dict["entropy_std"] = float(entropy_img.std())
        features_dict.update(compute_glcm_features(entropy_img, "entropy"))

        # --- gaussian ---
        gaussian_img = filters.gaussian(gray, sigma=1)
        gaussian_uint8 = safe_to_uint8(gaussian_img)
        features_dict["gaussian_mean"] = float(gaussian_uint8.mean())
        features_dict["gaussian_std"] = float(gaussian_uint8.std())
        features_dict.update(compute_glcm_features(gaussian_uint8, "gaussian"))

        # --- sobel ---
        sobel_img = filters.sobel(gray)
        sobel_uint8 = safe_to_uint8(sobel_img)
        features_dict["sobel_mean"] = float(sobel_uint8.mean())
        features_dict["sobel_std"] = float(sobel_uint8.std())
        features_dict.update(compute_glcm_features(sobel_uint8, "sobel"))

        # --- prewitt ---
        prewitt_img = filters.prewitt(gray)
        prewitt_uint8 = safe_to_uint8(prewitt_img)
        features_dict["prewitt_mean"] = float(prewitt_uint8.mean())
        features_dict["prewitt_std"] = float(prewitt_uint8.std())
        features_dict.update(compute_glcm_features(prewitt_uint8, "prewitt"))

        # --- gabor (real + imag) ---
        gabor_real, gabor_imag = filters.gabor(gray, frequency=0.2)
        gabor_real_u8 = safe_to_uint8(gabor_real)
        gabor_imag_u8 = safe_to_uint8(gabor_imag)

        features_dict["gabor_real_mean"] = float(gabor_real_u8.mean())
        features_dict["gabor_real_std"] = float(gabor_real_u8.std())
        features_dict.update(compute_glcm_features(gabor_real_u8, "gabor_real"))

        features_dict["gabor_imag_mean"] = float(gabor_imag_u8.mean())
        features_dict["gabor_imag_std"] = float(gabor_imag_u8.std())
        features_dict.update(compute_glcm_features(gabor_imag_u8, "gabor_imag"))

        # --- hessian ---
        try:
            hess = filters.hessian(gray)

            # hess might be a tuple/list of arrays or a single ndarray.
            if isinstance(hess, (tuple, list)):
                # stack along new axis and take L2 norm across that axis
                arr = np.stack(hess, axis=0).astype(np.float32)
                hessian_mag = np.linalg.norm(arr, axis=0)
            else:
                # already some kind of magnitude / tensor; just take absolute value
                hessian_mag = np.abs(hess.astype(np.float32))

            hessian_uint8 = safe_to_uint8(hessian_mag)
            features_dict["hessian_mean"] = float(hessian_uint8.mean())
            features_dict["hessian_std"] = float(hessian_uint8.std())
            features_dict.update(compute_glcm_features(hessian_uint8, "hessian"))

        except Exception as e:
            # Don't kill the whole feature set if Hessian fails – just log a warning.
            print(f"[WARN] Hessian failed for {image_path}: {e}")

        return features_dict

    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}")
        return {}


# ---------- CLI + orchestration ---------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Extract image features for MRI tumor dataset.")
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to input folder (mounted tumor_images_raw data asset)."
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        required=True,
        help="Path to output parquet file or directory."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers for multiprocessing."
    )
    return parser.parse_args()


def gather_image_paths(root: Path):
    """
    Expect structure:
        root/yes/*.*
        root/no/*.*
    Only include common image file extensions.
    """
    image_entries = []
    for label in ["yes", "no"]:
        label_dir = root / label
        if not label_dir.exists():
            print(f"[WARN] Label directory not found: {label_dir}")
            continue
        for p in sorted(label_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXT:
                image_entries.append((p, label))
    return image_entries


def main():
    args = parse_args()
    input_root = Path(args.input_data)
    output_path = Path(args.output_parquet)

    # If AzureML gives us a directory for the output, write a file inside it
    if output_path.is_dir() or (not output_path.suffix):
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "features.parquet"

    print(f"[INFO] Input root: {input_root}")
    print(f"[INFO] Output parquet: {output_path}")

    images = gather_image_paths(input_root)
    num_images = len(images)
    print(f"[INFO] Found {num_images} images.")

    if num_images == 0:
        raise RuntimeError("No images found under input_data (expected yes/ and no/ folders).")

    start_time = time.time()

    records = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {
            executor.submit(compute_all_features, path, label): (path, label)
            for path, label in images
        }
        for future in as_completed(future_to_path):
            res = future.result()
            if res:
                records.append(res)

    end_time = time.time()
    extraction_time_seconds = end_time - start_time

    if len(records) == 0:
        print("[WARN] No features were computed for any image. Output will be empty.")
        df = pd.DataFrame(columns=["image_id", "label"])
    else:
        df = pd.DataFrame(records)

    num_features = max(df.shape[1] - 2, 0)  # exclude image_id and label

    print(f"[INFO] num_images = {num_images}")
    print(f"[INFO] num_features = {num_features}")
    print(f"[INFO] extraction_time_seconds = {extraction_time_seconds:.3f}")

    compute_sku = os.environ.get("AZUREML_COMPUTE", "unknown")
    print(f"[INFO] compute SKU (AZUREML_COMPUTE) = {compute_sku}")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved features parquet to {output_path}")


if __name__ == "__main__":
    main()
