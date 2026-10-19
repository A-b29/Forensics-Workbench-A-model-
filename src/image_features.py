"""
image_features.py - Proxy feature extraction for error forensics.

IMPORTANT: Features such as Laplacian variance and contrast are statistical
proxies and heuristic signals, NOT certified clinical image quality metrics.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import cv2


def extract_image_proxies(image_path: Union[str, Path]) -> Dict[str, Optional[float]]:
    """
    Extract deterministic statistical proxies from an image.

    Returns:
        Dict with keys:
            - proxy_mean_intensity: mean grayscale value [0.0, 1.0]
            - proxy_contrast_rms: root-mean-square contrast [0.0, 1.0]
            - proxy_blur_laplacian_var: variance of the Laplacian (higher = sharper)
            - proxy_spatial_entropy: Shannon entropy of the intensity distribution
    """
    path = Path(image_path)
    if not path.is_file():
        return {
            "proxy_mean_intensity": None,
            "proxy_contrast_rms": None,
            "proxy_blur_laplacian_var": None,
            "proxy_spatial_entropy": None,
        }

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "proxy_mean_intensity": None,
            "proxy_contrast_rms": None,
            "proxy_blur_laplacian_var": None,
            "proxy_spatial_entropy": None,
        }

    # Normalize to [0, 1] float
    img_float = img.astype(np.float32) / 255.0

    mean_val = float(np.mean(img_float))
    contrast_rms = float(np.std(img_float))

    # Laplacian variance as a blur proxy
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    blur_var = float(laplacian.var())

    # Spatial intensity entropy
    hist, _ = np.histogram(img, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy_val = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

    return {
        "proxy_mean_intensity": mean_val,
        "proxy_contrast_rms": contrast_rms,
        "proxy_blur_laplacian_var": blur_var,
        "proxy_spatial_entropy": entropy_val,
    }
