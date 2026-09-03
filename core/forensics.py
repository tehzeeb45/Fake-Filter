"""Physics-based Digital Image Forensics (DIF) module.

Analyzes high-frequency sensor noise, chromatic gradient variance, and
compression residuals to detect prompt-generated diffusion images (Midjourney,
DALL-E 3, Flux, Stable Diffusion) alongside neural network deepfake detectors.
"""
from __future__ import annotations

import cv2
import numpy as np


def analyze_frequency_and_noise(bgr_image: np.ndarray, face_crop: np.ndarray | None = None) -> float:
    """Analyze image and face crop for synthetic diffusion noise signatures.

    Returns:
        float: synthetic probability between 0.0 (genuine camera sensor noise)
               and 1.0 (synthetic latent diffusion smoothness).
    """
    if bgr_image is None or bgr_image.size == 0:
        return 0.0

    target = face_crop if (face_crop is not None and face_crop.size > 0) else bgr_image
    
    # Ensure RGB / BGR consistency and size
    if target.shape[0] < 32 or target.shape[1] < 32:
        return 0.0

    # 1. Chrominance Noise Variance (Cr and Cb planes in YCrCb color space)
    # Real camera sensors exhibit natural photon shot noise across color channels.
    # Latent diffusion models exhibit over-smoothed chromatic gradients.
    try:
        ycrcb = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)

        lap_cr = cv2.Laplacian(cr, cv2.CV_32F)
        lap_cb = cv2.Laplacian(cb, cv2.CV_32F)

        var_cr = float(np.var(lap_cr))
        var_cb = float(np.var(lap_cb))
        chroma_noise_metric = (var_cr + var_cb) / 2.0
    except Exception:
        chroma_noise_metric = 50.0

    # 2. Error Level Analysis (ELA) Simulation
    # Checks discrepancy between spatial pixel structure and natural DCT compression
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        _, enc = cv2.imencode(".jpg", target, encode_param)
        res = cv2.imdecode(enc, 1)
        diff = cv2.absdiff(target, res).astype(np.float32)
        ela_mean = float(np.mean(diff))
    except Exception:
        ela_mean = 3.0

    # 3. High-Frequency Spectral Ratio (FFT)
    try:
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
        resized_gray = cv2.resize(gray, (224, 224)).astype(np.float32)
        f_transform = np.fft.fft2(resized_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Center (low frequency) vs Peripheral (high frequency) energy
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        radius = 28
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        
        low_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum) + 1e-6
        high_freq_ratio = float(1.0 - (low_freq_energy / total_energy))
    except Exception:
        high_freq_ratio = 0.50

    # Synthetic Likelihood Heuristic:
    # Diffusion models typically have high perceptual detail but low cross-channel sensor noise variance (< 35)
    # or elevated uniform ELA error with low high-frequency chromatic variance.
    synthetic_score = 0.0
    if chroma_noise_metric < 25.0:
        synthetic_score += 0.40
    elif chroma_noise_metric < 45.0:
        synthetic_score += 0.20

    if ela_mean > 4.5:
        synthetic_score += 0.35
    elif ela_mean > 3.2:
        synthetic_score += 0.20

    if high_freq_ratio > 0.65:
        synthetic_score += 0.25

    return float(np.clip(synthetic_score, 0.0, 1.0))
