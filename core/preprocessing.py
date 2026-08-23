"""Preprocessing pipeline (P2.0 / Level-2 sub-processes P2.1..P2.4).

Implements:
  FR-05  video frame extraction at configurable FPS (default 5)
  FR-06  face detection & localisation (MTCNN, conf >= 0.95)
  FR-07  no-face error raised to caller
  FR-08  face alignment (eyes horizontal) with margin around bounding box
  FR-09  resize to 224x224
  FR-10  ImageNet mean/std normalisation

Frame extraction tries, in order:
  1. FFmpeg on system PATH
  2. bundled ffmpeg binary shipped by `imageio-ffmpeg`
  3. OpenCV VideoCapture
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import Config

try:
    import imageio_ffmpeg as _iio_ffmpeg
    _HAS_IIO_FFMPEG = True
except Exception:  # pragma: no cover
    _HAS_IIO_FFMPEG = False


def find_ffmpeg() -> str | None:
    """Return a usable ffmpeg executable path, or None."""
    for probe in ("ffmpeg", "ffmpeg.exe"):
        try:
            r = subprocess.run([probe, "-version"], capture_output=True, timeout=15)
            if r.returncode == 0:
                return probe
        except Exception:
            pass
    if _HAS_IIO_FFMPEG:
        try:
            exe = _iio_ffmpeg.get_ffmpeg_exe()
            if Path(exe).is_file():
                return exe
        except Exception:
            pass
    return None


FFMPEG = find_ffmpeg()


class NoFaceError(Exception):
    """FR-07: no face detected in the uploaded media."""


def extract_video_frames(video_path: str | Path, fps: float = 5.0,
                         max_seconds: float | None = 60.0,
                         max_frames: int = 24,
                         tmp_dir: str | Path | None = None) -> list[Path]:
    """Extract at most `max_frames` uniformly-spaced frames from a video.

    FR-05: extraction rate is configurable via `fps`.
    FR-02: only the first `max_seconds` are considered (v1.0 limit).
    Returns a list of PNG frame paths (already downsampled to `max_frames`).
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    tmp_root = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="df_frames_"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    frames = _extract_all_frames(video_path, fps, max_seconds, tmp_root)

    # uniform subsample down to max_frames (keeps temporal spread)
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames


def _extract_all_frames(video_path: Path, fps: float, max_seconds: float | None,
                        out_dir: Path) -> list[Path]:
    if FFMPEG:
        return _extract_ffmpeg(video_path, fps, max_seconds, out_dir)
    return _extract_opencv(video_path, fps, max_seconds, out_dir)


def _extract_ffmpeg(video_path: Path, fps: float, max_seconds: float | None,
                    out_dir: Path) -> list[Path]:
    """Frame extraction via ffmpeg -vf fps=N (FR-05 default 5 fps)."""
    import re

    pattern = str(out_dir / "frame_%06d.png")
    args = [FFMPEG, "-y", "-i", str(video_path), "-vf", f"fps={fps}"]
    if max_seconds is not None:
        args += ["-t", str(max_seconds)]
    args += ["-q:v", "2", pattern]
    subprocess.run(args, capture_output=True, timeout=600)

    files = sorted(out_dir.glob("frame_*.png"),
                   key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    return files


def _extract_opencv(video_path: Path, fps: float, max_seconds: float | None,
                    out_dir: Path) -> list[Path]:
    """OpenCV VideoCapture fallback when no ffmpeg binary is available."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1.0, round(src_fps / max(fps, 0.1)))
        step = int(interval)
        frames: list[Path] = []
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_seconds is not None and n / max(src_fps, 1.0) > max_seconds:
                break
            if n % step == 0:
                out = out_dir / f"frame_{len(frames) + 1:06d}.png"
                cv2.imwrite(str(out), frame)
                frames.append(out)
            n += 1
        return frames
    finally:
        cap.release()


class FacePreprocessor:
    """MTCNN face detection, alignment and tensor normalisation (P2.2-P2.4)."""

    def __init__(self, cfg: Config):
        from facenet_pytorch import MTCNN  # lazy import keeps startup light

        prep = cfg.preprocessing
        self.target_size = int(prep.target_size)
        self.margin = float(prep.face_margin)
        self.conf_threshold = float(prep.mtcnn_confidence)
        self.normalize_mode = getattr(prep, "normalize_mode", "scale_01")
        self.mean = np.array(prep.imagenet_mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(prep.imagenet_std, dtype=np.float32).reshape(3, 1, 1)
        self._mtcnn = MTCNN(keep_all=False, post_process=False, device="cpu")

    # ------------------------------------------------------------------ faces
    def detect_face(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (aligned_face_rgb, box_xyxy, confidence) for the largest face.

        Raises NoFaceError when no face passes the confidence gate (FR-07).
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self._mtcnn.detect(rgb, landmarks=True)

        if boxes is None or len(boxes) == 0:
            raise NoFaceError(
                "No face detected. Please upload media containing a clearly visible human face."
            )

        best = 0
        best_conf = float(probs[best])
        for i in range(1, len(boxes)):
            if float(probs[i]) > best_conf:
                best, best_conf = i, float(probs[i])
        if best_conf < self.conf_threshold:
            raise NoFaceError(
                "No face detected. Please upload media containing a clearly visible human face."
            )

        box = boxes[best].astype(int)
        lm = landmarks[best] if landmarks is not None else None
        face = self._align_and_crop(rgb, box, lm)
        return face, box, best_conf

    def _align_and_crop(self, rgb: np.ndarray, box: np.ndarray,
                        landmarks) -> np.ndarray:
        """FR-08: rotate so eyes are horizontal, then crop with margin."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        margin_x = int(w * self.margin)
        margin_y = int(h * self.margin)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(rgb.shape[1] - 1, x2 + margin_x)
        y2 = min(rgb.shape[0] - 1, y2 + margin_y)

        crop = rgb[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            raise NoFaceError("Face crop out of bounds.")

        if landmarks is not None:
            crop = self._rotate_eyes_horizontal(rgb, landmarks, x1, y1, crop)

        return cv2.resize(crop, (self.target_size, self.target_size),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _rotate_eyes_horizontal(rgb, landmarks, x1, y1, crop) -> np.ndarray:
        """Rotate around the face center so the eye landmarks are level."""
        try:
            left = landmarks[0] - np.array([x1, y1])
            right = landmarks[1] - np.array([x1, y1])
            dy = right[1] - left[1]
            dx = right[0] - left[0]
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) < 0.5:
                return crop
            h, w = crop.shape[:2]
            center = (w / 2.0, h / 2.0)
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(crop, m, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return crop

    # ------------------------------------------------------------- tensors
    def face_to_tensor(self, face_rgb: np.ndarray) -> np.ndarray:
        """FR-09/FR-10: uint8 RGB -> float32 CHW [0..1] (scale_01) or ImageNet normalised."""
        img = face_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        if self.normalize_mode == "imagenet":
            img = (img - self.mean) / self.std
        return np.ascontiguousarray(img)

    def tensor_for_vis(self, face_rgb: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(face_rgb)  # original uint8 crop for overlay


def tensor_to_vis(t: np.ndarray, mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """Inverse ImageNet normalisation (used for Grad-CAM overlays)."""
    img = np.transpose(t, (1, 2, 0)) * np.array(std) + np.array(mean)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img
