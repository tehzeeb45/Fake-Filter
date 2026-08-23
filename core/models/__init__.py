"""Model wrappers — 4 new DeepShield-trained models:
    - NewXceptionNet    (timm legacy_xception, 2-class)
    - NewEfficientNetB3 (torchvision efficientnet_b3, 2-class)
    - NewViTSmall       (timm vit_small_patch16_224, 2-class)
    - NewViTLargeClip   (timm vit_large_patch14_clip_224, 2-class)

All trained on the DeepShield NPZ dataset with video-level split and
equal-weight soft voting ensemble.

Legacy wrappers (cnn_xception, cnn_efficientnet, vit_community, vit_lnclip,
vit_vision, lstm_temporal) are retained on disk but not used by the new
ModelBundle.
"""
from __future__ import annotations

import torch

from . import new_xception, new_efficientnet, new_vit_small, new_vit_large_clip


def get_device(value: str = "auto") -> torch.device:
    """Resolve the inference device (NFR-06: runs on CPU too)."""
    if value == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but not available"
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelBundle:
    """Container for the 4 new DeepShield-trained models + device."""

    def __init__(self,
                 xception=None,
                 efficientnet=None,
                 vit_small=None,
                 vit_large_clip=None,
                 device=None):
        self.xception = xception
        self.efficientnet = efficientnet
        self.vit_small = vit_small
        self.vit_large_clip = vit_large_clip
        self.device = device

    def eval_all(self):
        for m in (self.xception, self.efficientnet,
                  self.vit_small, self.vit_large_clip):
            if m is not None:
                m.eval()

    def __repr__(self):
        parts = []
        if self.xception is not None:
            parts.append(f"xception={type(self.xception).__name__}")
        if self.efficientnet is not None:
            parts.append(f"efficientnet={type(self.efficientnet).__name__}")
        if self.vit_small is not None:
            parts.append(f"vit_small={type(self.vit_small).__name__}")
        if self.vit_large_clip is not None:
            parts.append(f"vit_large_clip={type(self.vit_large_clip).__name__}")
        parts.append(f"device={self.device}")
        return f"ModelBundle({', '.join(parts)})"
