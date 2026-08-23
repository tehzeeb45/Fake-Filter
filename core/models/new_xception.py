"""New XceptionNet wrapper — trained from scratch on DeepShield dataset.

Architecture: timm 'legacy_xception', num_classes=2 (Real=0, Fake=1).
Checkpoint format: {'state_dict': ..., 'val_auc': ...}
P(Fake) = softmax(logits, dim=1)[:, 1]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, resolve


class NewXceptionNet(nn.Module):
    """timm legacy_xception with 2-class head (Real / Fake).

    Keys in the checkpoint are flat timm state_dict keys (no prefix),
    because the model was saved directly via timm.create_model — not
    wrapped in a submodule like the old cnn_xception.py.
    """

    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "legacy_xception", pretrained=False, num_classes=2
        )
        self.target_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, faces: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Return per-frame P(Fake) scores as a numpy array.

        Accepts either a pre-built [N,3,H,W] float32 array or a list of
        HWC uint8 face crops (pre-processed by FacePreprocessor).
        """
        if isinstance(faces, np.ndarray) and faces.ndim == 4:
            tensor = torch.from_numpy(faces).float().to(next(self.parameters()).device)
        else:
            import torchvision.transforms.functional as TF
            frames = []
            for f in faces:
                t = torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
                frames.append(t)
            tensor = torch.stack(frames).to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self.backbone(tensor)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy()

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        """Final conv block for Grad-CAM (FR-17)."""
        return self.backbone.conv4


def load_new_xception(cfg: Config, device=None):
    """Load the new DeepShield-trained XceptionNet checkpoint."""
    from . import get_device

    ckpt_path = resolve(cfg.models.new_xception.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"new_xception checkpoint missing: {ckpt_path}\n"
            "Put best_xception.pth in the models/ folder."
        )

    model = NewXceptionNet()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    missing, unexpected = model.backbone.load_state_dict(sd, strict=True)
    if missing:
        raise RuntimeError(f"Missing new_xception weights: {missing[:5]}")

    device = device or get_device(cfg.models.device)
    model.to(device).eval()
    return model
