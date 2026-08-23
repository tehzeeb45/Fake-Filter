"""New ViT-Small wrapper — trained from scratch on DeepShield dataset.

Architecture: timm 'vit_small_patch16_224', num_classes=2 (Real=0, Fake=1).
Checkpoint format: {'state_dict': ..., 'val_auc': ...}
P(Fake) = softmax(logits, dim=1)[:, 1]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, resolve


class NewViTSmall(nn.Module):
    """timm ViT-Small/16 with 2-class classification head."""

    def __init__(self):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "vit_small_patch16_224", pretrained=False, num_classes=2
        )
        self.target_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def predict_proba(self, faces) -> np.ndarray:
        """Return per-frame P(Fake) scores as a numpy array."""
        if isinstance(faces, np.ndarray) and faces.ndim == 4:
            tensor = torch.from_numpy(faces).float().to(next(self.parameters()).device)
        else:
            frames = []
            for f in faces:
                t = torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
                frames.append(t)
            tensor = torch.stack(frames).to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self.vit(tensor)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy()


def load_new_vit_small(cfg: Config, device=None):
    """Load the new DeepShield-trained ViT-Small checkpoint."""
    from . import get_device

    ckpt_path = resolve(cfg.models.new_vit_small.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"new_vit_small checkpoint missing: {ckpt_path}\n"
            "Put best_vit_small.pth in the models/ folder."
        )

    model = NewViTSmall()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    missing, unexpected = model.vit.load_state_dict(sd, strict=True)
    if missing:
        raise RuntimeError(f"Missing new_vit_small weights: {missing[:5]}")

    device = device or get_device(cfg.models.device)
    model.to(device).eval()
    return model
