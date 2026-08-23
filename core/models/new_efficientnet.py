"""New EfficientNet-B3 wrapper — trained from scratch on DeepShield dataset.

Architecture: torchvision efficientnet_b3, classifier replaced with
    Linear(in_features, 2)  → 2-class (Real=0, Fake=1)
Checkpoint format: {'state_dict': ..., 'val_auc': ...}
P(Fake) = softmax(logits, dim=1)[:, 1]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, resolve


class NewEfficientNetB3(nn.Module):
    """torchvision EfficientNet-B3 with a 2-class classifier head."""

    def __init__(self):
        super().__init__()
        import torchvision.models as tv_models
        base = tv_models.efficientnet_b3(weights=None)
        in_f = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_f, 2)
        self.net = base
        self.target_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
            logits = self.net(tensor)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy()

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        return self.net.features[-1]


def load_new_efficientnet(cfg: Config, device=None):
    """Load the new DeepShield-trained EfficientNet-B3 checkpoint."""
    from . import get_device

    ckpt_path = resolve(cfg.models.new_efficientnet.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"new_efficientnet checkpoint missing: {ckpt_path}\n"
            "Put best_efficientnet_b3.pth in the models/ folder."
        )

    model = NewEfficientNetB3()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    missing, unexpected = model.net.load_state_dict(sd, strict=True)
    if missing:
        raise RuntimeError(f"Missing new_efficientnet weights: {missing[:5]}")

    device = device or get_device(cfg.models.device)
    model.to(device).eval()
    return model
