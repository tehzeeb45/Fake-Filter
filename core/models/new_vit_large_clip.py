"""New ViT-Large/CLIP wrapper — trained from scratch on DeepShield dataset.

Architecture: timm 'vit_large_patch14_clip_224', num_classes=2 (Real=0, Fake=1).
Checkpoint format: {'state_dict': ..., 'val_auc': ...}
P(Fake) = softmax(logits, dim=1)[:, 1]

Note: This model is ~1.1B parameters and uses 14x14 patches at 224px resolution.
      Use batch_size=16-32 to stay within GPU memory during inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, resolve


class NewViTLargeClip(nn.Module):
    """timm ViT-Large/14 CLIP with 2-class classification head."""

    def __init__(self):
        super().__init__()
        import timm
        self.vit = timm.create_model(
            "vit_large_patch14_clip_224", pretrained=False, num_classes=2
        )
        self.target_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def predict_proba(self, faces) -> np.ndarray:
        """Return per-frame P(Fake) scores as a numpy array (batched, bs=16)."""
        if isinstance(faces, np.ndarray) and faces.ndim == 4:
            all_frames = torch.from_numpy(faces).float()
        else:
            frames = []
            for f in faces:
                t = torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
                frames.append(t)
            all_frames = torch.stack(frames)

        device = next(self.parameters()).device
        all_probs = []
        # Process in small batches to avoid OOM on large ViT
        batch_size = 16
        with torch.no_grad():
            for start in range(0, len(all_frames), batch_size):
                batch = all_frames[start:start + batch_size].to(device)
                logits = self.vit(batch)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs)


def load_new_vit_large_clip(cfg: Config, device=None):
    """Load the new DeepShield-trained ViT-Large/CLIP checkpoint."""
    from . import get_device

    ckpt_path = resolve(cfg.models.new_vit_large_clip.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"new_vit_large_clip checkpoint missing: {ckpt_path}\n"
            "Put best_vit_large_clip.pth in the models/ folder."
        )

    model = NewViTLargeClip()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    missing, unexpected = model.vit.load_state_dict(sd, strict=True)
    if missing:
        raise RuntimeError(f"Missing new_vit_large_clip weights: {missing[:5]}")

    device = device or get_device(cfg.models.device)
    model.to(device).eval()
    return model
