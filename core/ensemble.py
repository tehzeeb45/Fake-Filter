"""Ensemble classifier - weighted soft voting with consensus anomaly fusion (FR-14..FR-16).

Combines CNNs and Vision Transformers with dual-evidence anomaly consensus:
1. Calculates baseline weighted average.
2. For static images:
   - Detects generative/inpainting consensus (e.g. EfficientNet + ViT-Large anomaly agreement).
   - Protects authentic photos against single-model false-alarm spikes.
3. Produces final verdict and confidence score.
"""
from __future__ import annotations

from .config import Config


def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of P(Fake) over the models actually present in `scores`."""
    total_w = 0.0
    acc = 0.0
    for key, p in scores.items():
        w = weights.get(key, 0.0)
        acc += w * p
        total_w += w
    if total_w <= 0:
        raise ValueError("ensemble: no weighted model scores provided")
    return acc / total_w


def consensus_fusion(scores: dict[str, float], weights: dict[str, float], threshold: float, kind: str, forensic_score: float = 0.0) -> float:
    """Multi-model consensus fusion for static image and video deepfake forensics."""
    p_base = weighted_average(scores, weights)

    # Model scores
    effnet = scores.get("new_efficientnet", 0.0)
    vit_large = scores.get("new_vit_large_clip", 0.0)
    xception = scores.get("new_xception", 0.0)
    vit_small = scores.get("new_vit_small", 0.0)

    if kind == "video":
        # 1. Genuine Video Protection (Resolves video compression and motion blur false alarms):
        # When facial consistency anchor (EfficientNet) confirms authentic camera skin (< 0.20)
        if effnet < 0.20:
            p_real_video = (effnet * 0.55 + xception * 0.20 + vit_small * 0.15 + vit_large * 0.10)
            return min(p_base, p_real_video)

        # 2. Multi-Model Video Deepfake Consensus (2+ models strongly flagging manipulation >= 0.60):
        high_video = [s for s in [effnet, vit_large, xception, vit_small] if s >= 0.60]
        if len(high_video) >= 2:
            p_multi_video = sum(high_video) / len(high_video)
            return max(p_base, p_multi_video)

        return p_base

    # 1. Real Human Photo Protection (Resolves ethnic clothing / celebrity / indoor lighting false alarms):
    # Key insight: Real humans: ViT-Small 12%-49%. AI fakes: ViT-Small < 12% (ChatGPT) or >= 50% (Midjourney v6)
    if effnet < 0.08 and xception < 0.08 and 0.12 <= vit_small < 0.50:
        p_real = (effnet * 0.50 + xception * 0.30 + vit_small * 0.20)
        return min(p_base, p_real)

    # 2. Strong Multimodal AI / Prompt Diffusion Detection (ViT-Large CLIP lead):
    if vit_large >= 0.50:
        p_vit = vit_large * 0.85 + max(effnet, xception, vit_small) * 0.15
        return max(p_base, p_vit)

    # 3. GAN & Generative Facial Consensus (EffNet >= 0.35 with supporting signal or forensic boost):
    if (effnet >= 0.35 and (xception >= 0.15 or vit_large >= 0.20 or vit_small >= 0.20)) or (effnet >= 0.40 and forensic_score >= 0.30):
        p_gan = max(effnet, xception) * 0.75 + p_base * 0.25
        return max(p_base, p_gan)

    # 4. Dual-CNN Consensus (EffNet + Xception >= 0.30):
    if effnet >= 0.30 and xception >= 0.30:
        p_cnn = (effnet + xception) / 2.0
        return max(p_base, p_cnn)

    # 5. Multi-Model Agreement (Any 2+ models detecting fake >= 0.45):
    high_votes = [s for s in [effnet, vit_large, xception, vit_small] if s >= 0.45]
    if len(high_votes) >= 2:
        p_multi = sum(high_votes) / len(high_votes)
        return max(p_base, p_multi)

    # 6. Physics-based Forensic Residual Boost:
    if forensic_score >= 0.50:
        max_nn = max(effnet, vit_large, xception, vit_small)
        if max_nn >= 0.25:
            p_synth = max(p_base, (forensic_score * 0.4 + max_nn * 0.6))
            return max(p_base, p_synth)

    return p_base


class Ensemble:
    """Configured weighted-soft-voting helper with consensus anomaly fusion."""

    def __init__(self, cfg: Config):
        default_thresh = float(cfg.ensemble.get("threshold", 0.50))
        self.image_threshold = float(cfg.ensemble.get("image_threshold", 0.45))
        self.video_threshold = float(cfg.ensemble.get("video_threshold", default_thresh))
        self.image_weights = {k: float(v) for k, v in cfg.ensemble.image_weights.items()}
        self.video_weights = {k: float(v) for k, v in cfg.ensemble.video_weights.items()}

    def combine(self, scores: dict[str, float], kind: str, forensic_score: float = 0.0) -> dict:
        weights = self.video_weights if kind == "video" else self.image_weights
        threshold = self.video_threshold if kind == "video" else self.image_threshold
        p_fake = consensus_fusion(scores, weights, threshold, kind, forensic_score=forensic_score)
        result = self.to_verdict(p_fake, threshold=threshold)
        result["scores"] = {k: round(v, 4) for k, v in scores.items()}
        result["disagreement"] = False
        return result

    def to_verdict(self, p_fake: float, threshold: float | None = None) -> dict:
        thresh = self.image_threshold if threshold is None else threshold
        return {
            "verdict": "FAKE" if p_fake >= thresh else "REAL",
            "p_fake": round(float(p_fake), 4),
            "confidence": round(
                p_fake * 100.0 if p_fake >= thresh else (1.0 - p_fake) * 100.0,
                2,
            ),
            "threshold": thresh,
        }