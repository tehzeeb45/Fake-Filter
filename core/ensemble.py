"""Ensemble classifier - weighted soft voting (FR-14..FR-16).

Weighted probability averaging over the available sub-models:
    P_ensemble = (w_cnn*P_cnn + w_vit*P_vit + w_lstm*P_lstm) / (sum of weights)

Verdict (FR-15):  FAKE if P_ensemble >= threshold else REAL
Confidence (FR-16): P*100 for FAKE, (1-P)*100 for REAL

Model-disagreement guard: removed (see git history). The ensemble always
votes, and per-model scores are surfaced so the user can judge the spread.
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


def decision_from_p(p_fake: float, threshold: float = 0.5) -> dict:
    """Build the final verdict + confidence record (FR-15/FR-16)."""
    verdict = "FAKE" if p_fake >= threshold else "REAL"
    confidence = p_fake * 100.0 if verdict == "FAKE" else (1.0 - p_fake) * 100.0
    return {
        "verdict": verdict,
        "p_fake": round(float(p_fake), 4),
        "confidence": round(float(confidence), 2),
        "threshold": threshold,
    }


class Ensemble:
    """Configured weighted-soft-voting helper."""

    def __init__(self, cfg: Config):
        default_thresh = float(cfg.ensemble.get("threshold", 0.50))
        self.image_threshold = float(cfg.ensemble.get("image_threshold", 0.50))
        self.video_threshold = float(cfg.ensemble.get("video_threshold", default_thresh))
        self.image_weights = {k: float(v) for k, v in cfg.ensemble.image_weights.items()}
        self.video_weights = {k: float(v) for k, v in cfg.ensemble.video_weights.items()}

    def combine(self, scores: dict[str, float], kind: str) -> dict:
        weights = self.video_weights if kind == "video" else self.image_weights
        threshold = self.video_threshold if kind == "video" else self.image_threshold
        p_fake = weighted_average(scores, weights)

        # Multi-model consensus for images: if multiple models detect manipulation (>= 0.50),
        # ensure p_fake reflects the consensus of the detecting models.
        if kind == "image" and scores:
            fake_votes = [p for p in scores.values() if p >= 0.50]
            if len(fake_votes) >= 2:
                p_fake = max(p_fake, sum(fake_votes) / len(fake_votes))

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