"""Ensemble soft-voting tests for 4 new DeepShield models (FR-14..FR-16).

Active models in config:
  - new_xception
  - new_efficientnet
  - new_vit_small
  - new_vit_large_clip
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from core.ensemble import Ensemble  # noqa: E402

CFG = load_config()
ENS = Ensemble(CFG)


def run(scores, kind):
    return ENS.combine(dict(scores), kind)


def test_image_votes_all_4_models():
    base = run({
        "new_xception": 0.1,
        "new_efficientnet": 0.1,
        "new_vit_small": 0.1,
        "new_vit_large_clip": 0.1
    }, "image")
    assert base["verdict"] == "REAL"
    assert abs(base["p_fake"] - 0.1) < 1e-6


def test_video_votes_all_4_models():
    base = run({
        "new_xception": 0.9,
        "new_efficientnet": 0.9,
        "new_vit_small": 0.9,
        "new_vit_large_clip": 0.9
    }, "video")
    assert base["verdict"] == "FAKE"
    assert abs(base["p_fake"] - 0.9) < 1e-6


def test_single_strong_fake_vote_is_averaged():
    # 4 models: one 0.90 vote, three 0.10 votes -> weighted average based on config -> REAL
    r = run({
        "new_xception": 0.90,
        "new_efficientnet": 0.10,
        "new_vit_small": 0.10,
        "new_vit_large_clip": 0.10
    }, "image")
    assert r["p_fake"] < 0.45
    assert r["verdict"] == "REAL"


def test_agreement_fake_image():
    r = run({
        "new_xception": 0.85,
        "new_efficientnet": 0.80,
        "new_vit_small": 0.75,
        "new_vit_large_clip": 0.90
    }, "image")
    assert r["verdict"] == "FAKE"
    assert r["disagreement"] is False


def test_agreement_fake_video():
    r = run({
        "new_xception": 0.70,
        "new_efficientnet": 0.80,
        "new_vit_small": 0.65,
        "new_vit_large_clip": 0.85
    }, "video")
    assert r["verdict"] == "FAKE"
    assert r["disagreement"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}  -> {exc}")
    print(f"\nensemble: {passed}/{len(fns)} checks passed")
    sys.exit(0 if passed == len(fns) else 1)