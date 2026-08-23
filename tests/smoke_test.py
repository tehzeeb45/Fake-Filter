""">End-to-end pipeline smoke test for 4 DeepShield-trained AI models (UC-03 image, UC-04 video).

Verifies the full request path: upload validation (FR-01/FR-02) -> face
detection (FR-06..FR-08) -> 4 New AI Models (Xception, EfficientNet-B3, ViT-Small, ViT-Large/CLIP)
inference -> soft-voting ensemble (FR-14..FR-16) -> Grad-CAM artifacts (FR-17).

Run:  python tests/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from core.detector import Detector  # noqa: E402
from core.validator import FileValidator  # noqa: E402

CFG = load_config()
FIX = Path(__file__).resolve().parent / "fixtures"
ART = Path(__file__).resolve().parent / "_smoke_artifacts"
IMAGE = FIX / "sample_face.png"
VIDEO = FIX / "sample_face.mp4"

V = FileValidator(CFG)
D = Detector(CFG)

checks = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        checks.append(True)
    except Exception as exc:  # noqa: BLE001 - surfacing smoke failures
        print(f"FAIL  {name}  -> {type(exc).__name__}: {exc}")
        checks.append(False)


def t_image_uploads():
    meta = V.validate(IMAGE.name, IMAGE.read_bytes())
    assert meta["kind"] == "image", meta


def t_video_uploads():
    meta = V.validate(VIDEO.name, VIDEO.read_bytes())
    assert meta["kind"] == "video", meta


def t_image_detect():
    r = D.detect_image(IMAGE, ART, original_name=IMAGE.name)
    assert r["kind"] == "image"
    assert r["verdict"] in ("REAL", "FAKE")
    assert 0.0 <= r["p_fake"] <= 1.0
    for key in ("new_xception", "new_efficientnet", "new_vit_small", "new_vit_large_clip"):
        assert r["scores"].get(key) is not None, f"missing {key} score"
    assert (ART / r["heatmap_path"]).is_file()
    assert (ART / r["face_crop_path"]).is_file()
    assert r["faces_analyzed"] == 1


def t_video_detect():
    r = D.detect_video(VIDEO, ART, original_name=VIDEO.name)
    assert r["kind"] == "video"
    assert r["verdict"] in ("REAL", "FAKE")
    for key in ("new_xception", "new_efficientnet", "new_vit_small", "new_vit_large_clip"):
        assert r["scores"].get(key) is not None, f"missing {key} score"
    assert (ART / r["heatmap_path"]).is_file()
    assert (ART / r["face_crop_path"]).is_file()
    assert 0 < r["faces_analyzed"] <= int(CFG.preprocessing.max_frames)
    assert r["most_manipulated_frame"] >= 1
    assert r["analyzed_seconds"] > 0


if __name__ == "__main__":
    ART.mkdir(parents=True, exist_ok=True)
    check("FR-01/FR-02 upload-validation accepted fixtures", t_image_uploads)
    check("FR-01/FR-02 upload-validation accepted video", t_video_uploads)
    check("UC-03 image pipeline (4 New Models + Soft Voting + Grad-CAM)", t_image_detect)
    check("UC-04 video pipeline (4 New Models + Top-K Pooling + Soft Voting)", t_video_detect)

    print(f"\nsmoke: {sum(checks)}/{len(checks)} checks passed")
    sys.exit(0 if all(checks) else 1)