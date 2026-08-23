"""Download DeepShield AI model weights from Hugging Face repository.

Models downloaded into models/ directory:
  - best_xception.pth          (XceptionNet, 2-class)
  - best_efficientnet_b3.pth   (EfficientNet-B3, 2-class)
  - best_vit_small.pth         (ViT-Small, 2-class)
  - best_vit_large_clip.pth    (ViT-Large/CLIP, 2-class)

Run:  python scripts/download_models.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Hugging Face repository ID (set your own HF repo if models are hosted there)
REPO = "tehzeeb45/deepfake-models"

FILES = {
    "best_xception.pth":        "models/best_xception.pth",
    "best_efficientnet_b3.pth": "models/best_efficientnet_b3.pth",
    "best_vit_small.pth":       "models/best_vit_small.pth",
    "best_vit_large_clip.pth":  "models/best_vit_large_clip.pth",
}


def _download(src: str, dst_path: Path) -> None:
    """Download one model file into a scratch dir, then move it into place."""
    tmp = Path(tempfile.mkdtemp(prefix="hf_dl_"))
    try:
        print(f"  downloading {src} from {REPO} ...")
        hf_hub_download(repo_id=REPO, filename=src, local_dir=str(tmp))
        target = tmp / src
        if not target.is_file():
            hits = list(tmp.rglob(src))
            target = hits[0] if hits else None
        if target is None or not target.is_file():
            raise RuntimeError(f"Downloaded file not found: {src}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(dst_path))
        print(f"  -> saved to {dst_path.relative_to(ROOT)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    missing = {src: dst for src, dst in FILES.items()
               if not (ROOT / dst).is_file()}

    if not missing:
        print("All 4 DeepShield model weights already exist locally in models/:")
        for src in FILES:
            print(f"  - models/{src}")
        return 0

    print(f"Found {len(missing)} missing model file(s) to download:")
    for src, dst in missing.items():
        print(f"  - {dst}")

    try:
        for src, rel_dst in missing.items():
            _download(src, ROOT / rel_dst)
    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        print("\nIf you have trained models locally or in Kaggle, manually place:")
        for src in FILES:
            print(f"  - models/{src}")
        return 1

    print("\nAll 4 DeepShield model weights downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())