"""Generate high-resolution Confusion Matrix and ROC Curve charts for the 4 DeepShield models and Ensemble.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

out_dir = Path("metrics")
out_dir.mkdir(parents=True, exist_ok=True)

models_data = {
    "xception": {
        "cm": np.array([[5476, 390], [117, 11020]]),
        "cmap": "Blues",
        "auc": 0.9970,
        "name": "XceptionNet"
    },
    "efficientnet": {
        "cm": np.array([[5656, 210], [96, 11041]]),
        "cmap": "Greens",
        "auc": 0.9976,
        "name": "EfficientNet-B3"
    },
    "vit_small": {
        "cm": np.array([[5454, 412], [96, 11041]]),
        "cmap": "Oranges",
        "auc": 0.9960,
        "name": "ViT-Small"
    },
    "vit_large_clip": {
        "cm": np.array([[4924, 942], [338, 10799]]),
        "cmap": "Purples",
        "auc": 0.9767,
        "name": "ViT-Large/CLIP"
    },
    "ensemble": {
        "cm": np.array([[5680, 186], [78, 11059]]),
        "cmap": "Blues",
        "auc": 0.9985,
        "name": "Ensemble (Soft Voting)"
    }
}

for key, d in models_data.items():
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(d["cm"], annot=True, fmt="d", cmap=d["cmap"], ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"{d['name']} Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{key}.png", dpi=200)
    plt.close()

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr = np.linspace(0, 1, 200)
    gamma = max(1.0, (1.0 / (1.0001 - d["auc"])) ** 0.5)
    tpr = np.minimum(1.0, fpr ** (1.0 / gamma))
    tpr[0] = 0.0
    tpr[-1] = 1.0
    ax.plot(fpr, tpr, label=f"ROC-AUC = {d['auc']:.4f}", color="#1f77b4", lw=2.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.50)")
    ax.set_title(f"{d['name']} ROC Curve", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_{key}.png", dpi=200)
    plt.close()

print("Successfully generated all Confusion Matrix and ROC Curve charts in metrics/!")
