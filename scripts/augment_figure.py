"""Figura antes/depois da augmentation fixa (#8), em patches reais com MA, HE, EX e SE (IDRiD_03)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.augment import apply_augmentation, build_augmentation
from src.data import CLASSES, load_image, load_masks

COLORS = {"MA": (255, 0, 0), "HE": (0, 90, 255), "EX": (255, 220, 0), "SE": (0, 220, 120)}


def overlay(img, masks, valid):
    ov = img.copy()
    for k, c in enumerate(CLASSES):
        if valid[k]:
            ov[masks[..., k] > 0] = COLORS[c]
    return np.array(np.clip(0.4 * img + 0.6 * ov, 0, 255), dtype=np.uint8)


img = load_image("IDRiD_03", "train")
masks4, valid = load_masks("IDRiD_03", "train", img.shape[:2])
y, x = 1100, 1900  # recorte com MA, HE e EX visíveis
crop_img = img[y:y + 512, x:x + 512]
crop_mask = masks4.transpose(1, 2, 0)[y:y + 512, x:x + 512]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col in range(4):
    t = build_augmentation(seed=col)
    out = apply_augmentation(t, crop_img.copy(), crop_mask.copy())
    axes[0, col].imshow(overlay(crop_img, crop_mask, valid)); axes[0, col].set_title("antes"); axes[0, col].axis("off")
    axes[1, col].imshow(overlay(out["image"], out["mask"], valid)); axes[1, col].set_title(f"depois (seed={col})")
    axes[1, col].axis("off")
handles = [plt.Line2D([], [], marker="s", ls="", color=np.array(COLORS[c]) / 255, label=c) for c in CLASSES]
fig.legend(handles=handles, loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out_path = Path("results/eda/augment_antes_depois.png")
fig.savefig(out_path, dpi=110)
print("salvo em", out_path)
