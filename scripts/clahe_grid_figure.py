"""Figura comparando as 8 configurações da grade do CLAHE (issue #24) em recortes com MA, EX e SE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.make_clahe_grid import CLIP_LIMITS, TILE_SIZES
from src.data import load_image
from src.preprocess import preprocess_clahe
from src.split import source_split

out = Path("results/eda")
out.mkdir(parents=True, exist_ok=True)

samples = ["IDRiD_03", "IDRiD_17"]  # MA+HE+EX+SE; HE/EX grandes
configs = [(clip, tile) for clip in CLIP_LIMITS for tile in TILE_SIZES]

fig, axes = plt.subplots(len(samples), len(configs), figsize=(2.6 * len(configs), 5.2 * len(samples)))
for row, image_id in enumerate(samples):
    img = load_image(image_id, source_split(image_id))
    y0, x0 = img.shape[0] // 2 - 300, img.shape[1] // 2 - 300
    crop = img[y0:y0 + 600, x0:x0 + 600]
    for col, (clip, tile) in enumerate(configs):
        ax = axes[row, col]
        ax.imshow(preprocess_clahe(crop, clip_limit=float(clip), tile_grid_size=(tile, tile)))
        ax.axis("off")
        if row == 0:
            ax.set_title(f"clip={clip}\ntile={tile}x{tile}", fontsize=9)
        if col == 0:
            ax.text(-0.15, 0.5, image_id, transform=ax.transAxes, rotation=90, va="center", ha="center")
fig.tight_layout()
fig.savefig(out / "clahe_grid_comparacao.png", dpi=110)
print("salvo em", out / "clahe_grid_comparacao.png")
