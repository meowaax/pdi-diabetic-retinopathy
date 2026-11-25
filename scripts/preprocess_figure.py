"""Figura comparando RGB, CLAHE e canal verde em imagens com MA, EX e SE (issue #6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_image
from src.preprocess import MODES
from src.split import source_split

out = Path("results/eda")
out.mkdir(parents=True, exist_ok=True)

# IDRiD_03: MA+HE+EX+SE; IDRiD_81: EX grande; IDRiD_61: HE grande
samples = ["IDRiD_03", "IDRiD_81", "IDRiD_61"]
modes = ["rgb", "clahe", "green", "clahe_green"]

fig, axes = plt.subplots(len(samples), len(modes), figsize=(4 * len(modes), 4 * len(samples)))
for row, image_id in enumerate(samples):
    img = load_image(image_id, source_split(image_id))
    y0, x0 = img.shape[0] // 2 - 400, img.shape[1] // 2 - 400  # recorte central, mais fácil de inspecionar
    crop = img[y0:y0 + 800, x0:x0 + 800]
    for col, mode in enumerate(modes):
        ax = axes[row, col]
        ax.imshow(MODES[mode](crop))
        ax.axis("off")
        if row == 0:
            ax.set_title(mode)
        if col == 0:
            ax.set_ylabel(image_id)
fig.tight_layout()
fig.savefig(out / "preprocess_comparacao.png", dpi=110)
print("salvo em", out / "preprocess_comparacao.png")
