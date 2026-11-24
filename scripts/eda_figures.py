"""Figuras da EDA a partir de results/eda/*.csv e sobreposição de máscaras."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CLASSES, load_image, load_masks

out = Path("results/eda")
img_df = pd.read_csv(out / "per_image.csv")
comp = pd.read_csv(out / "per_component.csv")
COLORS = {"MA": (255, 0, 0), "HE": (0, 90, 255), "EX": (255, 220, 0), "SE": (0, 220, 120)}

# 1) fração de pixels por classe (escala log), por imagem
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([img_df[img_df.cls == c].frac_fov * 100 for c in CLASSES], tick_labels=CLASSES)
ax.set_yscale("log"); ax.set_ylabel("% do campo de visão (log)"); ax.set_title("Fração de pixels de lesão por imagem")
fig.tight_layout(); fig.savefig(out / "frac_pixels_por_classe.png", dpi=130); plt.close(fig)

# 2) tamanho dos componentes conexos
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, c in zip(axes, CLASSES):
    a = comp[comp.cls == c].area
    ax.hist(np.log10(a), bins=30); ax.set_title(f"{c} (n={len(a)})"); ax.set_xlabel("log10(área em px)")
fig.tight_layout(); fig.savefig(out / "tamanho_componentes.png", dpi=130); plt.close(fig)

# 3) componentes por imagem
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([img_df[img_df.cls == c].n_components for c in CLASSES], tick_labels=CLASSES)
ax.set_yscale("symlog"); ax.set_ylabel("nº de componentes por imagem"); fig.tight_layout()
fig.savefig(out / "componentes_por_imagem.png", dpi=130); plt.close(fig)

# 4) sobreposição de máscaras em algumas imagens (treino e teste, com e sem SE)
def overlay(image_id, split):
    img = load_image(image_id, split)
    masks, valid = load_masks(image_id, split, img.shape[:2])
    ov = img.copy()
    for k, c in enumerate(CLASSES):
        if valid[k]:
            ov[masks[k] > 0] = COLORS[c]
    return cv2.resize(cv2.addWeighted(img, 0.4, ov, 0.6, 0), (1072, 712)), valid

samples = [("IDRiD_03", "train"), ("IDRiD_10", "train"), ("IDRiD_17", "train"), ("IDRiD_43", "train"),
           ("IDRiD_61", "test"), ("IDRiD_81", "test")]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (i, s) in zip(axes.ravel(), samples):
    o, v = overlay(i, s)
    ax.imshow(o); ax.axis("off")
    ax.set_title(f"{i} ({s}) | máscaras ausentes: {[c for c, ok in zip(CLASSES, v) if not ok] or '-'}", fontsize=9)
handles = [plt.Line2D([], [], marker="s", ls="", color=np.array(COLORS[c]) / 255, label=c) for c in CLASSES]
fig.legend(handles=handles, loc="lower center", ncol=4)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out / "overlay_exemplos.png", dpi=110); plt.close(fig)

# 5) tabela resumo
summary = img_df.groupby("cls").agg(
    imagens=("image_id", "count"),
    frac_fov_media_pct=("frac_fov", lambda s: 100 * s.mean()),
    componentes_mediana=("n_components", "median"),
).reindex(CLASSES)
summary["componentes_total"] = comp.groupby("cls").size().reindex(CLASSES)
summary["area_mediana_px"] = comp.groupby("cls").area.median().reindex(CLASSES)
summary["area_max_px"] = comp.groupby("cls").area.max().reindex(CLASSES)
tot = img_df.groupby("cls").pixels.sum().reindex(CLASSES)
summary["pct_dos_pixels_de_lesao"] = 100 * tot / tot.sum()
summary.round(3).to_csv(out / "resumo_por_classe.csv")
print(summary.round(3).to_string())
