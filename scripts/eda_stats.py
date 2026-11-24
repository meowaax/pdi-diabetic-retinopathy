"""Estatísticas por imagem/classe e por componente conexo. Saídas em results/eda/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src.data import CLASSES, availability_table, load_image, load_masks

out = Path("results/eda")
out.mkdir(parents=True, exist_ok=True)

per_image, per_comp = [], []
for r in availability_table().itertuples():
    img = load_image(r.image_id, r.split)
    h, w = img.shape[:2]
    # campo de visão (FOV): pixels não pretos, para dar uma fração mais justa
    fov = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 15)
    fov = cv2.morphologyEx(fov.astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)).astype(bool)
    masks, valid = load_masks(r.image_id, r.split, (h, w))
    for k, cls in enumerate(CLASSES):
        if not valid[k]:
            continue
        m = masks[k]
        n_comp, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        per_image.append(dict(image_id=r.image_id, split=r.split, cls=cls, pixels=int(m.sum()),
                              frac_image=m.sum() / (h * w), frac_fov=m.sum() / fov.sum(),
                              n_components=len(areas), has_lesion=int(m.sum() > 0)))
        per_comp += [dict(image_id=r.image_id, split=r.split, cls=cls, area=int(a)) for a in areas]
    print(r.image_id, end=" ", flush=True)

pd.DataFrame(per_image).to_csv(out / "per_image.csv", index=False)
pd.DataFrame(per_comp).to_csv(out / "per_component.csv", index=False)
print("\nsalvo em", out)
