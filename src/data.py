"""Carga de imagens e máscaras do IDRiD (A. Segmentation)."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("data/A. Segmentation")
CLASSES = ["MA", "HE", "EX", "SE"]
CLASS_DIRS = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates", "SE": "4. Soft Exudates"}
SPLITS = {"train": "a. Training Set", "test": "b. Testing Set"}


def image_path(image_id: str, split: str, root: Path = ROOT) -> Path:
    return root / "1. Original Images" / SPLITS[split] / f"{image_id}.jpg"


def mask_path(image_id: str, cls: str, split: str, root: Path = ROOT) -> Path:
    return root / "2. All Segmentation Groundtruths" / SPLITS[split] / CLASS_DIRS[cls] / f"{image_id}_{cls}.tif"


def list_images(split: str, root: Path = ROOT) -> list[str]:
    folder = root / "1. Original Images" / SPLITS[split]
    return sorted(p.stem for p in folder.glob("*.jpg"))


def availability_table(root: Path = ROOT) -> pd.DataFrame:
    """Uma linha por imagem; colunas MA/HE/EX/SE = 1 se a máscara existe, 0 se ausente."""
    rows = []
    for split in SPLITS:
        for image_id in list_images(split, root):
            row = {"image_id": image_id, "split": split}
            for cls in CLASSES:
                row[cls] = int(mask_path(image_id, cls, split, root).exists())
            rows.append(row)
    return pd.DataFrame(rows)


def load_image(image_id: str, split: str, root: Path = ROOT) -> np.ndarray:
    """Imagem RGB uint8 (H, W, 3)."""
    img = cv2.imread(str(image_path(image_id, split, root)), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path(image_id, split, root))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_masks(image_id: str, split: str, shape: tuple[int, int], root: Path = ROOT):
    """Retorna (máscaras (4, H, W) uint8 em {0,1}, validade (4,) uint8).

    Máscara ausente -> canal zerado e validade 0 (não significa ausência de lesão).
    As tifs do IDRiD usam valor 76 para lesão, por isso binarizamos com > 0.
    """
    masks = np.zeros((len(CLASSES), *shape), dtype=np.uint8)
    valid = np.zeros(len(CLASSES), dtype=np.uint8)
    for k, cls in enumerate(CLASSES):
        p = mask_path(image_id, cls, split, root)
        if not p.exists():
            continue
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise IOError(f"não foi possível ler {p}")
        if m.ndim == 3:
            # IDRiD_81_EX.tif é BGRA (lesão no canal vermelho, alfa = 255 em tudo): descartar o alfa
            m = m[..., :3].max(axis=2)
        if m.shape != shape:
            raise ValueError(f"{p.name}: formato {m.shape} difere da imagem {shape}")
        masks[k] = (m > 0).astype(np.uint8)
        valid[k] = 1
    return masks, valid
