"""Representações de entrada do núcleo experimental (etapa 2 do planejamento): RGB, CLAHE e canal verde.

Todas recebem e devolvem imagem RGB uint8 (H, W, 3), para compor com augmentation (#8, antes) e com
a normalização ImageNet feita em src/patches.py (depois). O valor final dos parâmetros do CLAHE é
decidido pela sensibilidade da issue #24; aqui eles só são expostos como argumentos.
"""
import cv2
import numpy as np

DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = (8, 8)


def preprocess_rgb(img: np.ndarray) -> np.ndarray:
    """A: RGB original. A normalização de escala (0-1, ImageNet) é feita no Dataset."""
    return img


def preprocess_clahe(img: np.ndarray, clip_limit: float = DEFAULT_CLIP_LIMIT,
                      tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE) -> np.ndarray:
    """B: CLAHE no canal L do espaço LAB, reconvertido para RGB (preserva cor)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_green(img: np.ndarray) -> np.ndarray:
    """C: canal verde replicado nos 3 canais de entrada."""
    g = img[..., 1]
    return np.repeat(g[..., None], 3, axis=2)


def preprocess_clahe_green(img: np.ndarray, clip_limit: float = DEFAULT_CLIP_LIMIT,
                            tile_grid_size: tuple[int, int] = DEFAULT_TILE_GRID_SIZE) -> np.ndarray:
    """D (opcional, #16): CLAHE seguido de canal verde, nessa ordem."""
    return preprocess_green(preprocess_clahe(img, clip_limit, tile_grid_size))


MODES = {"rgb": preprocess_rgb, "clahe": preprocess_clahe, "green": preprocess_green, "clahe_green": preprocess_clahe_green}


def preprocess(img: np.ndarray, mode: str, **params) -> np.ndarray:
    """Função única exigida pela issue #6. mode em {'rgb', 'clahe', 'green', 'clahe_green'}."""
    if mode not in MODES:
        raise ValueError(f"mode deve ser um de {sorted(MODES)}, recebeu {mode!r}")
    return MODES[mode](img, **params)
