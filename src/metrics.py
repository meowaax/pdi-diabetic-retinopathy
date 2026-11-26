"""Métricas de segmentação por classe (issue #11): Dice, IoU, precisão, sensibilidade.

Duas agregações, para usos diferentes:
- `pool_confusion` + `rates_from_confusion`: soma TP/FP/FN sobre todas as imagens antes de dividir
  (estilo do artigo de referência, que evita IoU indefinida em imagens sem lesão). Usada para
  comparar com o artigo e como leitura "robusta ao desbalanceamento entre imagens".
- `per_image_metrics` + `bootstrap_ci`: uma linha por (imagem, classe), pedida pelo planejamento
  para reportar variabilidade com intervalos de confiança (bootstrap sobre imagens). Imagens sem
  anotação da classe (valid=0) são descartadas antes do cálculo, não entram como NaN silencioso.

`macro_dice`: média do Dice entre as 4 classes (peso igual), calculada sobre o Dice agregado por
pool (uma classe, uma vez) — é a métrica principal de seleção de checkpoint/configuração.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CLASSES

EPS = 1e-8


@dataclass
class Confusion:
    tp: np.ndarray  # (C,)
    fp: np.ndarray
    fn: np.ndarray


def pool_confusion(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> Confusion:
    """pred, target: (N, C, H, W) binários (0/1). valid: (N, C).

    Soma TP/FP/FN sobre todas as imagens antes de dividir (estilo do artigo de referência), mas
    ignorando, por imagem e classe, as entradas com valid=0 (máscara ausente no IDRiD)."""
    pred, target = pred.astype(bool), target.astype(bool)
    v = valid.astype(bool)[:, :, None, None]
    tp = (pred & target & v).sum(axis=(0, 2, 3))
    fp = (pred & ~target & v).sum(axis=(0, 2, 3))
    fn = (~pred & target & v).sum(axis=(0, 2, 3))
    return Confusion(tp, fp, fn)


def rates_from_confusion(c: Confusion) -> pd.DataFrame:
    dice = (2 * c.tp + EPS) / (2 * c.tp + c.fp + c.fn + EPS)
    iou = (c.tp + EPS) / (c.tp + c.fp + c.fn + EPS)
    precision = (c.tp + EPS) / (c.tp + c.fp + EPS)
    sensitivity = (c.tp + EPS) / (c.tp + c.fn + EPS)
    return pd.DataFrame({"cls": CLASSES, "dice": dice, "iou": iou, "precision": precision, "sensitivity": sensitivity})


def macro_dice(c: Confusion) -> float:
    return float(rates_from_confusion(c).dice.mean())


def per_image_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, image_ids) -> pd.DataFrame:
    """pred, target: (N, C, H, W). valid: (N, C). Uma linha por (image_id, classe) válida."""
    rows = []
    for n, image_id in enumerate(image_ids):
        for k, cls in enumerate(CLASSES):
            if not valid[n, k]:
                continue
            p, t = pred[n, k].astype(bool), target[n, k].astype(bool)
            tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
            rows.append(dict(
                image_id=image_id, cls=cls,
                dice=(2 * tp + EPS) / (2 * tp + fp + fn + EPS),
                iou=(tp + EPS) / (tp + fp + fn + EPS),
                precision=(tp + EPS) / (tp + fp + EPS),
                sensitivity=(tp + EPS) / (tp + fn + EPS),
            ))
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap não paramétrico sobre as imagens. Devolve (média, limite inferior, limite superior)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def summarize_with_ci(per_image: pd.DataFrame, metric: str = "dice", n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Média e IC 95% (bootstrap sobre imagens) da métrica, por classe."""
    rows = []
    for cls in CLASSES:
        vals = per_image.loc[per_image.cls == cls, metric].to_numpy()
        mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        rows.append(dict(cls=cls, n=len(vals), mean=mean, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)
