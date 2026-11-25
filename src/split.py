"""Acesso à divisão fixa configs/split.csv."""
from pathlib import Path

import pandas as pd

SPLIT_CSV = Path("configs/split.csv")


def load_split(path: Path = SPLIT_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def ids_for(subset: str, path: Path = SPLIT_CSV) -> list[str]:
    """subset em {'train', 'val', 'test'}."""
    df = load_split(path)
    return df[df.subset == subset].image_id.tolist()


def source_split(image_id: str) -> str:
    """Pasta de origem no IDRiD ('train' ou 'test'): IDRiD_01..54 -> train; 55..81 -> test."""
    return "train" if int(image_id.split("_")[1]) <= 54 else "test"


def fold_ids(k: int, path: Path = SPLIT_CSV) -> tuple[list[str], list[str]]:
    """Dobra k (0-4) da validação cruzada: (treino, validação), só com imagens oficiais de treino."""
    df = load_split(path)
    df = df[df.fold >= 0]
    return df[df.fold != k].image_id.tolist(), df[df.fold == k].image_id.tolist()
