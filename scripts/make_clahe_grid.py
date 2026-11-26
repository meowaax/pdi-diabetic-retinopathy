"""Gera configs/clahe_grid/*.yaml: a grade de parâmetros do CLAHE (issue #24).

Cada arquivo herda configs/train_default.yaml e só muda preprocess_mode/preprocess_params.
Os hiperparâmetros de treino (lr, batch_size etc.) vêm do E1 (#13); até lá, os arquivos usam
os valores de configs/train_default.yaml como placeholder.
"""
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1] / "configs" / "train_default.yaml"
OUT = Path(__file__).resolve().parents[1] / "configs" / "clahe_grid"

CLIP_LIMITS = (1, 2, 3, 4)
TILE_SIZES = (8, 16)


def build_grid(base_path: Path = BASE, out_dir: Path = OUT) -> list[Path]:
    base = yaml.safe_load(base_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for clip in CLIP_LIMITS:
        for tile in TILE_SIZES:
            cfg = dict(base)
            cfg["preprocess_mode"] = "clahe"
            cfg["preprocess_params"] = {"clip_limit": float(clip), "tile_grid_size": [tile, tile]}
            name = f"clahe_clip{clip}_tile{tile}"
            path = out_dir / f"{name}.yaml"
            header = (
                f"# Gerado por scripts/make_clahe_grid.py (issue #24). NAO editar manualmente aqui;\n"
                f"# edite a grade no script e regere. Herda configs/train_default.yaml; lr/batch_size\n"
                f"# etc. sao placeholders ate o E1 (#13) congelar os hiperparametros reais.\n"
            )
            path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
            paths.append(path)
    return paths


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for p in build_grid():
        print(p)
