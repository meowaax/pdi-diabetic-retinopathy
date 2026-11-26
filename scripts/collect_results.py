"""Agrega results/runs/*.json (saída de scripts/run_experiment.py) em tabelas (issue #24, #17, #13).

Dois modos de leitura, conforme o campo "selection" de cada JSON:
  - {"mode": "subset"}        -> uma linha por run (ex.: a grade do CLAHE, issue #24)
  - {"mode": "fold", "fold":k} -> agrupadas por nome base do run (sem o sufixo da dobra) para a
    comparação final E1-E4 (issue #17): média e desvio-padrão entre dobras, por classe.

Uso:
  python scripts/collect_results.py                      # processa tudo em results/runs/
  python scripts/collect_results.py --prefix clahe_       # só os runs cujo nome começa com isso
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import CLASSES

RUNS_DIR = Path("results/runs")
OUT_DIR = Path("results/summaries")


def load_runs(runs_dir: Path = RUNS_DIR, prefix: str = "") -> pd.DataFrame:
    rows = []
    for p in sorted(runs_dir.glob("*.json")):
        if not p.stem.startswith(prefix):
            continue
        r = json.loads(p.read_text())
        sel = r.get("selection", {})
        row = {
            "run_name": p.stem,
            "config_path": r.get("config_path"),
            "mode": sel.get("mode"),
            "fold": sel.get("fold"),
            "best_epoch": r.get("best_epoch"),
            "macro_dice": r.get("best_val_macro_dice"),
        }
        for cls in CLASSES:
            row[f"dice_{cls}"] = r.get("per_class_dice", {}).get(cls)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["run_name", "mode", "fold", "macro_dice"] + [f"dice_{c}" for c in CLASSES])
    return pd.DataFrame(rows)


def base_name(run_name: str) -> str:
    """Remove um sufixo de dobra (ex.: 'e1_rgb_fold3' -> 'e1_rgb') para agrupar por experimento."""
    return re.sub(r"_fold\d+$", "", run_name)


def summarize_subset(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["run_name", "best_epoch", "macro_dice"] + [f"dice_{c}" for c in CLASSES]
    return df[df["mode"] == "subset"][cols].sort_values("macro_dice", ascending=False).reset_index(drop=True)


def summarize_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio-padrão entre dobras, por experimento base e por métrica."""
    folds = df[df["mode"] == "fold"].copy()
    if folds.empty:
        return pd.DataFrame()
    folds["experiment"] = folds.run_name.map(base_name)
    metric_cols = ["macro_dice"] + [f"dice_{c}" for c in CLASSES]
    agg = folds.groupby("experiment")[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    df = load_runs(args.runs_dir, args.prefix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "all_runs.csv", index=False)
    print(f"{len(df)} runs encontrados em {args.runs_dir}")

    sub = summarize_subset(df)
    if not sub.empty:
        sub.to_csv(args.out_dir / "subset_summary.csv", index=False)
        print("\n== divisão de desenvolvimento (subset) ==")
        print(sub.to_string(index=False))

    folds = summarize_folds(df)
    if not folds.empty:
        folds.to_csv(args.out_dir / "fold_summary.csv", index=False)
        print("\n== validação cruzada (média ± desvio entre dobras) ==")
        print(folds.to_string(index=False))


if __name__ == "__main__":
    main()
