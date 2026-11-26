"""Dispara um treino a partir de um YAML (configs/*.yaml ou configs/clahe_grid/*.yaml).

Dois modos de seleção de dados, mutuamente exclusivos:
  --subset  usa a divisão de desenvolvimento (configs/split.csv: train/val), para o E1 (#13) e a
            sensibilidade do CLAHE (#24).
  --fold K  usa a dobra K da validação cruzada (src/split.fold_ids), para a comparação final
            E1-E4 (#17).

Exemplos:
  python scripts/run_experiment.py configs/train_default.yaml --subset --run-name e1_dev
  python scripts/run_experiment.py configs/clahe_grid/clahe_clip2_tile8.yaml --subset \
      --run-name clahe_clip2_tile8
  python scripts/run_experiment.py configs/train_default.yaml --fold 0 --run-name e1_fold0

Salva o resultado (best_epoch, best_val_macro_dice, caminhos de checkpoint/log, histórico e
per_class_dice do melhor checkpoint) em results/runs/{run_name}.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import build_model
from src.seed import get_device
from src.split import fold_ids, ids_for
from src.train import TrainConfig, evaluate, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML em configs/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subset", action="store_true", help="usa configs/split.csv: subset train/val")
    group.add_argument("--fold", type=int, help="usa a dobra K (0-4) da validação cruzada")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-epochs", type=int, default=None, help="override de max_epochs (ex.: smoke test)")
    args = parser.parse_args()

    if args.subset:
        train_ids, val_ids = ids_for("train"), ids_for("val")
    else:
        train_ids, val_ids = fold_ids(args.fold)

    overrides = {}
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    cfg = TrainConfig.from_yaml(args.config, run_name=args.run_name, **overrides)

    print(f"[{args.run_name}] treino={len(train_ids)} imagens, validação={len(val_ids)} imagens, "
          f"config={args.config}")
    result = train(train_ids, val_ids, cfg)

    import torch

    device = get_device()
    model = build_model(encoder_weights=cfg.encoder_weights).to(device)
    ckpt = torch.load(result["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, details = evaluate(model, val_ids, cfg, device)
    result["per_class_dice"] = {k: float(v) for k, v in details.get("per_class_dice", {}).items()}
    result["config_path"] = str(args.config)
    result["selection"] = {"mode": "subset"} if args.subset else {"mode": "fold", "fold": args.fold}

    out_dir = Path("results/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{args.run_name}] melhor época {result['best_epoch']}, "
          f"macro-Dice {result['best_val_macro_dice']:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
