"""Loop de treinamento com early stopping e checkpoints (issue #12).

Monta o Dataset de treino/validação (src/patches.py), o modelo (src/model.py), a loss mascarada
(src/losses.py) e avalia por macro-Dice (src/metrics.py). Configuração via YAML (configs/*.yaml);
ver configs/train_default.yaml para os valores de partida (do artigo e do planejamento).

Avaliação de valida��ão: usa IDRiDPatches em modo "eval" (grade determinística), reconstituindo a
imagem inteira com stitch() antes de calcular as métricas — evita medir só em patches recortados,
que distorceria o Dice de lesões cortadas na borda de um patch de treino.
"""
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import CLASSES
from .losses import bce_dice_loss
from .metrics import macro_dice, pool_confusion, rates_from_confusion
from .model import build_model
from .patches import IDRiDPatches, stitch
from .preprocess import preprocess as preprocess_fn
from .seed import get_device, set_seed


@dataclass
class TrainConfig:
    seed: int = 42
    preprocess_mode: str = "rgb"
    preprocess_params: dict = field(default_factory=dict)
    encoder_weights: str | None = "imagenet"
    lr: float = 1e-3
    batch_size: int = 6
    bce_weight: float = 0.5
    max_epochs: int = 100
    patience: int = 10
    patches_per_image: int = 24
    lesion_prob: float = 0.7
    num_workers: int = 0
    mixed_precision: bool = True
    eval_stride: int = 256
    eval_min_fov: float = 0.2
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "results/logs"
    run_name: str = "run"

    @classmethod
    def from_yaml(cls, path: str | Path, run_name: str | None = None, **overrides) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text())
        flat = {
            "seed": raw.get("seed", 42),
            "preprocess_mode": raw.get("preprocess_mode", "rgb"),
            "preprocess_params": raw.get("preprocess_params", {}),
            "encoder_weights": raw.get("model", {}).get("encoder_weights", "imagenet"),
            "lr": raw.get("optim", {}).get("lr", 1e-3),
            "batch_size": raw.get("optim", {}).get("batch_size", 6),
            "bce_weight": raw.get("loss", {}).get("bce_weight", 0.5),
            "max_epochs": raw.get("train", {}).get("max_epochs", 100),
            "patience": raw.get("train", {}).get("patience", 10),
            "patches_per_image": raw.get("train", {}).get("patches_per_image", 24),
            "lesion_prob": raw.get("train", {}).get("lesion_prob", 0.7),
            "num_workers": raw.get("train", {}).get("num_workers", 0),
            "mixed_precision": raw.get("train", {}).get("mixed_precision", True),
            "eval_stride": raw.get("eval", {}).get("stride", 256),
            "eval_min_fov": raw.get("eval", {}).get("min_fov", 0.2),
            "checkpoint_dir": raw.get("paths", {}).get("checkpoint_dir", "results/checkpoints"),
            "log_dir": raw.get("paths", {}).get("log_dir", "results/logs"),
        }
        flat.update(overrides)
        if run_name is not None:
            flat["run_name"] = run_name
        return cls(**flat)


def _make_preprocess(cfg: TrainConfig):
    if cfg.preprocess_mode == "rgb" and not cfg.preprocess_params:
        return None  # identidade: pula uma cópia desnecessária
    return lambda img: preprocess_fn(img, cfg.preprocess_mode, **cfg.preprocess_params)


@torch.no_grad()
def evaluate(model, eval_ids, cfg: TrainConfig, device, transform=None, progress_fn=None) -> tuple[float, dict]:
    """Avalia por imagem completa (patches da grade, remontados), devolve (macro_dice, detalhes).

    detalhes["per_class_dice"] vem de rates_from_confusion sobre a mesma matriz de confusão
    agregada usada no macro_dice (consistência entre a métrica de seleção e o detalhamento).

    progress_fn(i, n, image_id), se fornecida, é chamada antes de processar cada imagem — útil
    para dar visibilidade quando a validação tem muitas imagens (ex.: a grade completa de uma
    imagem 2848x4288 em patches 512x512 pode levar dezenas de segundos)."""
    model.eval()
    preds, targets, valids = [], [], []
    for i, image_id in enumerate(eval_ids):
        if progress_fn is not None:
            progress_fn(i, len(eval_ids), image_id)
        ds = IDRiDPatches([image_id], mode="eval", stride=cfg.eval_stride, min_fov=cfg.eval_min_fov,
                           preprocess=_make_preprocess(cfg), transform=transform)
        if len(ds) == 0:
            continue
        logits_list, masks_list, positions = [], [], []
        for i in range(0, len(ds), cfg.batch_size):
            batch = [ds[j] for j in range(i, min(i + cfg.batch_size, len(ds)))]
            x = torch.stack([b["image"] for b in batch]).to(device)
            logits_list.append(model(x).cpu().numpy())
            masks_list += [b["mask"].numpy() for b in batch]
            positions += [(b["y"], b["x"]) for b in batch]
        logits = np.concatenate(logits_list, axis=0)
        masks = np.stack(masks_list, axis=0)
        full_logits = stitch(logits, positions, ds.shapes[image_id])
        full_target = stitch(masks, positions, ds.shapes[image_id])
        pred_bin = (full_logits > 0).astype(np.uint8)[None]  # sigmoid(x) > 0.5  <=>  x > 0, sem overflow
        preds.append(pred_bin)
        targets.append((full_target > 0.5).astype(np.uint8)[None])
        valids.append(ds._meta[image_id]["valid"][None])
    if not preds:
        return float("nan"), {}
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    valid = np.concatenate(valids, axis=0)
    conf = pool_confusion(pred, target, valid)
    md = macro_dice(conf)
    return md, {"per_class_dice": dict(zip(CLASSES, rates_from_confusion(conf).dice))}


def train(train_ids, val_ids, cfg: TrainConfig, transform=None, log_fn=print) -> dict:
    """Treina até max_epochs ou patience épocas sem melhora do macro-Dice de validação.

    Salva o melhor checkpoint (maior macro-Dice de validação) em
    {checkpoint_dir}/{run_name}_best.pt e um log por época em {log_dir}/{run_name}.csv.
    Devolve um resumo com o histórico e o caminho do melhor checkpoint.
    """
    set_seed(cfg.seed)
    device = get_device()
    model = build_model(encoder_weights=cfg.encoder_weights).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_ds = IDRiDPatches(train_ids, mode="train", patches_per_image=cfg.patches_per_image,
                             lesion_prob=cfg.lesion_prob, preprocess=_make_preprocess(cfg),
                             transform=transform, seed=cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    ckpt_dir, log_dir = Path(cfg.checkpoint_dir), Path(cfg.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg.run_name}_best.pt"
    log_path = log_dir / f"{cfg.run_name}.csv"

    # float16 autocast sem GradScaler faz os logits explodirem e a loss virar NaN em poucas
    # épocas (reproduzido em MPS; é o comportamento documentado do PyTorch para float16 puro).
    # bfloat16 tem a faixa dinâmica do float32 (só perde mantissa) e não sofre esse overflow,
    # então não precisa de escala de gradiente — mas só existe autocast de bfloat16 em CPU/CUDA/MPS
    # recentes, não em todo hardware; se faltar suporte, caímos para float32 (sem amp).
    use_amp = cfg.mixed_precision and device.type in ("cuda", "mps")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    best_dice, best_epoch, epochs_no_improve = -1.0, -1, 0
    history = []
    had_nan = False
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_macro_dice", "seconds", "is_best"])

        for epoch in range(cfg.max_epochs):
            train_ds.set_epoch(epoch)
            model.train()
            t0 = time.time()
            losses = []
            n_batches = len(loader)
            for b, batch in enumerate(loader):
                x = batch["image"].to(device)
                y = batch["mask"].to(device)
                v = batch["valid"].to(device)
                optim.zero_grad()
                if use_amp:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        logits = model(x)
                        loss = bce_dice_loss(logits, y, v, bce_weight=cfg.bce_weight)
                    if device.type == "cuda":
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()
                    else:
                        loss.backward()
                        optim.step()
                else:
                    logits = model(x)
                    loss = bce_dice_loss(logits, y, v, bce_weight=cfg.bce_weight)
                    loss.backward()
                    optim.step()
                loss_value = loss.item()
                if not np.isfinite(loss_value):
                    had_nan = True
                    log_fn(f"[{cfg.run_name}] ATENÇÃO: loss não-finita (nan/inf) na época {epoch}, "
                           f"batch {b + 1}. Resultado deste run deve ser descartado.")
                losses.append(loss_value)
                if (b + 1) % max(1, n_batches // 4) == 0 or b + 1 == n_batches:
                    log_fn(f"[{cfg.run_name}] época {epoch}: treino batch {b + 1}/{n_batches} "
                           f"(loss médio até aqui: {np.mean(losses):.4f})")

            def _eval_progress(i, n, image_id):
                log_fn(f"[{cfg.run_name}] época {epoch}: validando imagem {i + 1}/{n} ({image_id})")

            val_dice, _ = evaluate(model, val_ids, cfg, device, transform=None, progress_fn=_eval_progress)
            is_best = val_dice > best_dice
            if is_best:
                best_dice, best_epoch, epochs_no_improve = val_dice, epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_macro_dice": val_dice,
                            "config": cfg.__dict__}, ckpt_path)
            else:
                epochs_no_improve += 1

            dt = time.time() - t0
            train_loss = float(np.mean(losses))
            writer.writerow([epoch, train_loss, val_dice, round(dt, 2), int(is_best)])
            f.flush()
            history.append(dict(epoch=epoch, train_loss=train_loss, val_macro_dice=val_dice, seconds=dt))
            log_fn(f"[{cfg.run_name}] época {epoch}: loss={train_loss:.4f} val_macro_dice={val_dice:.4f} "
                   f"({'novo melhor' if is_best else f'{epochs_no_improve}/{cfg.patience} sem melhora'})")

            if epochs_no_improve >= cfg.patience:
                log_fn(f"[{cfg.run_name}] early stopping na época {epoch} (melhor: época {best_epoch}, "
                       f"macro-Dice {best_dice:.4f})")
                break

    return {"best_epoch": best_epoch, "best_val_macro_dice": best_dice, "checkpoint_path": str(ckpt_path),
            "log_path": str(log_path), "history": history, "had_nan": had_nan}
