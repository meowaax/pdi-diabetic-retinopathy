"""Testes do loop de treino (issue #12). Usa data/cache/ (gerado por scripts/build_cache.py)."""
from pathlib import Path

import pytest
import yaml

from src.train import TrainConfig, evaluate, train
from src.model import build_model
from src.seed import get_device

pytestmark = pytest.mark.skipif(not Path("data/cache/IDRiD_01_meta.npz").exists(), reason="rode scripts/build_cache.py")


def test_config_from_yaml_le_o_arquivo_padrao():
    cfg = TrainConfig.from_yaml("configs/train_default.yaml", run_name="teste")
    assert cfg.lr == 0.001 and cfg.batch_size == 6 and cfg.patience == 10 and cfg.max_epochs == 100
    assert cfg.run_name == "teste"


def test_config_from_yaml_aceita_overrides(tmp_path):
    cfg = TrainConfig.from_yaml("configs/train_default.yaml", run_name="x", lr=5e-4, max_epochs=3)
    assert cfg.lr == 5e-4 and cfg.max_epochs == 3


def test_overfit_em_poucas_imagens_eleva_o_dice_de_treino(tmp_path):
    """Sanidade pedida pela issue: sem pesos pré-treinados (rápido) e poucas imagens, o modelo
    deve conseguir memorizar. Checa dois sinais:
    (a) a loss de treino cai de forma consistente ao longo das épocas (prova que o gradiente
        flui e o otimizador atualiza os pesos na direção certa);
    (b) o macro-Dice de validação (medido nas próprias imagens de treino, remontadas da grade)
        melhora em relação ao modelo com pesos aleatórios.
    Não prova que o modelo generaliza para imagens novas; prova que o grafo dataset -> modelo ->
    loss -> otimizador está conectado corretamente, sem NaN/Inf ao longo do treino."""
    ids = ["IDRiD_03", "IDRiD_08"]
    cfg = TrainConfig(
        encoder_weights=None, lr=1e-3, batch_size=2, max_epochs=20, patience=20,
        patches_per_image=8, lesion_prob=1.0, eval_stride=2048, checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"), run_name="overfit_sanity",
    )
    device = get_device()
    model_antes = build_model(encoder_weights=None).to(device)
    dice_antes, _ = evaluate(model_antes, ids, cfg, device)

    result = train(ids, ids, cfg)  # mesmas imagens em treino e validação: mede capacidade de memorizar

    assert not result["had_nan"], "loss não-finita durante o treino (ver src/train.py: mixed precision)"
    losses = [h["train_loss"] for h in result["history"]]
    media_inicial = sum(losses[:3]) / 3
    media_final = sum(losses[-3:]) / 3
    assert media_final < media_inicial, f"loss não caiu: inicial={media_inicial:.4f} final={media_final:.4f}"
    assert result["best_val_macro_dice"] > dice_antes
    assert Path(result["checkpoint_path"]).exists()
    assert Path(result["log_path"]).exists()
    assert len(result["history"]) >= 1


def test_mixed_precision_nao_produz_loss_nan(tmp_path):
    """Regressão: float16 autocast sem GradScaler fazia os logits explodirem e a loss virar NaN
    em poucas épocas (reproduzido em MPS). mixed_precision=True é o padrão de configs/*.yaml."""
    ids = ["IDRiD_03", "IDRiD_08"]
    cfg = TrainConfig(
        encoder_weights=None, lr=1e-3, batch_size=2, max_epochs=15, patience=15,
        patches_per_image=6, lesion_prob=1.0, eval_stride=2048, mixed_precision=True,
        checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"), run_name="amp_sanity",
    )
    result = train(ids, ids, cfg, log_fn=lambda *a, **k: None)
    assert not result["had_nan"]
    assert all(h["train_loss"] == h["train_loss"] for h in result["history"])  # nenhum NaN (NaN != NaN)


def test_checkpoint_salvo_contem_o_estado_do_modelo(tmp_path):
    import torch

    ids = ["IDRiD_03"]
    cfg = TrainConfig(
        encoder_weights=None, batch_size=1, max_epochs=2, patience=2, patches_per_image=4,
        eval_stride=2048, checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"),
        run_name="ckpt_sanity",
    )
    result = train(ids, ids, cfg)
    ckpt = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    assert "model_state" in ckpt and "val_macro_dice" in ckpt and "config" in ckpt


def test_early_stopping_interrompe_antes_de_max_epochs(tmp_path):
    """patience=0: para na primeira época sem melhora, ou seja, na 2a época no pior caso."""
    ids = ["IDRiD_03"]
    cfg = TrainConfig(
        encoder_weights=None, batch_size=1, max_epochs=50, patience=0, patches_per_image=2,
        eval_stride=2048, checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"),
        run_name="es_sanity",
    )
    result = train(ids, ids, cfg, log_fn=lambda *a, **k: None)
    assert len(result["history"]) < 50


def test_log_csv_tem_uma_linha_por_epoca_executada(tmp_path):
    import csv

    ids = ["IDRiD_03"]
    cfg = TrainConfig(
        encoder_weights=None, batch_size=1, max_epochs=3, patience=3, patches_per_image=2,
        eval_stride=2048, checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"),
        run_name="log_sanity",
    )
    result = train(ids, ids, cfg, log_fn=lambda *a, **k: None)
    with open(result["log_path"]) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["epoch", "train_loss", "val_macro_dice", "seconds", "is_best"]
    assert len(rows) - 1 == len(result["history"])
