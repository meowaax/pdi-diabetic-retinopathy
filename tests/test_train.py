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
    deve conseguir memorizar e o Dice (medido nas próprias imagens de treino) deve subir bastante
    em poucas épocas. Não prova que o modelo generaliza; prova que o grafo de treino está correto."""
    ids = ["IDRiD_03", "IDRiD_08"]
    cfg = TrainConfig(
        encoder_weights=None, lr=1e-3, batch_size=2, max_epochs=15, patience=15,
        patches_per_image=6, lesion_prob=1.0, eval_stride=2048, checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"), run_name="overfit_sanity",
    )
    device = get_device()
    model_antes = build_model(encoder_weights=None).to(device)
    dice_antes, _ = evaluate(model_antes, ids, cfg, device)

    result = train(ids, ids, cfg)  # mesmas imagens em treino e validação: mede capacidade de memorizar

    assert result["best_val_macro_dice"] > dice_antes + 0.2
    assert Path(result["checkpoint_path"]).exists()
    assert Path(result["log_path"]).exists()
    assert len(result["history"]) >= 1


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
