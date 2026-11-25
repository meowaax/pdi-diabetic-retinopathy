"""Testes de propriedades estatísticas do sampler (mais lentos; cobertos por test_patches.py no essencial)."""
from pathlib import Path

import numpy as np
import pytest

from src.data import CLASSES, load_image, load_masks
from src.patches import IDRiDPatches
from src.split import ids_for, source_split

pytestmark = pytest.mark.skipif(not Path("data/cache/IDRiD_01_meta.npz").exists(), reason="rode scripts/build_cache.py")


def test_eval_gera_grade_completa_em_imagem_cheia():
    ds = IDRiDPatches(["IDRiD_03"], mode="eval")
    assert len(ds) == 142  # grade 512/stride 256 cobrindo 2848x4288 com borda encostada


def test_amostragem_enviesada_eleva_fracao_de_lesao_acima_do_natural():
    train_ids = (ids_for("train") + ids_for("val"))[:12]
    natural = {c: [] for c in CLASSES}
    for iid in train_ids:
        img = load_image(iid, source_split(iid))
        masks, valid = load_masks(iid, source_split(iid), img.shape[:2])
        for k, c in enumerate(CLASSES):
            if valid[k]:
                natural[c].append(masks[k].mean())

    ds = IDRiDPatches(train_ids, mode="train", patches_per_image=20, lesion_prob=0.7)
    sampled = {c: [] for c in CLASSES}
    for i in range(len(ds)):
        s = ds[i]
        for k, c in enumerate(CLASSES):
            if s["valid"][k] == 1:
                sampled[c].append(s["mask"][k].float().mean().item())

    for c in CLASSES:
        n_mean, s_mean = np.mean(natural[c]), np.mean(sampled[c])
        assert s_mean > n_mean, f"{c}: amostragem não elevou a fração de lesão ({s_mean} <= {n_mean})"
