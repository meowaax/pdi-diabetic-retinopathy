import numpy as np
import pytest

from src.augment import apply_augmentation, build_augmentation


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128, 4), dtype=np.uint8)
    mask[60:68, 60:68, 0] = 1  # MA: lesão pequena (8x8 px) no centro
    return img, mask


def test_formato_tipo_e_canais_preservados(sample):
    img, mask = sample
    t = build_augmentation(seed=1)
    out = apply_augmentation(t, img, mask)
    assert out["image"].shape == img.shape and out["image"].dtype == img.dtype
    assert out["mask"].shape == mask.shape and out["mask"].dtype == mask.dtype


def test_mascara_permanece_binaria(sample):
    img, mask = sample
    t = build_augmentation(seed=2)
    out = apply_augmentation(t, img, mask)
    assert set(np.unique(out["mask"])) <= {0, 1}


def test_determinismo_por_semente(sample):
    img, mask = sample
    a = apply_augmentation(build_augmentation(seed=5), img.copy(), mask.copy())
    b = apply_augmentation(build_augmentation(seed=5), img.copy(), mask.copy())
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["mask"], b["mask"])


def test_produz_variacao_entre_chamadas_sem_semente_fixa(sample):
    img, mask = sample
    t = build_augmentation(seed=None)
    outs = [apply_augmentation(t, img.copy(), mask.copy())["image"] for _ in range(8)]
    assert any(not np.array_equal(outs[0], o) for o in outs[1:])


def test_elastic_nao_apaga_lesao_pequena_de_ma_na_maioria_das_vezes():
    """ElasticTransform (alpha=1.0, sigma=10, parâmetros do artigo) não deve, na prática,
    remover por completo uma lesão de 8x8 px: ela apenas distorce levemente a geometria."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128, 4), dtype=np.uint8)
    mask[60:68, 60:68, 0] = 1
    sobrevive = 0
    for seed in range(30):
        import albumentations as A

        t = A.Compose([A.ElasticTransform(alpha=1.0, sigma=10, p=1.0)], seed=seed)
        out = t(image=img, mask=mask)
        sobrevive += out["mask"][..., 0].sum() > 0
    assert sobrevive >= 27, f"MA apagada em {30 - sobrevive}/30 execuções com ElasticTransform isolada"


def test_vetor_de_validade_preservado_apos_augmentation():
    """Canal sem anotação (tudo zero) continua todo zero após a augmentation."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128, 4), dtype=np.uint8)
    mask[40:50, 40:50, 2] = 1  # só EX tem anotação; HE e SE ficam vazios (inválidos)
    t = build_augmentation(seed=7)
    out = apply_augmentation(t, img, mask)
    assert out["mask"][..., 1].sum() == 0 and out["mask"][..., 3].sum() == 0
