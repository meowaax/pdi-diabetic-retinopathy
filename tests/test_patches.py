import numpy as np
import pytest

from src.data import CLASSES
from src.patches import IDRiDPatches, grid_positions, stitch
from src.split import fold_ids, ids_for

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("data/cache/IDRiD_01_meta.npz").exists(), reason="rode scripts/build_cache.py"
)


def test_grid_cobre_a_imagem_e_encosta_nas_bordas():
    pos = grid_positions(2848, 4288, 512, 256)
    assert max(y for y, _ in pos) == 2848 - 512 and max(x for _, x in pos) == 4288 - 512
    cov = np.zeros((2848, 4288), bool)
    for y, x in pos:
        cov[y:y + 512, x:x + 512] = True
    assert cov.all()


def test_stitch_de_patches_recompõe_a_imagem():
    rng = np.random.default_rng(0)
    full = rng.normal(size=(2, 700, 900)).astype(np.float32)
    pos = grid_positions(700, 900, 512, 256)
    patches = np.stack([full[:, y:y + 512, x:x + 512] for y, x in pos])
    assert np.allclose(stitch(patches, pos, (700, 900)), full, atol=1e-5)


def test_formato_e_validade():
    ds = IDRiDPatches(["IDRiD_43", "IDRiD_03"], mode="train", patches_per_image=2)
    assert len(ds) == 4
    for i in range(len(ds)):
        s = ds[i]
        assert s["image"].shape == (3, 512, 512) and s["mask"].shape == (4, 512, 512) and s["valid"].shape == (4,)
        if s["image_id"] == "IDRiD_43":
            assert s["valid"].tolist() == [1, 0, 1, 0] and s["mask"][1].sum() == 0 and s["mask"][3].sum() == 0


def test_amostragem_centrada_em_lesao_acerta_lesao():
    ds = IDRiDPatches(["IDRiD_03"], mode="train", patches_per_image=40, lesion_prob=1.0)
    com_lesao = sum(ds[i]["mask"].sum() > 0 for i in range(len(ds)))
    assert com_lesao >= 38


def test_determinismo_por_epoca():
    ds = IDRiDPatches(["IDRiD_03", "IDRiD_05"], mode="train", patches_per_image=3)
    a = [(ds[i]["y"], ds[i]["x"]) for i in range(len(ds))]
    b = [(ds[i]["y"], ds[i]["x"]) for i in range(len(ds))]
    ds.set_epoch(1)
    c = [(ds[i]["y"], ds[i]["x"]) for i in range(len(ds))]
    assert a == b and a != c


def test_eval_cobre_a_mascara_inteira_da_imagem():
    """Toda lesão do FOV cai em ao menos um patch de avaliação (recomposição sem perda)."""
    from src.patches import _unpack
    ds = IDRiDPatches(["IDRiD_05"], mode="eval")
    lab = np.load("data/cache/IDRiD_05_lab.npy")
    total = (_unpack(lab)[0].sum(axis=(0, 1)))
    cov = np.zeros(lab.shape, bool)
    for _, y, x in ds.items:
        cov[y:y + 512, x:x + 512] = True
    kept = (_unpack(lab)[0] * cov[..., None]).sum(axis=(0, 1))
    assert (kept == total).all()


def test_sem_vazamento_entre_conjuntos():
    for k in range(5):
        tr, va = fold_ids(k)
        a = IDRiDPatches(tr[:2], mode="eval"); b = IDRiDPatches(va[:2], mode="eval")
        assert not {i for i, _, _ in a.items} & {i for i, _, _ in b.items}
    assert not set(ids_for("test")) & set(ids_for("train") + ids_for("val"))
