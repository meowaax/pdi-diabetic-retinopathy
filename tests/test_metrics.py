import numpy as np
import pandas as pd

from src.data import CLASSES
from src.metrics import bootstrap_ci, macro_dice, per_image_metrics, pool_confusion, rates_from_confusion, summarize_with_ci


def _caso(n=2, h=8, w=8):
    pred = np.zeros((n, 4, h, w), dtype=np.uint8)
    target = np.zeros((n, 4, h, w), dtype=np.uint8)
    valid = np.ones((n, 4), dtype=np.uint8)
    return pred, target, valid


def test_predicao_perfeita_da_dice_e_iou_1():
    pred, target, valid = _caso()
    target[:, 0, :4, :4] = 1
    pred[:, 0, :4, :4] = 1
    c = pool_confusion(pred, target, valid)
    r = rates_from_confusion(c)
    row = r[r.cls == "MA"].iloc[0]
    assert abs(row.dice - 1) < 1e-4 and abs(row.iou - 1) < 1e-4 and abs(row.precision - 1) < 1e-4
    assert abs(row.sensitivity - 1) < 1e-4


def test_predicao_vazia_sem_lesao_real_da_metricas_proximas_de_1_pelo_eps():
    """Sem TP, FP, FN (tudo zero): dice/iou/precisão/sensibilidade ficam próximas de 1 (eps/eps)."""
    pred, target, valid = _caso()
    c = pool_confusion(pred, target, valid)
    r = rates_from_confusion(c)
    assert (r.dice > 0.99).all() and (r.iou > 0.99).all()


def test_predicao_vazia_com_lesao_real_da_dice_zero():
    pred, target, valid = _caso()
    target[:, 1, :4, :4] = 1  # lesão real de HE, nada predito
    c = pool_confusion(pred, target, valid)
    r = rates_from_confusion(c)
    row = r[r.cls == "HE"].iloc[0]
    assert row.dice < 1e-3 and row.iou < 1e-3 and row.sensitivity < 1e-3


def test_classe_invalida_ignorada_no_pool():
    pred, target, valid = _caso()
    target[0, 2, :4, :4] = 1
    pred[0, 2, :2, :2] = 1  # acerta só 1/4 da lesão -> dice baixo se contasse
    valid[0, 2] = 0  # mas a imagem 0 não tem anotação de EX
    target[1, 2, :4, :4] = 1
    pred[1, 2, :4, :4] = 1  # imagem 1 acerta tudo
    c = pool_confusion(pred, target, valid)
    row = rates_from_confusion(c)[rates_from_confusion(c).cls == "EX"].iloc[0]
    assert abs(row.dice - 1) < 1e-4  # só a imagem 1 (válida) conta, e ela é perfeita


def test_macro_dice_e_a_media_das_4_classes():
    pred, target, valid = _caso()
    target[:, 0, :4, :4] = 1
    pred[:, 0, :4, :4] = 1  # MA perfeita
    target[:, 1, :4, :4] = 1  # HE sem predição -> dice ~0
    c = pool_confusion(pred, target, valid)
    r = rates_from_confusion(c)
    esperado = r.dice.mean()
    assert abs(macro_dice(c) - esperado) < 1e-9


def test_per_image_metrics_ignora_entradas_invalidas():
    pred, target, valid = _caso(n=3)
    valid[1, 3] = 0
    target[:, 3, :2, :2] = 1
    pred[:, 3, :2, :2] = 1
    df = per_image_metrics(pred, target, valid, image_ids=["a", "b", "c"])
    se = df[df.cls == "SE"]
    assert set(se.image_id) == {"a", "c"}
    assert np.allclose(se.dice, 1.0, atol=1e-3)


def test_bootstrap_ci_contem_a_media_e_encolhe_com_menos_variancia():
    rng = np.random.default_rng(0)
    alta_var = rng.uniform(0, 1, 60)
    mean, lo, hi = bootstrap_ci(alta_var, n_boot=1000)
    assert lo <= mean <= hi

    baixa_var = np.full(60, 0.8)
    _, lo2, hi2 = bootstrap_ci(baixa_var, n_boot=1000)
    assert (hi2 - lo2) < (hi - lo)


def test_bootstrap_ci_lista_vazia_nao_quebra():
    mean, lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_summarize_with_ci_cobre_as_4_classes_mesmo_sem_amostra():
    df = pd.DataFrame({"image_id": ["a"], "cls": ["MA"], "dice": [1.0], "iou": [1.0],
                        "precision": [1.0], "sensitivity": [1.0]})
    out = summarize_with_ci(df, metric="dice")
    assert set(out.cls) == set(CLASSES)
    se = out[out.cls == "SE"]
    assert int(se.n.iloc[0]) == 0 and np.isnan(float(se["mean"].iloc[0]))
