import torch

from src.losses import bce_dice_loss, masked_bce, masked_dice_loss


def _dados(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, 4, 16, 16, generator=g)
    target = torch.randint(0, 2, (3, 4, 16, 16), generator=g).float()
    return logits, target


def test_classe_invalida_nao_contribui_para_bce_nem_dice():
    logits, target = _dados()
    valid = torch.ones(3, 4)
    valid[1, 2] = 0  # imagem 1, classe 2 (EX) sem anotação

    # substituir o conteúdo da entrada inválida por qualquer coisa não deve mudar a loss
    logits_mod = logits.clone()
    logits_mod[1, 2] = torch.randn(16, 16) * 50  # logits absurdos na entrada mascarada
    target_mod = target.clone()
    target_mod[1, 2] = 1 - target[1, 2]  # máscara invertida na entrada mascarada

    for fn in (masked_bce, masked_dice_loss):
        a = fn(logits, target, valid)
        b = fn(logits_mod, target_mod, valid)
        assert torch.allclose(a, b, atol=1e-6), fn.__name__


def test_classe_invalida_nao_gera_gradiente():
    logits, target = _dados()
    valid = torch.zeros(3, 4)
    valid[:, 0] = 1  # só MA é válida
    logits.requires_grad_(True)

    loss = bce_dice_loss(logits, target, valid)
    loss.backward()

    assert (logits.grad[:, 0] != 0).any()
    assert torch.allclose(logits.grad[:, 1:], torch.zeros_like(logits.grad[:, 1:]), atol=1e-7)


def test_com_todas_as_classes_validas_bce_mascarada_igual_a_bce_padrao():
    logits, target = _dados()
    valid = torch.ones(3, 4)
    import torch.nn.functional as F

    esperado = F.binary_cross_entropy_with_logits(logits, target)
    assert torch.allclose(masked_bce(logits, target, valid), esperado, atol=1e-6)


def test_com_todas_as_classes_validas_dice_mascarada_igual_a_dice_padrao_por_amostra():
    logits, target = _dados()
    valid = torch.ones(3, 4)
    proba = torch.sigmoid(logits)
    inter = (proba * target).sum(dim=(2, 3))
    union = proba.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    esperado = (1 - (2 * inter + 1e-6) / (union + 1e-6)).mean()
    assert torch.allclose(masked_dice_loss(logits, target, valid), esperado, atol=1e-6)


def test_predicao_perfeita_da_dice_loss_proxima_de_zero():
    target = torch.randint(0, 2, (2, 4, 16, 16)).float()
    logits = (target * 20 - 10)  # sigmoid(logits) ~ target
    valid = torch.ones(2, 4)
    assert masked_dice_loss(logits, target, valid).item() < 1e-2


def test_bce_weight_interpola_entre_bce_pura_e_dice_pura():
    logits, target = _dados()
    valid = torch.ones(3, 4)
    bce = masked_bce(logits, target, valid)
    dice = masked_dice_loss(logits, target, valid)
    assert torch.allclose(bce_dice_loss(logits, target, valid, bce_weight=1.0), bce, atol=1e-6)
    assert torch.allclose(bce_dice_loss(logits, target, valid, bce_weight=0.0), dice, atol=1e-6)


def test_toda_a_imagem_invalida_nao_gera_nan():
    logits, target = _dados()
    valid = torch.zeros(3, 4)
    loss = bce_dice_loss(logits, target, valid)
    assert torch.isfinite(loss)
