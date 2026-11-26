"""Loss BCE + Dice por classe, com supervisão mascarada (issue #10).

`valid` tem forma (N, C): 1 se a classe C da imagem N tem anotação, 0 se a máscara está ausente
(ver src/data.py). Uma classe inválida não deve contribuir para a loss nem gerar gradiente — uma
máscara ausente no IDRiD não significa ausência de lesão (planejamento, seção 3.1).

BCE é calculada por pixel e depois reduzida por imagem/classe antes do mascaramento, para que uma
classe inválida não puxe a média de pixels por nada (BCE "verdadeira" seria indefinida, não zero).
Dice é calculada por imagem/classe diretamente (já é uma razão por amostra). As duas reduções
ponderadas por `valid` descartam por completo as entradas inválidas da média final.
"""
import torch
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """logits, target: (N, C, H, W). valid: (N, C). Devolve escalar: média da BCE por imagem/classe,
    ponderada por valid (equivalente a calcular a BCE só sobre as entradas válidas)."""
    per_sample = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(2, 3))  # (N, C)
    return (per_sample * valid).sum() / (valid.sum() + eps)


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - Dice, por imagem/classe, ponderado por valid."""
    proba = torch.sigmoid(logits)
    inter = (proba * target).sum(dim=(2, 3))
    union = proba.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    per_sample = 1 - dice  # (N, C)
    return (per_sample * valid).sum() / (valid.sum() + eps)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                   bce_weight: float = 0.5) -> torch.Tensor:
    """Combinação convexa de BCE e Dice, ambas mascaradas. bce_weight em [0, 1]."""
    bce = masked_bce(logits, target, valid)
    dice = masked_dice_loss(logits, target, valid)
    return bce_weight * bce + (1 - bce_weight) * dice
