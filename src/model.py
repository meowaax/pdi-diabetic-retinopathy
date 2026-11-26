"""Modelo de segmentação: UNet++ com encoder VGG16 pré-treinado e saída multirrótulo (issue #9).

Saída com 4 canais (MA, HE, EX, SE), ativação sigmoid aplicada fora do forward (logits brutos),
para compor com a loss BCE+Dice mascarada da issue #10, que espera logits.
"""
import segmentation_models_pytorch as smp
import torch

from .data import CLASSES

ENCODER_NAME = "vgg16"
ENCODER_WEIGHTS = "imagenet"


def build_model(encoder_weights: str | None = ENCODER_WEIGHTS) -> smp.UnetPlusPlus:
    """Devolve logits (N, 4, H, W), uma ordem fixa de classes = CLASSES = [MA, HE, EX, SE].

    encoder_weights=None evita o download do Hugging Face (útil em testes/CI); use "imagenet"
    para treino real.
    """
    return smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=len(CLASSES),
        activation=None,
    )


def predict_proba(model: smp.UnetPlusPlus, x: torch.Tensor) -> torch.Tensor:
    """Sigmoid sobre os logits; cada canal é independente (multirrótulo, não softmax)."""
    return torch.sigmoid(model(x))


# Notas de memória (medidas em Mac 16 GB, GPU MPS; patches 512x512, encoder_weights="imagenet"):
#   batch 4: OK (~0,6 GB alocados pelo modelo/ativações rastreadas)
#   batch 6: OK (valor usado no artigo de referência)
#   batch 8: falha por falta de memória nesta máquina
# Em GPU dedicada (Colab/compartilhada) o teto deve ser maior; o loop de treino (#12) deve expor o
# batch size como parâmetro e cair para 4 se 6 não couber. Mixed precision via
# `torch.autocast(device_type=device.type, dtype=torch.float16)` reduziu a necessidade de memória
# e funcionou em MPS e deve funcionar em CUDA; usar escala de gradiente (`torch.cuda.amp.GradScaler`)
# quando o dispositivo for CUDA.
