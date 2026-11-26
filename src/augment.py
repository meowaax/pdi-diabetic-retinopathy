"""Augmentation fixa (issue #8), herdada do artigo de referência (CBIC 2025): melhor combinação
para UNet++/VGG16 em EX no IDRiD (IoU 0,510; Dice 0,676) — HorizontalFlip + ElasticTransform +
RandomBrightnessContrast, cada técnica aplicada com probabilidade 50% (independente das outras).

Excluídas deliberadamente (pioraram no artigo): RGBShift (piorou em toda combinação) e
ShiftScaleRotate (pior técnica isolada).

Parâmetros do artigo:
- RandomBrightnessContrast: brilho e contraste variam até 30% cada.
- ElasticTransform: alpha=1.0, sigma=10, alpha_affine=10. A versão instalada do Albumentations
  (2.x) não tem mais o parâmetro alpha_affine (a distorção afim foi descontinuada nesta
  transformação); usamos alpha e sigma do artigo e registramos o desvio.

Atua sobre o RGB original, antes do pré-processamento (src/preprocess.py), conforme a ordem fixada
nesta issue. Usada em E1 a E5; não é reajustada por experimento.
"""
import albumentations as A

ELASTIC_ALPHA = 1.0
ELASTIC_SIGMA = 10
BRIGHTNESS_CONTRAST_LIMIT = 0.3
P_EACH = 0.5


def build_augmentation(seed: int = 42) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=P_EACH),
            A.ElasticTransform(alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA, p=P_EACH),
            A.RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS_CONTRAST_LIMIT, contrast_limit=BRIGHTNESS_CONTRAST_LIMIT, p=P_EACH
            ),
        ],
        seed=seed,
    )


def apply_augmentation(transform: A.Compose, image, mask):
    """Adapta A.Compose(image=, mask=) à assinatura esperada por IDRiDPatches (image, mask) -> dict.

    mask tem formato (H, W, 4); RandomBrightnessContrast/HorizontalFlip/ElasticTransform não
    alteram a máscara em valor, só a geometria (flip/distorção), preservando o vetor de validade.
    """
    out = transform(image=image, mask=mask)
    return {"image": out["image"], "mask": out["mask"]}
