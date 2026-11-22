"""Verifica ambiente: bibliotecas, dispositivo e um forward/backward da UNet++ com VGG16."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations
import cv2
import matplotlib
import numpy
import pandas
import segmentation_models_pytorch as smp
import sklearn
import torch
import torchmetrics

from src.seed import get_device, set_seed

set_seed()
device = get_device()

print("python        :", sys.version.split()[0])
for name, mod in [
    ("torch", torch), ("smp", smp), ("albumentations", albumentations),
    ("opencv", cv2), ("numpy", numpy), ("pandas", pandas),
    ("matplotlib", matplotlib), ("scikit-learn", sklearn), ("torchmetrics", torchmetrics),
]:
    print(f"{name:14}: {mod.__version__}")
print("dispositivo   :", device)

# encoder sem baixar pesos, só para checar que a arquitetura roda
model = smp.UnetPlusPlus(encoder_name="vgg16", encoder_weights=None, in_channels=3, classes=4).to(device)
x = torch.randn(1, 3, 256, 256, device=device)
y = model(x)
y.mean().backward()
print("saída         :", tuple(y.shape), "-> OK")
