"""Arquitecturas de red.

Por defecto una ResNet pre-entrenada en ImageNet, siguiendo a Wu & Boada
(2019), que usan una ResNet-34 inicializada con pesos de ImageNet. El transfer
learning importa: entrenando desde cero haria falta un dataset mucho mayor
para aprender los filtros de bajo nivel que ImageNet ya provee.

Se incluye ademas una CNN simple como punto de comparacion honesto.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

from .config import Config

log = logging.getLogger(__name__)

_RESNETS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class CNNSimple(nn.Module):
    """CNN de tres bloques, entrenada desde cero. Baseline, no el modelo final."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        capas = []
        canales = [3, 32, 64, 128]
        for entrada, salida in zip(canales[:-1], canales[1:], strict=True):
            capas += [
                nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
                nn.BatchNorm2d(salida),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*capas)
        # Pooling adaptativo: la cabeza no depende del tamano de entrada.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).squeeze(1)


def _resnet(nombre: str, preentrenado: bool, dropout: float) -> nn.Module:
    constructor, pesos = _RESNETS[nombre]
    red = constructor(weights=pesos if preentrenado else None)
    # Cabeza de regresion: una sola salida escalar, sin activacion final.
    red.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(red.fc.in_features, 1))

    forward_base = red.forward
    red.forward = lambda x: forward_base(x).squeeze(1)  # type: ignore[method-assign]
    return red


def construir_modelo(cfg: Config) -> nn.Module:
    nombre = cfg.modelo.arquitectura.lower()
    dropout = cfg.modelo.dropout

    if nombre == "cnn_simple":
        modelo = CNNSimple(dropout=dropout)
    elif nombre in _RESNETS:
        modelo = _resnet(nombre, cfg.modelo.preentrenado, dropout)
    else:
        raise ValueError(
            f"Arquitectura '{nombre}' desconocida. "
            f"Opciones: cnn_simple, {', '.join(_RESNETS)}"
        )

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    log.info("Modelo: %s (%.1fM parametros entrenables)", nombre, n_params / 1e6)
    return modelo
