#!/usr/bin/env python
"""Paso 3: entrena la CNN.

    python scripts/03_entrenar.py
    python scripts/03_entrenar.py --arquitectura cnn_simple --nombre baseline

Salidas en results/<nombre>/: checkpoints/mejor.pt, historial.json
"""

from __future__ import annotations

import argparse

from gasmet import dataset, model, train
from gasmet.config import cargar, ruta_absoluta
from gasmet.utils import configurar_logging, fijar_semilla


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--nombre", default="resnet34", help="subcarpeta en results/")
    parser.add_argument("--arquitectura", default=None, help="sobrescribe modelo.arquitectura")
    parser.add_argument("--epocas", type=int, default=None)
    args = parser.parse_args()

    configurar_logging()
    cfg = cargar(args.config)
    if args.arquitectura:
        cfg["modelo"]["arquitectura"] = args.arquitectura
    if args.epocas:
        cfg["entrenamiento"]["epocas"] = args.epocas

    fijar_semilla(cfg.entrenamiento.semilla)

    imagenes, meta = dataset.cargar_datos(cfg)
    stats = dataset.cargar_estadisticas(cfg)
    loaders = dataset.construir_loaders(cfg, imagenes, meta, stats)

    red = model.construir_modelo(cfg)
    dir_salida = ruta_absoluta(cfg.entrenamiento.dir_resultados) / args.nombre
    train.entrenar(cfg, red, loaders, stats, dir_salida=dir_salida)


if __name__ == "__main__":
    main()
