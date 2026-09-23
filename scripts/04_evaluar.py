#!/usr/bin/env python
"""Paso 4: evalua el mejor checkpoint sobre el test y genera las figuras.

    python scripts/04_evaluar.py --nombre resnet34

Es el unico paso que toca el conjunto de test.
"""

from __future__ import annotations

import argparse

import torch

from gasmet import dataset, evaluate, model, plots
from gasmet.config import cargar, ruta_absoluta
from gasmet.utils import configurar_logging, fijar_semilla


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--nombre", default="resnet34")
    parser.add_argument("--arquitectura", default=None)
    args = parser.parse_args()

    configurar_logging()
    cfg = cargar(args.config)
    if args.arquitectura:
        cfg["modelo"]["arquitectura"] = args.arquitectura
    fijar_semilla(cfg.entrenamiento.semilla)

    imagenes, meta = dataset.cargar_datos(cfg)
    stats = dataset.cargar_estadisticas(cfg)
    loaders = dataset.construir_loaders(cfg, imagenes, meta, stats)

    dir_salida = ruta_absoluta(cfg.entrenamiento.dir_resultados) / args.nombre
    checkpoint = torch.load(dir_salida / "checkpoints" / "mejor.pt", map_location="cpu")

    red = model.construir_modelo(cfg)
    red.load_state_dict(checkpoint["modelo"])

    evaluate.evaluar(cfg, red, loaders, meta, stats, dir_salida=dir_salida)
    plots.generar_todas(cfg, meta, dir_salida)


if __name__ == "__main__":
    main()
