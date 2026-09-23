#!/usr/bin/env python
"""Paso 1: descarga los catalogos MPA-JHU y construye la muestra.

    python scripts/01_construir_catalogo.py

Descarga ~2.4 GB la primera vez y los deja cacheados en data/catalog/.
Salida: data/catalog/muestra.parquet
"""

from __future__ import annotations

import argparse

from gasmet import catalog
from gasmet.config import cargar
from gasmet.utils import configurar_logging, fijar_semilla


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="ruta a un config.yaml alternativo")
    args = parser.parse_args()

    configurar_logging()
    cfg = cargar(args.config)
    fijar_semilla(cfg.muestreo.semilla)

    muestra = catalog.construir_muestra(cfg)
    catalog.guardar(muestra, cfg)


if __name__ == "__main__":
    main()
