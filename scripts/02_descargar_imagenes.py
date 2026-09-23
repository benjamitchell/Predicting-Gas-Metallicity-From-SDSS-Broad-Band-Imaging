#!/usr/bin/env python
"""Paso 2: descarga los cutouts de SDSS y los empaqueta para entrenar.

    python scripts/02_descargar_imagenes.py

Un request por galaxia al servicio ImgCutout. Para 100k galaxias con 8 workers
toma del orden de horas; la descarga es reanudable, las imagenes ya bajadas
se saltan.

Salidas:
    data/images/....jpg            cutouts individuales
    data/processed/imagenes.npy    memmap uint8 (N, px, px, 3)
    data/processed/metadatos.parquet
    data/processed/norm_stats.json
"""

from __future__ import annotations

import argparse

import numpy as np

from gasmet import catalog, dataset, images
from gasmet.config import cargar
from gasmet.utils import configurar_logging, fijar_semilla


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--limite", type=int, default=None,
        help="descargar solo las primeras N galaxias (util para probar)",
    )
    args = parser.parse_args()

    configurar_logging()
    cfg = cargar(args.config)
    fijar_semilla(cfg.muestreo.semilla)

    muestra = catalog.cargar_muestra(cfg)
    if args.limite:
        muestra = muestra.head(args.limite)

    muestra = images.descargar_muestra(cfg, muestra)
    _, ruta_meta = images.empaquetar(cfg, muestra)

    # Splits y estadisticas de normalizacion, calculadas solo sobre train.
    import pandas as pd

    meta = pd.read_parquet(ruta_meta)
    meta = dataset.hacer_splits(cfg, meta)
    meta.to_parquet(ruta_meta, index=False)

    imgs = np.load(ruta_meta.parent / "imagenes.npy", mmap_mode="r")
    dataset.calcular_estadisticas(cfg, imgs, meta)


if __name__ == "__main__":
    main()
