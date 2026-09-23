"""Carga de config.yaml.

Un unico punto de entrada para los parametros del pipeline: los scripts leen
de aqui y nunca hardcodean valores.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Raiz del repo: este archivo vive en src/gasmet/config.py
RAIZ = Path(__file__).resolve().parents[2]

# Variable de entorno para sacar los datos pesados fuera del repo. Hace falta
# en dos situaciones concretas:
#   - en local, si el repo vive dentro de una carpeta sincronizada (OneDrive,
#     Dropbox), donde 2.4 GB de FITS y 100k imagenes se subirian a la nube;
#   - en un cluster, donde los datos van a scratch y no al home, que suele
#     tener cuota chica.
#
#   export GASMET_DATA_DIR=/scratch/$USER/gasmet-data
VAR_DIR_DATOS = "GASMET_DATA_DIR"


class Config(dict):
    """dict con acceso por atributo, anidado.

    Permite escribir ``cfg.imagenes.tamano_px`` en vez de
    ``cfg["imagenes"]["tamano_px"]``, que se vuelve ilegible al anidar.
    """

    def __getattr__(self, nombre: str) -> Any:
        try:
            valor = self[nombre]
        except KeyError as exc:
            raise AttributeError(
                f"'{nombre}' no existe en la configuracion. "
                f"Claves disponibles: {sorted(self.keys())}"
            ) from exc
        return Config(valor) if isinstance(valor, dict) else valor


def cargar(ruta: str | Path | None = None) -> Config:
    """Lee el YAML de configuracion y lo devuelve como Config."""
    ruta = Path(ruta) if ruta else RAIZ / "config.yaml"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro la configuracion en {ruta}")
    with open(ruta, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def ruta_absoluta(ruta_relativa: str | Path) -> Path:
    """Resuelve una ruta del config contra la raiz del repo.

    Asi los scripts funcionan igual desde cualquier directorio de trabajo, que
    es lo que pasa al lanzarlos con SLURM.

    Si GASMET_DATA_DIR esta definida, las rutas que empiezan en ``data/`` se
    redirigen ahi en vez de quedar dentro del repo.
    """
    ruta = Path(ruta_relativa)
    if ruta.is_absolute():
        return ruta

    partes = ruta.parts
    base_datos = os.environ.get(VAR_DIR_DATOS)
    if base_datos and partes and partes[0] == "data":
        return Path(base_datos).expanduser().resolve().joinpath(*partes[1:])

    return RAIZ / ruta
