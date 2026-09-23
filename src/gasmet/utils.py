"""Utilidades compartidas: logging, semillas y descargas con reintento."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Logging a stdout, con formato legible tambien en los .out de SLURM."""
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def fijar_semilla(semilla: int) -> None:
    """Fija las semillas de random, numpy y torch (si esta disponible)."""
    random.seed(semilla)
    np.random.seed(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)
    try:
        import torch

        torch.manual_seed(semilla)
        torch.cuda.manual_seed_all(semilla)
    except ImportError:
        pass


def descargar_archivo(
    url: str, destino: Path, descripcion: str = "", chunk: int = 1 << 20
) -> Path:
    """Descarga con barra de progreso, saltando el archivo si ya esta completo.

    Escribe primero en un .parcial y renombra al final, de modo que una
    descarga interrumpida nunca deje un archivo truncado que parezca valido.
    """
    destino = Path(destino)
    if destino.exists():
        esperado = _tamano_remoto(url)
        if esperado is None or destino.stat().st_size == esperado:
            log.info("%s ya esta descargado, se omite", descripcion or destino.name)
            return destino
        log.warning(
            "%s tiene tamano inesperado (%d vs %d); se vuelve a descargar",
            destino.name, destino.stat().st_size, esperado,
        )

    parcial = destino.with_suffix(destino.suffix + ".parcial")
    destino.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        barra = tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=descripcion or destino.name,
        )
        with open(parcial, "wb") as fh, barra:
            for bloque in resp.iter_content(chunk_size=chunk):
                fh.write(bloque)
                barra.update(len(bloque))

    parcial.replace(destino)
    return destino


def _tamano_remoto(url: str) -> int | None:
    """Content-length del recurso, o None si el servidor no lo informa."""
    try:
        resp = requests.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        largo = resp.headers.get("content-length")
        return int(largo) if largo else None
    except requests.RequestException:
        return None


def reintentar(fn, intentos: int = 3, espera_base: float = 1.0):
    """Ejecuta fn() con backoff exponencial. Devuelve None si agota intentos."""
    for intento in range(intentos):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - se reintenta ante cualquier fallo de red
            if intento == intentos - 1:
                log.debug("Agotados %d intentos: %s", intentos, exc)
                return None
            time.sleep(espera_base * (2 ** intento))
    return None
