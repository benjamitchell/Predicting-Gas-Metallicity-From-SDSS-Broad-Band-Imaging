"""Descarga y empaquetado de cutouts de SDSS.

El servicio ImgCutout devuelve un JPEG RGB: una composicion Lupton de las
bandas g, r e i. Son TRES canales con informacion distinta.

Advertencia importante, y es el error que invalido el intento anterior del
proyecto (ver docs/legacy/): este endpoint NO acepta un parametro `band`. Si se
le pasa `band=u`, `band=g`, etc., lo ignora en silencio y devuelve siempre la
misma imagen, con codigo 200 y sin ningun warning. Pedir cinco "bandas" por
galaxia entrega cinco archivos byte-identicos.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from .config import Config, ruta_absoluta
from .utils import reintentar

log = logging.getLogger(__name__)

# Parametros que el endpoint si reconoce. Cualquier otro se ignora en silencio.
PARAMS_VALIDOS = {"ra", "dec", "scale", "width", "height", "opt", "query"}


def url_cutout(cfg: Config, ra: float, dec: float) -> str:
    params = {
        "ra": f"{ra:.6f}",
        "dec": f"{dec:.6f}",
        "scale": cfg.imagenes.escala_arcsec_px,
        "width": cfg.imagenes.tamano_px,
        "height": cfg.imagenes.tamano_px,
    }
    assert set(params) <= PARAMS_VALIDOS, "parametro no soportado por ImgCutout"
    consulta = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{cfg.imagenes.url}?{consulta}"


def _ruta_imagen(dir_base: Path, img_id: int) -> Path:
    """Agrupa en subcarpetas de 1000 para no dejar 100k archivos en un solo dir."""
    sub = dir_base / f"{img_id // 1000:04d}"
    return sub / f"galaxia_{img_id:06d}.jpg"


def descargar_una(
    cfg: Config, sesion: requests.Session, dir_base: Path, fila: pd.Series
) -> tuple[int, bool]:
    """Descarga el cutout de una galaxia. Devuelve (img_id, exito)."""
    destino = _ruta_imagen(dir_base, int(fila["img_id"]))
    if destino.exists() and destino.stat().st_size > 0:
        return int(fila["img_id"]), True

    url = url_cutout(cfg, fila["RA"], fila["DEC"])
    esperado = (cfg.imagenes.tamano_px, cfg.imagenes.tamano_px)

    def intento() -> bytes:
        resp = sesion.get(url, timeout=cfg.imagenes.timeout_s)
        resp.raise_for_status()
        # El servicio devuelve 200 con un JPEG de error si las coordenadas caen
        # fuera del footprint, asi que validamos que sea una imagen del tamano
        # pedido antes de aceptarla.
        img = Image.open(io.BytesIO(resp.content))
        if img.size != esperado:
            raise ValueError(f"tamano inesperado {img.size}, se esperaba {esperado}")
        if img.mode != "RGB":
            raise ValueError(f"modo inesperado {img.mode}")
        return resp.content

    contenido = reintentar(intento, intentos=cfg.imagenes.reintentos)
    if contenido is None:
        return int(fila["img_id"]), False

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(contenido)
    return int(fila["img_id"]), True


def descargar_muestra(cfg: Config, catalogo: pd.DataFrame) -> pd.DataFrame:
    """Descarga los cutouts de todo el catalogo en paralelo.

    Devuelve el catalogo con una columna ``imagen_ok`` que marca los fallos,
    para que el dataset los pueda excluir explicitamente en vez de tropezar
    con archivos faltantes durante el entrenamiento.
    """
    dir_base = ruta_absoluta(cfg.imagenes.dir_local)
    dir_base.mkdir(parents=True, exist_ok=True)

    resultados: dict[int, bool] = {}
    with requests.Session() as sesion:
        with ThreadPoolExecutor(max_workers=cfg.imagenes.workers) as pool:
            futuros = [
                pool.submit(descargar_una, cfg, sesion, dir_base, fila)
                for _, fila in catalogo.iterrows()
            ]
            for fut in tqdm(
                as_completed(futuros), total=len(futuros), desc="cutouts SDSS"
            ):
                img_id, ok = fut.result()
                resultados[img_id] = ok

    catalogo = catalogo.copy()
    catalogo["imagen_ok"] = catalogo["img_id"].map(resultados).fillna(False)

    n_ok = int(catalogo["imagen_ok"].sum())
    log.info(
        "Descargadas %d/%d imagenes (%d fallos)",
        n_ok, len(catalogo), len(catalogo) - n_ok,
    )
    return catalogo


def empaquetar(cfg: Config, catalogo: pd.DataFrame) -> tuple[Path, Path]:
    """Convierte los JPEG a un unico array memmap uint8 (N, H, W, 3).

    Leer 100k JPEG sueltos por epoca satura el sistema de archivos de un nodo
    de cluster. Un memmap se lee sin descomprimir y sin castigar al disco.
    """
    dir_base = ruta_absoluta(cfg.imagenes.dir_local)
    dir_salida = ruta_absoluta("data/processed")
    dir_salida.mkdir(parents=True, exist_ok=True)

    validos = catalogo[catalogo["imagen_ok"]].reset_index(drop=True)
    n, px = len(validos), cfg.imagenes.tamano_px

    ruta_imgs = dir_salida / "imagenes.npy"
    ruta_meta = dir_salida / "metadatos.parquet"

    arr = np.lib.format.open_memmap(
        ruta_imgs, mode="w+", dtype=np.uint8, shape=(n, px, px, 3)
    )
    for i, img_id in enumerate(tqdm(validos["img_id"], desc="empaquetando")):
        with Image.open(_ruta_imagen(dir_base, int(img_id))) as img:
            arr[i] = np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr.flush()

    # El indice de fila en el memmap es la llave; se guarda explicito para que
    # metadatos e imagenes no se puedan desalinear.
    validos["fila"] = np.arange(n)
    validos.to_parquet(ruta_meta, index=False)

    log.info("Empaquetadas %d imagenes en %s (%.1f GB)", n, ruta_imgs, arr.nbytes / 1e9)
    return ruta_imgs, ruta_meta
