"""Construccion de la muestra a partir de los catalogos MPA-JHU DR8.

Flujo: descargar los tres FITS -> leer solo las columnas necesarias ->
cruzar por SPECOBJID -> aplicar cortes de calidad y fisicos -> guardar parquet.

El target es OH_P50: la mediana de la distribucion de verosimilitud de
12 + log(O/H) que ajusta el pipeline MPA-JHU siguiendo Tremonti et al. (2004).
Es el mismo target que usa Wu & Boada (2019).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .config import Config, ruta_absoluta
from .utils import descargar_archivo

log = logging.getLogger(__name__)

# Columnas que necesitamos de cada catalogo. Leer solo estas evita cargar
# 1.7 GB de galSpecLine en memoria.
COLS_INFO = ["SPECOBJID", "RA", "DEC", "Z", "SN_MEDIAN"]
# Columnas de calidad que usamos si el catalogo las trae. No estan verificadas
# contra DR8, asi que se piden aparte en vez de hacer fallar toda la corrida.
COLS_INFO_OPCIONALES = ["Z_WARNING", "RELIABLE"]
COLS_EXTRA = ["SPECOBJID", "OH_P50", "LGM_TOT_P50", "SFR_TOT_P50", "BPTCLASS"]


def _a_nativo(arr: np.ndarray) -> np.ndarray:
    """Pasa un array de FITS (big-endian) al orden de bytes nativo.

    Sin esto pandas levanta 'Big-endian buffer not supported' en x86.
    """
    if arr.dtype.byteorder in (">", "<") and arr.dtype.byteorder != "=":
        return arr.astype(arr.dtype.newbyteorder("="))
    return arr


def leer_columnas(
    ruta: Path, columnas: list[str], opcionales: list[str] | None = None
) -> pd.DataFrame:
    """Lee un subconjunto de columnas de un FITS de tabla, via memmap.

    Las columnas de ``columnas`` son obligatorias y su ausencia es un error.
    Las de ``opcionales`` se incluyen solo si el catalogo las trae.
    """
    with fits.open(ruta, memmap=True) as hdul:
        datos = hdul[1].data
        disponibles = set(datos.columns.names)
        faltantes = [c for c in columnas if c not in disponibles]
        if faltantes:
            raise KeyError(
                f"{ruta.name} no tiene las columnas {faltantes}. "
                f"Disponibles: {sorted(disponibles)[:20]}..."
            )
        presentes = list(columnas)
        for col in opcionales or []:
            if col in disponibles:
                presentes.append(col)
            else:
                log.warning("%s no tiene la columna %s; se omite ese corte", ruta.name, col)
        df = pd.DataFrame({c: _a_nativo(np.asarray(datos[c])) for c in presentes})

    # SPECOBJID lo normalizamos a str para que los merges y el parquet se
    # comporten de forma predecible. Segun la version de astropy puede llegar
    # como bytes o ya como unicode, asi que se cubren ambos casos.
    if "SPECOBJID" in df.columns:
        col = df["SPECOBJID"]
        if col.dtype == object and len(col) and isinstance(col.iloc[0], bytes):
            col = col.str.decode("utf-8")
        df["SPECOBJID"] = col.astype(str).str.strip()
    return df


def descargar_catalogos(cfg: Config) -> dict[str, Path]:
    """Descarga los tres FITS si no estan ya en disco."""
    destino = ruta_absoluta(cfg.catalogo.dir_local)
    destino.mkdir(parents=True, exist_ok=True)

    rutas = {}
    for clave, nombre in cfg.catalogo.archivos.items():
        ruta = destino / nombre
        url = f"{cfg.catalogo.base_url}/{nombre}"
        descargar_archivo(url, ruta, descripcion=nombre)
        rutas[clave] = ruta
    return rutas


def _unir(*tablas: pd.DataFrame) -> pd.DataFrame:
    """Une los catalogos por posicion de fila, no por SPECOBJID.

    Los tres archivos de MPA-JHU son paralelos: misma longitud y misma fila
    para el mismo espectro. Cruzarlos con un merge ademas de innecesario es
    peligroso, porque ~370k filas de DR8 traen el SPECOBJID en blanco y el
    join las multiplicaria entre si (en DR8 eso son ~1.4e11 filas).

    La alineacion se verifica en vez de asumirse.
    """
    base = tablas[0]
    for otra in tablas[1:]:
        if len(otra) != len(base):
            raise ValueError(
                f"Los catalogos no tienen el mismo largo ({len(base)} vs {len(otra)}); "
                "no se pueden unir por posicion."
            )
        if not base["SPECOBJID"].equals(otra["SPECOBJID"]):
            raise ValueError(
                "Los catalogos no estan alineados fila a fila: SPECOBJID difiere. "
                "Revisa que los tres archivos sean del mismo data release."
            )

    columnas = [tablas[0]] + [t.drop(columns=["SPECOBJID"]) for t in tablas[1:]]
    return pd.concat(columnas, axis=1)


def _columnas_linea(lineas: list[str]) -> list[str]:
    cols = ["SPECOBJID"]
    for linea in lineas:
        cols += [f"{linea}_FLUX", f"{linea}_FLUX_ERR"]
    return cols


def construir_muestra(cfg: Config) -> pd.DataFrame:
    """Aplica toda la cadena de cortes y devuelve la muestra final.

    Cada corte se loguea con cuantas galaxias sobreviven, para que la seleccion
    quede documentada y sea auditable.
    """
    rutas = descargar_catalogos(cfg)
    sel = cfg.seleccion

    log.info("Leyendo catalogos...")
    info = leer_columnas(rutas["info"], COLS_INFO, COLS_INFO_OPCIONALES)
    extra = leer_columnas(rutas["extra"], COLS_EXTRA)
    line = leer_columnas(rutas["line"], _columnas_linea(sel.lineas_snr))
    log.info("Catalogo base: %d espectros", len(info))

    df = _unir(info, extra, line)
    registro = [("catalogo completo", len(df))]

    # ~370k filas de DR8 traen SPECOBJID en blanco (espectros sin contraparte
    # espectroscopica util). Se descartan aca: si se intentara cruzar por esa
    # columna, todas esas filas colisionarian entre si.
    df = df[df["SPECOBJID"] != ""]
    registro.append(("SPECOBJID valido", len(df)))

    # --- Calidad del espectro ---
    if "Z_WARNING" in df.columns:
        df = df[df["Z_WARNING"] == 0]
        registro.append(("redshift sin warnings", len(df)))

    if "RELIABLE" in df.columns:
        df = df[df["RELIABLE"] != 0]
        registro.append(("espectro marcado confiable", len(df)))

    df = df[df["SN_MEDIAN"] > sel.sn_median_minimo]
    registro.append((f"S/N mediana > {sel.sn_median_minimo}", len(df)))

    # --- S/N por linea de emision ---
    # Las cuatro lineas alimentan el estimador de metalicidad; si alguna es
    # ruidosa, OH_P50 no es confiable.
    for linea in sel.lineas_snr:
        snr = df[f"{linea}_FLUX"] / df[f"{linea}_FLUX_ERR"]
        df = df[snr > sel.snr_minimo]
    registro.append((f"S/N > {sel.snr_minimo} en {len(sel.lineas_snr)} lineas", len(df)))

    # --- Solo star-forming (BPT) ---
    df = df[df["BPTCLASS"] == sel.bptclass]
    registro.append(("star-forming (BPT)", len(df)))

    # --- Cortes fisicos ---
    df = df[(df["OH_P50"] > sel.oh_min) & (df["OH_P50"] < sel.oh_max)]
    registro.append((f"metalicidad en ({sel.oh_min}, {sel.oh_max})", len(df)))

    df = df[(df["Z"] > sel.z_min) & (df["Z"] < sel.z_max)]
    registro.append((f"redshift en ({sel.z_min}, {sel.z_max})", len(df)))

    # --- Coordenadas validas ---
    df = df[np.isfinite(df["RA"]) & np.isfinite(df["DEC"])]
    registro.append(("coordenadas finitas", len(df)))

    log.info("Cadena de seleccion:")
    previo = registro[0][1]
    for etiqueta, n in registro:
        log.info("  %-42s %8d  (%5.1f%%)", etiqueta, n, 100 * n / previo)

    if df.empty:
        raise RuntimeError("Ninguna galaxia sobrevivio a los cortes.")

    # --- Muestreo ---
    # Distribucion natural: si hay que recortar, se hace al azar, sin tocar la
    # forma de la distribucion de metalicidad.
    n_objetivo = cfg.muestreo.n_galaxias
    if n_objetivo is not None and len(df) > n_objetivo:
        df = df.sample(n=n_objetivo, random_state=cfg.muestreo.semilla)
        log.info("Submuestreo aleatorio a %d galaxias", n_objetivo)

    df = df.reset_index(drop=True)
    df["img_id"] = df.index

    columnas_salida = [
        "img_id", "SPECOBJID", "RA", "DEC", "Z",
        "OH_P50", "LGM_TOT_P50", "SFR_TOT_P50", "SN_MEDIAN",
    ]
    df = df[columnas_salida].rename(
        columns={"OH_P50": "metalicidad", "LGM_TOT_P50": "log_masa", "SFR_TOT_P50": "log_sfr"}
    )

    log.info(
        "Muestra final: %d galaxias | metalicidad %.3f - %.3f (mediana %.3f, sigma %.3f)",
        len(df), df["metalicidad"].min(), df["metalicidad"].max(),
        df["metalicidad"].median(), df["metalicidad"].std(),
    )
    return df


def guardar(df: pd.DataFrame, cfg: Config) -> Path:
    salida = ruta_absoluta(cfg.catalogo.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(salida, index=False)
    log.info("Muestra guardada en %s", salida)
    return salida


def cargar_muestra(cfg: Config) -> pd.DataFrame:
    """Lee la muestra ya construida."""
    ruta = ruta_absoluta(cfg.catalogo.salida)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Corre primero scripts/01_construir_catalogo.py"
        )
    return pd.read_parquet(ruta)
