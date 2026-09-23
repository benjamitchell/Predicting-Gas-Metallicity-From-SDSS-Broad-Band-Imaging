"""Metricas y figuras de evaluacion.

Se reportan las mismas metricas que Wu & Boada (2019) para que los numeros
sean directamente comparables: RMSE y NMAD sobre el residuo en dex.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, ruta_absoluta

log = logging.getLogger(__name__)


def nmad(x: np.ndarray) -> float:
    """Desviacion absoluta mediana normalizada.

    Para datos gaussianos aproxima sigma, pero es insensible a outliers.
    Es la metrica robusta que reporta el paper junto al RMSE.
    """
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuo = y_pred - y_real
    ss_res = float(np.sum(residuo ** 2))
    ss_tot = float(np.sum((y_real - y_real.mean()) ** 2))
    return {
        "rmse_dex": float(np.sqrt(np.mean(residuo ** 2))),
        "nmad_dex": nmad(residuo),
        "mae_dex": float(np.mean(np.abs(residuo))),
        "sesgo_dex": float(np.mean(residuo)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(len(y_real)),
    }


def metricas_por_bin(
    y_real: np.ndarray, y_pred: np.ndarray, ancho: float = 0.1
) -> pd.DataFrame:
    """Metricas en bins de metalicidad verdadera.

    Con la distribucion natural el grueso de las galaxias vive cerca de 8.8,
    asi que una metrica global esconde el desempeno en los extremos. Esto lo
    hace explicito en vez de taparlo reequilibrando el dataset.
    """
    bordes = np.arange(np.floor(y_real.min() * 10) / 10, y_real.max() + ancho, ancho)
    idx = np.digitize(y_real, bordes) - 1

    filas = []
    for i in range(len(bordes) - 1):
        mascara = idx == i
        if mascara.sum() < 20:  # bins con muy pocos objetos no son informativos
            continue
        fila = metricas(y_real[mascara], y_pred[mascara])
        fila["bin_centro"] = float(bordes[i] + ancho / 2)
        filas.append(fila)
    return pd.DataFrame(filas)


@torch.no_grad()
def predecir(
    modelo: nn.Module, loader: DataLoader, stats: dict, dispositivo: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (y_real, y_pred) en dex, deshaciendo la estandarizacion."""
    modelo.eval().to(dispositivo)
    preds, reales = [], []
    for x, y in loader:
        salida = modelo(x.to(dispositivo))
        preds.append(salida.float().cpu().numpy())
        reales.append(y.numpy())

    mu, sigma = stats["media_target"], stats["std_target"]
    y_pred = np.concatenate(preds) * sigma + mu
    y_real = np.concatenate(reales) * sigma + mu
    return y_real, y_pred


def baseline_media(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Predecir siempre la media del train.

    Es el piso contra el que hay que medirse: cualquier modelo que no lo supere
    no esta aprendiendo nada de las imagenes. En el paper este baseline da
    ~0.20 dex de RMSE.
    """
    return metricas(y_test, np.full_like(y_test, y_train.mean()))


def evaluar(
    cfg: Config,
    modelo: nn.Module,
    loaders: dict[str, DataLoader],
    meta: pd.DataFrame,
    stats: dict,
    dir_salida: Path | None = None,
) -> dict:
    dir_salida = dir_salida or ruta_absoluta(cfg.entrenamiento.dir_resultados)
    dir_salida.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_real, y_pred = predecir(modelo, loaders["test"], stats, dispositivo)
    y_train = meta.loc[meta["split"] == "train", "metalicidad"].to_numpy()

    resultado = {
        "test": metricas(y_real, y_pred),
        "baseline_media": baseline_media(y_train, y_real),
        "referencia_wu_boada_2019": {"rmse_dex": 0.085, "nmad_dex": 0.067},
    }

    por_bin = metricas_por_bin(y_real, y_pred)
    por_bin.to_csv(dir_salida / "metricas_por_bin.csv", index=False)

    np.savez_compressed(
        dir_salida / "predicciones_test.npz", y_real=y_real, y_pred=y_pred
    )
    (dir_salida / "metricas.json").write_text(
        json.dumps(resultado, indent=2), encoding="utf-8"
    )

    t, b = resultado["test"], resultado["baseline_media"]
    log.info("--- Test (n=%d) ---", t["n"])
    log.info("  RMSE  %.4f dex   (baseline media: %.4f | Wu & Boada: 0.085)",
             t["rmse_dex"], b["rmse_dex"])
    log.info("  NMAD  %.4f dex   (Wu & Boada: 0.067)", t["nmad_dex"])
    log.info("  MAE   %.4f dex", t["mae_dex"])
    log.info("  sesgo %+.4f dex", t["sesgo_dex"])
    log.info("  R2    %.4f", t["r2"])

    if t["rmse_dex"] >= b["rmse_dex"]:
        log.warning(
            "El modelo NO supera al baseline de predecir la media. "
            "No esta extrayendo informacion de las imagenes."
        )
    return resultado
