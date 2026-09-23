"""Figuras del proyecto.

Paleta categorica validada para separacion CVD en todos los pares
(azul / naranjo / aqua). El aqua queda bajo 3:1 de contraste sobre fondo claro,
asi que toda serie que lo use lleva etiqueta directa; ademas las metricas se
publican como tabla en metricas_por_bin.csv.

Las densidades usan una rampa neutra de un solo tono (claro -> oscuro), que es
la convencion en la literatura y lo que usa Wu & Boada (2019).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # backend sin display, necesario en nodos de cluster

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from .config import Config, ruta_absoluta  # noqa: E402

log = logging.getLogger(__name__)

AZUL, NARANJO, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_SUAVE = "#0b0b0b", "#52514e"
DENSIDAD = "Greys"

# Referencia del paper, para marcarla en las figuras.
WU_BOADA_RMSE = 0.085


def _estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelcolor": TINTA,
        "axes.edgecolor": "#c9c9c6",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e4e4e1",       # grilla recesiva
        "grid.linewidth": 0.6,
        "text.color": TINTA,
        "xtick.color": TINTA_SUAVE,
        "ytick.color": TINTA_SUAVE,
        "legend.frameon": False,
        "lines.linewidth": 2.0,        # marcas finas
    })


def _mediana_movil(x, y, bordes):
    """Mediana y percentiles 16/84 de y en bins de x."""
    idx = np.digitize(x, bordes) - 1
    centros, p16, p50, p84 = [], [], [], []
    for i in range(len(bordes) - 1):
        m = idx == i
        if m.sum() < 20:
            continue
        centros.append((bordes[i] + bordes[i + 1]) / 2)
        p16.append(np.percentile(y[m], 16))
        p50.append(np.percentile(y[m], 50))
        p84.append(np.percentile(y[m], 84))
    return [np.asarray(v) for v in (centros, p16, p50, p84)]


def relacion_masa_metalicidad(meta: pd.DataFrame, destino: Path) -> None:
    """MZR de la muestra.

    Control de sanidad del catalogo: si no reproduce Tremonti+04, hay algo mal
    en la seleccion y no tiene sentido seguir al entrenamiento.
    """
    _estilo()
    df = meta[np.isfinite(meta["log_masa"]) & (meta["log_masa"] > 7)]
    fig, ax = plt.subplots(figsize=(7, 5))

    hb = ax.hexbin(df["log_masa"], df["metalicidad"], gridsize=70,
                   cmap=DENSIDAD, norm=LogNorm(), mincnt=1, linewidths=0)
    fig.colorbar(hb, ax=ax, label="galaxias por celda")

    centros, p16, p50, p84 = _mediana_movil(
        df["log_masa"].to_numpy(), df["metalicidad"].to_numpy(),
        np.arange(8, 11.6, 0.15),
    )
    if len(centros) >= 2:
        ax.fill_between(centros, p16, p84, color=NARANJO, alpha=0.18, linewidth=0,
                        label="rango 16-84%")
        ax.plot(centros, p50, color=NARANJO, label="mediana")
        # Leyenda abajo a la derecha: la MZR sube hacia la derecha, asi que esa
        # esquina siempre queda libre.
        ax.legend(loc="lower right")
    else:
        log.warning("Muy pocos bins de masa poblados para la mediana movil")

    # Limites ajustados a los datos, con un margen. Dejar los ejes automaticos
    # deja franjas vacias grandes porque la MZR ocupa una banda diagonal.
    ax.set_xlim(np.percentile(df["log_masa"], 0.2) - 0.2,
                np.percentile(df["log_masa"], 99.8) + 0.2)
    ax.set_ylim(np.percentile(df["metalicidad"], 0.2) - 0.1,
                np.percentile(df["metalicidad"], 99.8) + 0.1)

    ax.set_xlabel(r"log $M_\star$ [$M_\odot$]")
    ax.set_ylabel(r"12 + log(O/H)")
    ax.set_title(f"Relacion masa-metalicidad de la muestra (N = {len(df):,})")
    fig.savefig(destino)
    plt.close(fig)


def curvas_entrenamiento(historial: dict, destino: Path) -> None:
    _estilo()
    epocas = np.arange(1, len(historial["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epocas, historial["train_loss"], color=AZUL, label="entrenamiento")
    ax1.plot(epocas, historial["val_loss"], color=NARANJO, label="validacion")
    ax1.set_xlabel("epoca")
    ax1.set_ylabel("MSE (target estandarizado)")
    ax1.set_title("Perdida")
    ax1.legend()

    ax2.plot(epocas, historial["val_rmse_dex"], color=AZUL, label="validacion")
    ax2.axhline(WU_BOADA_RMSE, color=TINTA_SUAVE, ls="--", lw=1.2)
    ax2.annotate("Wu & Boada 2019 (0.085)", (epocas[-1], WU_BOADA_RMSE),
                 xytext=(-6, 5), textcoords="offset points",
                 ha="right", color=TINTA_SUAVE, fontsize=9)

    if historial["epoca_mejor"] > 0:
        ax2.axvline(historial["epoca_mejor"], color=NARANJO, ls=":", lw=1.2)
        ax2.annotate(
            "mejor: epoca {}".format(historial["epoca_mejor"]),
            (historial["epoca_mejor"], max(historial["val_rmse_dex"])),
            xytext=(6, -6), textcoords="offset points",
            color=NARANJO, fontsize=9,
        )

    ax2.set_xlabel("epoca")
    ax2.set_ylabel("RMSE [dex]")
    ax2.set_title("Error de validacion")

    fig.savefig(destino)
    plt.close(fig)


def prediccion_vs_real(y_real, y_pred, metricas: dict, destino: Path) -> None:
    """Figura principal, con la misma estructura que la Figura 3 del paper."""
    _estilo()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    hb = ax.hexbin(y_real, y_pred, gridsize=60, cmap=DENSIDAD,
                   norm=LogNorm(), mincnt=1, linewidths=0)
    # shrink: el eje es cuadrado (aspecto 1:1), asi que la barra no debe
    # estirarse a lo alto de toda la figura.
    fig.colorbar(hb, ax=ax, label="galaxias por celda", shrink=0.82)

    lim = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
    ax.plot(lim, lim, color=TINTA, lw=1.2, label="identidad")

    bordes = np.arange(np.floor(y_real.min() * 10) / 10, y_real.max() + 0.1, 0.1)
    centros, _, p50, _ = _mediana_movil(y_real, y_pred, bordes)
    rmse, nm = metricas["rmse_dex"], metricas["nmad_dex"]

    # La mediana movil necesita al menos dos bins poblados para ser una linea.
    # Con menos, se omiten mediana, bandas y sus etiquetas: dibujar anotaciones
    # sueltas sin la serie que describen deja la figura mintiendo.
    if len(centros) >= 2:
        ax.plot(centros, p50, color=NARANJO, label="mediana")
        ax.plot(centros, p50 + rmse, color=AZUL, ls="--", lw=1.4)
        ax.plot(centros, p50 - rmse, color=AZUL, ls="--", lw=1.4)
        ax.plot(centros, p50 + nm, color=AQUA, ls=":", lw=1.6)
        ax.plot(centros, p50 - nm, color=AQUA, ls=":", lw=1.6)

        # Etiquetas directas: el aqua no alcanza 3:1 de contraste sobre el
        # fondo, asi que su identidad no puede depender solo del color.
        # Se anclan antes del extremo y se empujan hacia afuera de la banda,
        # para que no queden encima de las curvas.
        j = int(len(centros) * 0.7)
        ax.annotate(f"±1 RMSE ({rmse:.3f})", (centros[j], p50[j] + rmse),
                    xytext=(-4, 10), textcoords="offset points",
                    ha="right", va="bottom", color=AZUL, fontsize=9)
        ax.annotate(f"±1 NMAD ({nm:.3f})", (centros[j], p50[j] - nm),
                    xytext=(-4, -10), textcoords="offset points",
                    ha="right", va="top", color=AQUA, fontsize=9)
    else:
        log.warning(
            "Muy pocos bins poblados (%d) para la mediana movil; "
            "se omiten mediana y bandas en %s", len(centros), destino.name,
        )
        ax.annotate(
            f"RMSE {rmse:.3f} · NMAD {nm:.3f} dex",
            xy=(0.03, 0.88), xycoords="axes fraction", color=TINTA_SUAVE, fontsize=9,
        )

    ax.set_xlabel(r"12 + log(O/H) espectroscopica")
    ax.set_ylabel(r"12 + log(O/H) predicha por la CNN")
    ax.set_title("Prediccion vs. valor real (test, N = {:,})".format(metricas["n"]))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    fig.savefig(destino)
    plt.close(fig)


def residuos(y_real, y_pred, metricas: dict, destino: Path) -> None:
    _estilo()
    res = y_pred - y_real
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(res, bins=80, color=AZUL, edgecolor="none")
    ax.axvline(0, color=TINTA, lw=1.2)
    ax.axvline(metricas["sesgo_dex"], color=NARANJO, ls="--", lw=1.4)
    ax.annotate(f"sesgo {metricas['sesgo_dex']:+.3f}",
                xy=(metricas["sesgo_dex"], 0.92), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points",
                color=NARANJO, fontsize=9)

    ax.set_xlabel(r"$\Delta Z$ = predicha - real [dex]")
    ax.set_ylabel("galaxias")
    ax.set_title("Residuos (RMSE {:.3f}, NMAD {:.3f} dex)".format(
        metricas["rmse_dex"], metricas["nmad_dex"]))
    fig.savefig(destino)
    plt.close(fig)


def error_por_bin(por_bin: pd.DataFrame, destino: Path) -> None:
    """Como se degrada el error en los extremos de metalicidad.

    Con la distribucion natural el grueso de la muestra esta cerca de 8.8; esta
    figura muestra explicitamente donde el modelo es peor, en vez de esconderlo
    reequilibrando el dataset.
    """
    _estilo()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(por_bin["bin_centro"], por_bin["rmse_dex"], color=AZUL,
            marker="o", markersize=5, label="RMSE")
    ax.plot(por_bin["bin_centro"], por_bin["nmad_dex"], color=AQUA,
            marker="s", markersize=5, ls=":", label="NMAD")

    ax.axhline(WU_BOADA_RMSE, color=TINTA_SUAVE, ls="--", lw=1.2)
    ax.annotate("RMSE global Wu & Boada 2019",
                (por_bin["bin_centro"].iloc[-1], WU_BOADA_RMSE),
                xytext=(-6, 5), textcoords="offset points",
                ha="right", color=TINTA_SUAVE, fontsize=9)

    ax.set_xlabel(r"12 + log(O/H) real")
    ax.set_ylabel("error [dex]")
    ax.set_title("Error por bin de metalicidad")
    ax.legend()
    fig.savefig(destino)
    plt.close(fig)


def galaxias_ejemplo(
    imagenes: np.ndarray,
    meta: pd.DataFrame,
    y_real,
    y_pred,
    destino: Path,
    n: int = 10,
) -> None:
    """Cutouts del test con su metalicidad real y la predicha."""
    _estilo()
    test = meta[meta["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test), size=min(n, len(test)), replace=False)

    filas = 2
    cols = int(np.ceil(len(idx) / filas))
    fig, axs = plt.subplots(filas, cols, figsize=(2.0 * cols, 2.5 * filas))
    ejes = np.ravel(axs)

    # ejes puede ser mas largo que idx; los sobrantes se apagan abajo.
    for ax, i in zip(ejes, idx, strict=False):
        ax.imshow(imagenes[test.loc[i, "fila"]])
        ax.set_title(f"real {y_real[i]:.2f}\npred {y_pred[i]:.2f}", fontsize=9)
        ax.axis("off")
    for ax in ejes[len(idx):]:
        ax.axis("off")

    fig.suptitle("Cutouts gri de SDSS del conjunto de test", fontsize=12)
    fig.savefig(destino)
    plt.close(fig)


def generar_todas(cfg: Config, meta: pd.DataFrame, dir_resultados: Path) -> None:
    """Genera todas las figuras a partir de los artefactos ya escritos en disco."""
    dir_fig = ruta_absoluta("figures")
    dir_fig.mkdir(parents=True, exist_ok=True)

    relacion_masa_metalicidad(meta, dir_fig / "mzr_muestra.png")

    hist_path = dir_resultados / "historial.json"
    if hist_path.exists():
        curvas_entrenamiento(
            json.loads(hist_path.read_text(encoding="utf-8")),
            dir_fig / "curvas_entrenamiento.png",
        )

    pred_path = dir_resultados / "predicciones_test.npz"
    if pred_path.exists():
        datos = np.load(pred_path)
        y_real, y_pred = datos["y_real"], datos["y_pred"]
        met = json.loads(
            (dir_resultados / "metricas.json").read_text(encoding="utf-8")
        )["test"]

        prediccion_vs_real(y_real, y_pred, met, dir_fig / "prediccion_vs_real.png")
        residuos(y_real, y_pred, met, dir_fig / "residuos.png")

        por_bin = pd.read_csv(dir_resultados / "metricas_por_bin.csv")
        error_por_bin(por_bin, dir_fig / "error_por_bin.png")

        imagenes = np.load(
            ruta_absoluta("data/processed/imagenes.npy"), mmap_mode="r"
        )
        galaxias_ejemplo(imagenes, meta, y_real, y_pred,
                         dir_fig / "galaxias_ejemplo.png")

    log.info("Figuras generadas en %s", dir_fig)
