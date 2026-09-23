"""Splits, normalizacion y Dataset de PyTorch."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config, ruta_absoluta

log = logging.getLogger(__name__)


def hacer_splits(cfg: Config, meta: pd.DataFrame) -> pd.DataFrame:
    """Asigna cada galaxia a train / val / test de forma reproducible.

    El split es aleatorio y no estratificado: queremos que los tres conjuntos
    compartan la distribucion natural de metalicidad, que es justamente lo que
    el modelo debe enfrentar. El test se aparta aca y no se toca hasta el final.
    """
    pre = cfg.preprocesamiento
    rng = np.random.default_rng(pre.semilla_split)
    n = len(meta)
    orden = rng.permutation(n)

    n_train = int(pre.fraccion_train * n)
    n_val = int(pre.fraccion_val * n)

    split = np.empty(n, dtype=object)
    split[orden[:n_train]] = "train"
    split[orden[n_train:n_train + n_val]] = "val"
    split[orden[n_train + n_val:]] = "test"

    meta = meta.copy()
    meta["split"] = split
    log.info("Splits: %s", meta["split"].value_counts().to_dict())
    return meta


def calcular_estadisticas(cfg: Config, imagenes: np.ndarray, meta: pd.DataFrame) -> dict:
    """Media y desviacion por canal, y del target, calculadas SOLO en train.

    Usar el split completo filtraria informacion de val/test al entrenamiento.
    Las estadisticas quedan fijas y compartidas: nunca se normaliza imagen por
    imagen, porque eso borraria el color absoluto, que es la senal que predice
    la metalicidad.
    """
    filas_train = meta.loc[meta["split"] == "train", "fila"].to_numpy()

    # Muestreo para no recorrer 70k imagenes completas solo para las medias.
    rng = np.random.default_rng(cfg.preprocesamiento.semilla_split)
    if len(filas_train) > 10000:
        filas_muestra = rng.choice(filas_train, size=10000, replace=False)
    else:
        filas_muestra = filas_train
    muestra = imagenes[np.sort(filas_muestra)].astype(np.float32) / 255.0

    stats = {
        "media_canal": muestra.mean(axis=(0, 1, 2)).tolist(),
        "std_canal": muestra.std(axis=(0, 1, 2)).tolist(),
        "media_target": float(meta.loc[meta["split"] == "train", "metalicidad"].mean()),
        "std_target": float(meta.loc[meta["split"] == "train", "metalicidad"].std()),
        "n_train": int(len(filas_train)),
    }

    ruta = ruta_absoluta(cfg.preprocesamiento.estadisticas)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Estadisticas de normalizacion -> %s", ruta)
    log.info("  media canal: %s", np.round(stats["media_canal"], 4).tolist())
    log.info("  std canal:   %s", np.round(stats["std_canal"], 4).tolist())
    log.info("  target: mu=%.4f sigma=%.4f", stats["media_target"], stats["std_target"])
    return stats


def cargar_estadisticas(cfg: Config) -> dict:
    ruta = ruta_absoluta(cfg.preprocesamiento.estadisticas)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Corre primero scripts/02_descargar_imagenes.py"
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


class GalaxiasDataset(Dataset):
    """Imagenes gri de SDSS con la metalicidad como target.

    El target se entrega estandarizado (media 0, sigma 1 segun el split de
    train). Sin eso la red parte prediciendo ~0 contra objetivos de ~8.8 y
    gasta las primeras epocas solo aprendiendo el offset.
    """

    def __init__(
        self,
        imagenes: np.ndarray,
        meta: pd.DataFrame,
        stats: dict,
        aumentar: bool = False,
    ):
        self.imagenes = imagenes
        self.filas = meta["fila"].to_numpy()
        self.targets = meta["metalicidad"].to_numpy(dtype=np.float32)
        self.aumentar = aumentar

        self.media = np.asarray(stats["media_canal"], dtype=np.float32)
        self.std = np.asarray(stats["std_canal"], dtype=np.float32)
        self.mu_y = float(stats["media_target"])
        self.sigma_y = float(stats["std_target"])

    def __len__(self) -> int:
        return len(self.filas)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.imagenes[self.filas[idx]].astype(np.float32) / 255.0

        if self.aumentar:
            # Solo transformaciones del grupo diedrico: la orientacion de una
            # galaxia en el cielo no aporta informacion fisica sobre su
            # metalicidad, pero el zoom o el recorte si alterarian el tamano
            # aparente, que si correlaciona con masa y por tanto con Z.
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k, axes=(0, 1))
            if np.random.rand() < 0.5:
                img = np.fliplr(img)
            img = np.ascontiguousarray(img)

        img = (img - self.media) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        y = (self.targets[idx] - self.mu_y) / self.sigma_y
        return img, torch.tensor(y, dtype=torch.float32)


def construir_loaders(
    cfg: Config, imagenes: np.ndarray, meta: pd.DataFrame, stats: dict
) -> dict[str, DataLoader]:
    loaders = {}
    for nombre in ("train", "val", "test"):
        subset = meta[meta["split"] == nombre]
        ds = GalaxiasDataset(
            imagenes, subset, stats,
            aumentar=(nombre == "train" and cfg.entrenamiento.augmentation),
        )
        loaders[nombre] = DataLoader(
            ds,
            batch_size=cfg.entrenamiento.batch_size,
            shuffle=(nombre == "train"),
            num_workers=cfg.entrenamiento.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(nombre == "train"),
            persistent_workers=cfg.entrenamiento.num_workers > 0,
        )
    return loaders


def cargar_datos(cfg: Config) -> tuple[np.ndarray, pd.DataFrame]:
    """Abre el memmap de imagenes y sus metadatos con los splits ya asignados."""
    dir_proc = ruta_absoluta("data/processed")
    imagenes = np.load(dir_proc / "imagenes.npy", mmap_mode="r")
    meta = pd.read_parquet(dir_proc / "metadatos.parquet")
    if "split" not in meta.columns:
        meta = hacer_splits(cfg, meta)
    return imagenes, meta
