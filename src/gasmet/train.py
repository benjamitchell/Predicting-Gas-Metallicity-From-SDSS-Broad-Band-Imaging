"""Bucle de entrenamiento.

La perdida se calcula sobre el target estandarizado, pero todo lo que se
reporta va en dex, multiplicando por sigma_y. Un RMSE en unidades arbitrarias
no se puede comparar con la literatura, y el punto del proyecto es compararse
con los 0.085 dex de Wu & Boada (2019).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, ruta_absoluta

log = logging.getLogger(__name__)


@dataclass
class Historial:
    """Registro por epoca, se vuelca a JSON para poder graficar despues."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_rmse_dex: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoca_mejor: int = -1
    mejor_val_rmse_dex: float = float("inf")
    segundos: float = 0.0


def _dispositivo() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info("Dispositivo: %s", torch.cuda.get_device_name(0))
    else:
        dev = torch.device("cpu")
        log.warning("Sin GPU disponible; el entrenamiento sera lento")
    return dev


def _pasada(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
    optimizador: torch.optim.Optimizer | None = None,
    escalador: torch.cuda.amp.GradScaler | None = None,
    desc: str = "",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Una pasada completa. Si se pasa optimizador, entrena; si no, evalua."""
    entrenando = optimizador is not None
    modelo.train(entrenando)

    perdida_total, n = 0.0, 0
    preds, reales = [], []

    with torch.set_grad_enabled(entrenando):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(dispositivo, non_blocking=True)
            y = y.to(dispositivo, non_blocking=True)

            if entrenando:
                optimizador.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=escalador is not None):
                salida = modelo(x)
                perdida = criterio(salida, y)

            if entrenando:
                if escalador is not None:
                    escalador.scale(perdida).backward()
                    escalador.step(optimizador)
                    escalador.update()
                else:
                    perdida.backward()
                    optimizador.step()

            perdida_total += perdida.item() * x.size(0)
            n += x.size(0)
            preds.append(salida.detach().float().cpu().numpy())
            reales.append(y.detach().float().cpu().numpy())

    return perdida_total / n, np.concatenate(preds), np.concatenate(reales)


def entrenar(
    cfg: Config,
    modelo: nn.Module,
    loaders: dict[str, DataLoader],
    stats: dict,
    dir_salida: Path | None = None,
) -> tuple[nn.Module, Historial]:
    ent = cfg.entrenamiento
    dispositivo = _dispositivo()
    modelo = modelo.to(dispositivo)

    dir_salida = dir_salida or ruta_absoluta(ent.dir_resultados)
    (dir_salida / "checkpoints").mkdir(parents=True, exist_ok=True)
    ruta_mejor = dir_salida / "checkpoints" / "mejor.pt"

    criterio = nn.MSELoss()
    optimizador = torch.optim.AdamW(
        modelo.parameters(), lr=ent.learning_rate, weight_decay=ent.weight_decay
    )
    planificador = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=ent.epocas)
        if ent.scheduler == "cosine"
        else None
    )
    escalador = torch.cuda.amp.GradScaler() if dispositivo.type == "cuda" else None

    sigma_y = float(stats["std_target"])
    hist = Historial()
    sin_mejora = 0
    t0 = time.time()

    for epoca in range(1, ent.epocas + 1):
        train_loss, _, _ = _pasada(
            modelo, loaders["train"], criterio, dispositivo,
            optimizador, escalador, desc=f"epoca {epoca} train",
        )
        val_loss, preds, reales = _pasada(
            modelo, loaders["val"], criterio, dispositivo, desc=f"epoca {epoca} val"
        )

        # De espacio estandarizado a dex.
        val_rmse_dex = float(np.sqrt(np.mean((preds - reales) ** 2)) * sigma_y)

        hist.train_loss.append(train_loss)
        hist.val_loss.append(val_loss)
        hist.val_rmse_dex.append(val_rmse_dex)
        hist.lr.append(optimizador.param_groups[0]["lr"])

        if planificador:
            planificador.step()

        marca = ""
        if val_rmse_dex < hist.mejor_val_rmse_dex:
            hist.mejor_val_rmse_dex = val_rmse_dex
            hist.epoca_mejor = epoca
            sin_mejora = 0
            torch.save(
                {"modelo": modelo.state_dict(), "epoca": epoca,
                 "val_rmse_dex": val_rmse_dex, "config_modelo": dict(cfg.modelo)},
                ruta_mejor,
            )
            marca = "  <- mejor"
        else:
            sin_mejora += 1

        log.info(
            "Epoca %3d/%d | train %.4f | val %.4f | val RMSE %.4f dex%s",
            epoca, ent.epocas, train_loss, val_loss, val_rmse_dex, marca,
        )

        if sin_mejora >= ent.paciencia:
            log.info("Early stopping: %d epocas sin mejorar", ent.paciencia)
            break

    hist.segundos = time.time() - t0
    log.info(
        "Entrenamiento terminado en %.1f min. Mejor: %.4f dex (epoca %d)",
        hist.segundos / 60, hist.mejor_val_rmse_dex, hist.epoca_mejor,
    )

    (dir_salida / "historial.json").write_text(
        json.dumps(asdict(hist), indent=2), encoding="utf-8"
    )

    # Se devuelven los pesos del mejor checkpoint, no los de la ultima epoca.
    modelo.load_state_dict(torch.load(ruta_mejor, map_location=dispositivo)["modelo"])
    return modelo, hist
