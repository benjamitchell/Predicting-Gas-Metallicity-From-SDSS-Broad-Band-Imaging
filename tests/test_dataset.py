"""Tests de splits, normalizacion y Dataset.

Cubren dos errores concretos del intento original: normalizar cada imagen por
separado con min-max (lo que borra el color absoluto, que es la señal) y no
apartar un conjunto de test.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from gasmet.config import cargar
from gasmet.dataset import GalaxiasDataset, calcular_estadisticas, hacer_splits


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Config real del repo, pero escribiendo en un directorio temporal.

    El orden importa: primero se carga el YAML desde su ubicacion real y recien
    despues se redirige RAIZ, para que las rutas de salida apunten a tmp_path
    sin que la carga misma salga a buscar el config ahi.
    """
    import gasmet.config as gconfig

    c = cargar(gconfig.RAIZ / "config.yaml")
    monkeypatch.setattr(gconfig, "RAIZ", tmp_path)
    monkeypatch.delenv(gconfig.VAR_DIR_DATOS, raising=False)
    return c


@pytest.fixture
def meta() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "fila": np.arange(n),
        "metalicidad": rng.normal(8.9, 0.19, n).astype(np.float32),
        "log_masa": rng.normal(10.2, 0.5, n),
    })


@pytest.fixture
def imagenes() -> np.ndarray:
    rng = np.random.default_rng(1)
    # Canales con niveles distintos a proposito: si la normalizacion fuera
    # por imagen, esa diferencia se perderia.
    base = rng.integers(0, 120, size=(400, 16, 16, 3), dtype=np.uint8)
    base[..., 0] = np.clip(base[..., 0] + 60, 0, 255)
    return base


def test_splits_cubren_todo_sin_solaparse(cfg, meta):
    con_split = hacer_splits(cfg, meta)

    conteo = con_split["split"].value_counts()
    assert set(conteo.index) == {"train", "val", "test"}
    assert conteo.sum() == len(meta)

    grupos = {n: set(g["fila"]) for n, g in con_split.groupby("split")}
    assert not (grupos["train"] & grupos["val"])
    assert not (grupos["train"] & grupos["test"])
    assert not (grupos["val"] & grupos["test"])


def test_splits_respetan_las_proporciones(cfg, meta):
    con_split = hacer_splits(cfg, meta)
    frac = con_split["split"].value_counts(normalize=True)

    assert frac["train"] == pytest.approx(cfg.preprocesamiento.fraccion_train, abs=0.02)
    assert frac["val"] == pytest.approx(cfg.preprocesamiento.fraccion_val, abs=0.02)
    assert frac["test"] == pytest.approx(cfg.preprocesamiento.fraccion_test, abs=0.02)


def test_splits_son_reproducibles(cfg, meta):
    a = hacer_splits(cfg, meta)["split"].tolist()
    b = hacer_splits(cfg, meta)["split"].tolist()
    assert a == b


def test_estadisticas_se_calculan_solo_con_train(cfg, meta, imagenes):
    """Usar val o test para las medias filtraria informacion al entrenamiento."""
    con_split = hacer_splits(cfg, meta)
    stats = calcular_estadisticas(cfg, imagenes, con_split)

    solo_train = con_split[con_split["split"] == "train"]["metalicidad"]
    assert stats["n_train"] == len(solo_train)
    assert stats["media_target"] == pytest.approx(solo_train.mean(), rel=1e-5)
    assert stats["std_target"] == pytest.approx(solo_train.std(), rel=1e-5)

    # Si se hubiera usado todo el dataset, la media seria distinta.
    assert stats["media_target"] != pytest.approx(meta["metalicidad"].mean(), rel=1e-9)


def test_normalizacion_es_por_canal_y_compartida(cfg, meta, imagenes):
    """Regresion: nunca min-max por imagen.

    Normalizar cada galaxia por separado la reescala a un rango comun y borra
    el brillo y el color absolutos, que son justamente lo que predice la
    metalicidad. Wu & Boada lo cuantifican: pasar de solo r a gri baja el RMSE
    de 0.138 a 0.085 dex.
    """
    con_split = hacer_splits(cfg, meta)
    stats = calcular_estadisticas(cfg, imagenes, con_split)

    # El canal rojo se construyo mas brillante; eso debe verse en las medias.
    assert stats["media_canal"][0] > stats["media_canal"][1]
    assert len(set(stats["media_canal"])) == 3

    ds = GalaxiasDataset(imagenes, con_split, stats, aumentar=False)
    x0, _ = ds[0]
    x1, _ = ds[1]

    # Con min-max por imagen, cada tensor quedaria en [0,1] con el mismo minimo
    # y maximo. Con estadisticas compartidas, cada galaxia conserva su nivel.
    assert not (x0.min() == x1.min() and x0.max() == x1.max())


def test_target_se_entrega_estandarizado(cfg, meta, imagenes):
    """Sin esto la red parte prediciendo ~0 contra objetivos de ~8.9."""
    con_split = hacer_splits(cfg, meta)
    stats = calcular_estadisticas(cfg, imagenes, con_split)
    train = con_split[con_split["split"] == "train"]

    ds = GalaxiasDataset(imagenes, train, stats, aumentar=False)
    objetivos = torch.stack([ds[i][1] for i in range(len(ds))])

    assert float(objetivos.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(objetivos.std()) == pytest.approx(1.0, abs=0.05)


def test_dataset_entrega_la_forma_que_espera_torch(cfg, meta, imagenes):
    con_split = hacer_splits(cfg, meta)
    stats = calcular_estadisticas(cfg, imagenes, con_split)
    ds = GalaxiasDataset(imagenes, con_split, stats, aumentar=False)

    x, y = ds[0]
    assert x.shape == (3, 16, 16), "debe ser CHW, no HWC"
    assert x.dtype == torch.float32
    assert y.shape == ()


def test_augmentation_conserva_forma_y_target(cfg, meta, imagenes):
    """Solo rotaciones y flips: la orientacion no aporta informacion fisica.

    Un zoom o un recorte si alterarian el tamano aparente, que correlaciona con
    la masa y por tanto con la metalicidad.
    """
    con_split = hacer_splits(cfg, meta)
    stats = calcular_estadisticas(cfg, imagenes, con_split)
    ds = GalaxiasDataset(imagenes, con_split, stats, aumentar=True)

    formas = set()
    objetivos = set()
    for _ in range(20):
        x, y = ds[0]
        formas.add(tuple(x.shape))
        objetivos.add(round(float(y), 6))

    assert formas == {(3, 16, 16)}
    assert len(objetivos) == 1, "la augmentation no debe tocar el target"
