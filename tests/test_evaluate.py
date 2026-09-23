"""Tests de las metricas.

Se reportan RMSE y NMAD en dex, las mismas de Wu & Boada (2019), para que los
numeros sean comparables con la literatura.
"""

import numpy as np
import pytest

from gasmet.evaluate import baseline_media, metricas, metricas_por_bin, nmad


def test_nmad_aproxima_sigma_en_datos_gaussianos():
    x = np.random.default_rng(0).normal(0, 0.25, 200_000)
    assert nmad(x) == pytest.approx(0.25, rel=0.02)


def test_nmad_es_robusto_a_outliers():
    """Es su razon de ser: el RMSE se dispara con outliers, el NMAD no."""
    limpio = np.random.default_rng(0).normal(0, 0.1, 10_000)
    sucio = limpio.copy()
    sucio[:50] = 100.0  # 0.5% de valores absurdos

    assert nmad(sucio) == pytest.approx(nmad(limpio), rel=0.05)
    assert np.std(sucio) > 10 * np.std(limpio)


def test_prediccion_perfecta_da_error_cero():
    y = np.array([8.5, 8.8, 9.1, 9.0])
    m = metricas(y, y.copy())

    assert m["rmse_dex"] == pytest.approx(0.0)
    assert m["mae_dex"] == pytest.approx(0.0)
    assert m["sesgo_dex"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_rmse_y_sesgo_se_calculan_como_corresponde():
    y_real = np.array([8.0, 9.0])
    y_pred = np.array([8.2, 9.2])  # +0.2 constante

    m = metricas(y_real, y_pred)
    assert m["rmse_dex"] == pytest.approx(0.2)
    assert m["sesgo_dex"] == pytest.approx(0.2)
    assert m["mae_dex"] == pytest.approx(0.2)


def test_sesgo_distingue_el_signo():
    y = np.array([8.5, 9.0])
    assert metricas(y, y - 0.1)["sesgo_dex"] == pytest.approx(-0.1)
    assert metricas(y, y + 0.1)["sesgo_dex"] == pytest.approx(+0.1)


def test_baseline_de_la_media_es_el_piso_a_superar():
    """Su RMSE es aproximadamente sigma del target.

    Un modelo que no lo supere no esta extrayendo nada de las imagenes, por
    bueno que se vea su R2. En Wu & Boada este baseline da ~0.20 dex.
    """
    rng = np.random.default_rng(0)
    y_train = rng.normal(8.93, 0.188, 20_000)
    y_test = rng.normal(8.93, 0.188, 5_000)

    b = baseline_media(y_train, y_test)
    assert b["rmse_dex"] == pytest.approx(0.188, rel=0.05)
    assert b["r2"] == pytest.approx(0.0, abs=0.02)


def test_metricas_por_bin_ignora_bins_con_pocos_objetos():
    """Un bin con tres galaxias no dice nada sobre el desempeno."""
    y_real = np.concatenate([np.full(200, 8.85), np.full(3, 7.75)])
    y_pred = y_real + 0.05

    por_bin = metricas_por_bin(y_real, y_pred)

    assert len(por_bin) >= 1
    assert (por_bin["n"] >= 20).all()
    assert not np.isclose(por_bin["bin_centro"], 7.75, atol=0.05).any()


def test_metricas_por_bin_expone_el_error_en_los_extremos():
    """Con la distribucion natural el grueso esta cerca de 8.9.

    La metrica global esconde que el modelo es peor en los extremos; esta
    funcion lo hace explicito en vez de taparlo reequilibrando el dataset.
    """
    rng = np.random.default_rng(0)
    centro = np.full(500, 8.95)
    extremo = np.full(200, 8.35)
    y_real = np.concatenate([centro, extremo])
    y_pred = np.concatenate([
        centro + rng.normal(0, 0.05, 500),
        extremo + rng.normal(0, 0.30, 200),   # mucho peor en el extremo
    ])

    por_bin = metricas_por_bin(y_real, y_pred)
    peor = por_bin.loc[por_bin["rmse_dex"].idxmax(), "bin_centro"]

    assert peor < 8.6, "el bin de mayor error deberia ser el del extremo"
    assert por_bin["rmse_dex"].max() > 3 * por_bin["rmse_dex"].min()
