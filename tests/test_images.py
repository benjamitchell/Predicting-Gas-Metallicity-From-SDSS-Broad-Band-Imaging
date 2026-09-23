"""Tests del armado de URLs de cutouts.

El test central de este archivo es de regresion: el intento original del
proyecto se invalido porque se pedian las imagenes con `&band=u|g|r|i|z`, un
parametro que ImgCutout no acepta. Lo ignora en silencio y devuelve siempre la
misma imagen, con codigo 200 y sin warning, asi que el error no producia ningun
sintoma: la CNN entrenaba con cinco copias del mismo canal.

Si alguien vuelve a agregar ese parametro, estos tests fallan.
"""

from urllib.parse import parse_qs, urlparse

import pytest

from gasmet.config import Config, cargar
from gasmet.images import PARAMS_VALIDOS, url_cutout


@pytest.fixture
def cfg() -> Config:
    return cargar()


def test_url_no_lleva_parametro_band(cfg):
    """Regresion: `band` no debe aparecer nunca en la URL.

    ImgCutout no lo soporta. Pedirlo da la falsa impresion de estar bajando
    bandas separadas cuando en realidad son archivos identicos.
    """
    url = url_cutout(cfg, 184.9511, 12.8386)
    assert "band" not in url.lower()
    assert "band" not in parse_qs(urlparse(url).query)


def test_url_solo_usa_parametros_soportados(cfg):
    """Cualquier parametro fuera de la lista se ignora en silencio."""
    query = parse_qs(urlparse(url_cutout(cfg, 10.0, -5.0)).query)
    assert set(query) <= PARAMS_VALIDOS, (
        f"parametros no soportados por ImgCutout: {set(query) - PARAMS_VALIDOS}"
    )


def test_url_respeta_la_configuracion(cfg):
    query = parse_qs(urlparse(url_cutout(cfg, 10.0, -5.0)).query)
    assert int(query["width"][0]) == cfg.imagenes.tamano_px
    assert int(query["height"][0]) == cfg.imagenes.tamano_px
    assert float(query["scale"][0]) == cfg.imagenes.escala_arcsec_px


@pytest.mark.parametrize(
    ("ra", "dec"),
    [(0.0, 0.0), (359.999999, -89.9), (184.9511, 12.8386), (1e-7, -1e-7)],
)
def test_coordenadas_se_formatean_sin_notacion_cientifica(cfg, ra, dec):
    """Un `1e-07` en la URL haria que el servicio devuelva el campo equivocado."""
    query = parse_qs(urlparse(url_cutout(cfg, ra, dec)).query)
    assert "e" not in query["ra"][0].lower()
    assert "e" not in query["dec"][0].lower()
    assert float(query["ra"][0]) == pytest.approx(ra, abs=1e-6)
    assert float(query["dec"][0]) == pytest.approx(dec, abs=1e-6)


def test_configuracion_pide_tres_canales_no_cinco_bandas(cfg):
    """El cutout es una composicion RGB de g, r, i: tres canales, no cinco bandas.

    Si alguien reintroduce una lista de bandas en el config, es señal de que se
    volvio al modelo mental equivocado.
    """
    assert "bandas" not in cfg.imagenes, (
        "El cutout JPEG no entrega bandas separadas. Para ugriz real hay que "
        "usar los FITS por banda del SAS, no ImgCutout."
    )
