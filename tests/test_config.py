"""Tests de la carga de configuracion y la resolucion de rutas."""


import pytest

from gasmet.config import VAR_DIR_DATOS, Config, cargar, ruta_absoluta


def test_acceso_por_atributo_anidado():
    cfg = cargar()
    assert cfg.imagenes.tamano_px == cfg["imagenes"]["tamano_px"]
    assert isinstance(cfg.entrenamiento, Config)


def test_clave_inexistente_da_un_error_util():
    cfg = cargar()
    with pytest.raises(AttributeError, match="Claves disponibles"):
        _ = cfg.no_existe


def test_config_inexistente_falla_temprano(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar(tmp_path / "no_esta.yaml")


def test_rutas_relativas_se_resuelven_contra_la_raiz(monkeypatch):
    monkeypatch.delenv(VAR_DIR_DATOS, raising=False)
    assert ruta_absoluta("results").is_absolute()
    assert ruta_absoluta("data/catalog").parts[-2:] == ("data", "catalog")


def test_rutas_absolutas_se_respetan(monkeypatch, tmp_path):
    monkeypatch.setenv(VAR_DIR_DATOS, str(tmp_path / "otro"))
    absoluta = (tmp_path / "data" / "x.parquet").resolve()
    assert ruta_absoluta(absoluta) == absoluta


def test_la_variable_de_entorno_redirige_solo_los_datos(monkeypatch, tmp_path):
    """Hace falta cuando el repo vive en una carpeta sincronizada o en un
    cluster donde los datos van a scratch y no al home."""
    destino = tmp_path / "scratch" / "gasmet-data"
    monkeypatch.setenv(VAR_DIR_DATOS, str(destino))

    datos = ruta_absoluta("data/processed/imagenes.npy")
    assert datos == destino.resolve() / "processed" / "imagenes.npy"

    # results/ y figures/ se quedan en el repo: son salidas versionables.
    assert "scratch" not in str(ruta_absoluta("results"))
    assert "scratch" not in str(ruta_absoluta("figures"))


def test_sin_la_variable_los_datos_quedan_en_el_repo(monkeypatch):
    monkeypatch.delenv(VAR_DIR_DATOS, raising=False)
    assert ruta_absoluta("data/processed").parts[-2:] == ("data", "processed")


def test_la_configuracion_por_defecto_es_coherente():
    """Chequeos baratos que pillan una edicion descuidada del YAML."""
    cfg = cargar()
    pre = cfg.preprocesamiento

    total = pre.fraccion_train + pre.fraccion_val + pre.fraccion_test
    assert total == pytest.approx(1.0), "las fracciones del split deben sumar 1"

    assert cfg.seleccion.oh_min < cfg.seleccion.oh_max
    assert cfg.seleccion.z_min < cfg.seleccion.z_max
    assert cfg.imagenes.tamano_px > 0
    assert cfg.entrenamiento.batch_size > 0

    # El JPEG de SDSS ya trae aplicado el stretch asinh de Lupton; volver a
    # aplicarlo seria un error.
    assert pre.stretch is None
