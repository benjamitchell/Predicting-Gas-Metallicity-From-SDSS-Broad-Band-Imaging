"""Tests de la union de catalogos.

Los tres archivos de MPA-JHU son paralelos: misma longitud, misma fila para el
mismo espectro. Cruzarlos con un merge sobre SPECOBJID es innecesario y ademas
peligroso, porque ~370k filas de DR8 traen ese campo en blanco y el join las
multiplica entre si. En DR8 eso pedia 1023 GiB de memoria.
"""

import pandas as pd
import pytest

from gasmet.catalog import _unir


def _tabla(ids, **columnas) -> pd.DataFrame:
    return pd.DataFrame({"SPECOBJID": ids, **columnas})


def test_une_por_posicion_de_fila():
    info = _tabla(["a", "b", "c"], RA=[1.0, 2.0, 3.0])
    extra = _tabla(["a", "b", "c"], OH_P50=[8.5, 8.7, 9.0])

    unido = _unir(info, extra)

    assert list(unido.columns) == ["SPECOBJID", "RA", "OH_P50"]
    assert len(unido) == 3
    assert unido.loc[1, "RA"] == 2.0
    assert unido.loc[1, "OH_P50"] == 8.7


def test_no_duplica_la_columna_de_cruce():
    info = _tabla(["a", "b"], RA=[1.0, 2.0])
    extra = _tabla(["a", "b"], Z=[0.1, 0.2])
    line = _tabla(["a", "b"], H_ALPHA_FLUX=[10.0, 20.0])

    unido = _unir(info, extra, line)

    assert list(unido.columns).count("SPECOBJID") == 1


def test_ids_en_blanco_no_provocan_explosion_combinatoria():
    """El caso real de DR8: muchas filas con SPECOBJID vacio.

    Un merge sobre esa columna daria n*n filas. Unir por posicion las deja
    intactas, y el corte de SPECOBJID valido las elimina despues.
    """
    n = 500
    ids = [""] * n
    info = _tabla(ids, RA=list(range(n)))
    extra = _tabla(ids, OH_P50=[8.8] * n)

    unido = _unir(info, extra)

    assert len(unido) == n, "unir por posicion no debe multiplicar filas"
    assert len(unido) != n * n


def test_rechaza_catalogos_de_distinto_largo():
    info = _tabla(["a", "b", "c"], RA=[1.0, 2.0, 3.0])
    extra = _tabla(["a", "b"], OH_P50=[8.5, 8.7])

    with pytest.raises(ValueError, match="mismo largo"):
        _unir(info, extra)


def test_rechaza_catalogos_desalineados():
    """La alineacion se verifica en vez de asumirse.

    Si los archivos fueran de data releases distintos, unir por posicion
    asignaria en silencio la metalicidad de una galaxia a la imagen de otra.
    """
    info = _tabla(["a", "b", "c"], RA=[1.0, 2.0, 3.0])
    extra = _tabla(["a", "c", "b"], OH_P50=[8.5, 9.0, 8.7])

    with pytest.raises(ValueError, match="alineados"):
        _unir(info, extra)
