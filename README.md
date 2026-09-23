# Predicción de metalicidad gaseosa desde imágenes de banda ancha de SDSS

[![CI](https://github.com/benjamitchell/Predicting-Gas-Metallicity-From-SDSS-Broad-Band-Imaging/actions/workflows/ci.yml/badge.svg)](https://github.com/benjamitchell/Predicting-Gas-Metallicity-From-SDSS-Broad-Band-Imaging/actions/workflows/ci.yml)

Una CNN que estima la metalicidad de fase gaseosa de una galaxia, `12 + log(O/H)`,
a partir únicamente de su imagen óptica — sin espectro.

Proyecto del curso **AS4501 Astroinformática** (Universidad de Chile), desarrollado
junto a Ian Rivera y Esteban Ibarra. Esta es una reescritura completa del trabajo
original; en [docs/legacy/](docs/legacy/) está documentado qué se hizo la primera
vez y por qué no funcionó.

## El problema

La metalicidad gaseosa se mide normalmente del espectro, a partir de las razones
entre líneas de emisión del oxígeno. Eso es caro: requiere seguimiento
espectroscópico galaxia por galaxia. Pero la metalicidad correlaciona fuerte con
propiedades macroscópicas que sí se ven en la luz estelar — sobre todo con la masa,
vía la relación masa-metalicidad de [Tremonti et al. (2004)](https://doi.org/10.1086/423264).

La pregunta es si una CNN puede recuperar la metalicidad directamente de la imagen.
[Wu & Boada (2019)](https://arxiv.org/abs/1810.12913) mostraron que sí: con imágenes
*gri* de 128×128 px obtienen un RMSE de **0.085 dex**, mejor que un random forest
sobre fotometría de cinco bandas. Este repo reproduce esa metodología.

## Método

**Catálogo.** Se parte de los catálogos MPA-JHU DR8 (`galSpecInfo`, `galSpecLine`,
`galSpecExtra`), ~1.84 millones de espectros. El target es `OH_P50`, la mediana de
la distribución de verosimilitud de metalicidad que ajusta el pipeline MPA-JHU.

Cortes aplicados, en orden:

| Corte | Motivo |
|---|---|
| S/N mediana del espectro > 10 | calidad del espectro |
| S/N > 5 en Hα, Hβ, [NII]6584, [OIII]5007 | las líneas que alimentan el estimador de Z |
| `BPTCLASS == 1` (star-forming) | en presencia de un AGN el calibrador de metalicidad no es válido |
| 7.6 < 12+log(O/H) < 9.5 | rango físico |
| 0.02 < z < 0.38 | bajo 0.02 la fibra cubre muy poca galaxia; sobre 0.38 la imagen no resuelve |

El corte por BPT no está en el paper original; es una mejora deliberada.

**Imágenes.** Un cutout RGB de 128×128 px a 0.296″/px (38″×38″) por galaxia, desde
el servicio `ImgCutout` del SkyServer. Es una composición Lupton de las bandas
*g*, *r* e *i*: **tres canales con información distinta**.

> ⚠️ El endpoint `getjpeg` **no acepta un parámetro `band`**. Si se le pasa
> `band=u`, lo ignora en silencio y devuelve la misma imagen, con código 200 y sin
> warning. Pedir cinco "bandas" entrega cinco archivos byte-idénticos. Este fue
> exactamente el error que invalidó el intento anterior.

**Normalización.** Media y desviación por canal, calculadas una sola vez sobre el
split de train y reutilizadas en todo lo demás. Nunca min-max por imagen: eso
reescala cada galaxia por separado y borra el color absoluto, que es justamente la
señal. El paper lo cuantifica — pasar de solo *r* a *gri* baja el RMSE de 0.138 a
0.085 dex.

**Muestreo.** Distribución natural de metalicidad, sin reequilibrar por bins.
Reequilibrar infla artificialmente la varianza del target y vuelve el RMSE
incomparable con la literatura. El desbalance se reporta con métricas por bin.

**Modelo.** ResNet-34 pre-entrenada en ImageNet con cabeza de regresión, igual que
el paper. El target se entrega estandarizado para que la red no gaste las primeras
épocas aprendiendo el offset de ~8.8.

## Uso

```bash
git clone https://github.com/benjamitchell/Predicting-Gas-Metallicity-From-SDSS-Broad-Band-Imaging.git
cd Predicting-Gas-Metallicity-From-SDSS-Broad-Band-Imaging

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Los datos pesados (2.4 GB de FITS + ~4 GB de imágenes) van fuera del repo:

```bash
export GASMET_DATA_DIR=/ruta/con/espacio/gasmet-data     # en el cluster: scratch
```

Es necesario si el repo vive en una carpeta sincronizada (OneDrive, Dropbox) o si
el home del cluster tiene cuota chica. Sin la variable, todo queda en `data/`.

El pipeline son cuatro pasos, en orden:

```bash
python scripts/01_construir_catalogo.py     # descarga 2.4 GB de FITS, cachea
python scripts/02_descargar_imagenes.py     # un request por galaxia, reanudable
python scripts/03_entrenar.py
python scripts/04_evaluar.py                # único paso que toca el test
```

Para probar la cadena completa sin bajar 100k imágenes:

```bash
python scripts/02_descargar_imagenes.py --limite 500
python scripts/03_entrenar.py --arquitectura cnn_simple --nombre prueba --epocas 5
python scripts/04_evaluar.py  --arquitectura cnn_simple --nombre prueba
```

En un cluster con SLURM:

```bash
sbatch cluster/descargar_datos.slurm
sbatch cluster/entrenar.slurm resnet34
```

Los `.slurm` traen `--partition`, `--account` y los `module load` como marcadores;
hay que ajustarlos al cluster.

Todos los parámetros viven en [config.yaml](config.yaml). Ningún script hardcodea
valores.

## Resultados

**La muestra.** De 1.843.200 espectros de DR8, sobreviven a los cortes **118.103
galaxias** — cerca de las 116.429 de Wu & Boada, lo que da confianza en que la
selección está bien. La metalicidad va de 7.67 a 9.47 con mediana 8.99 y σ = 0.188
dex. La MZR de la muestra reproduce la forma de Tremonti+04.

**Corrida de validación** (2000 galaxias, ResNet-18, 6 épocas, CPU) — sirve para
verificar la cadena, no como resultado:

| | RMSE | NMAD |
|---|---|---|
| Baseline (predecir la media) | 0.194 | — |
| Esta corrida (n=301 test) | **0.120** | 0.090 |
| Wu & Boada (2019) | 0.085 | 0.067 |

Supera claramente al baseline con 1399 imágenes de entrenamiento, y la pérdida de
validación seguía bajando en la última época.

_Pendiente: el entrenamiento sobre la muestra completa (100k, ResNet-34, GPU)._

La evaluación reporta RMSE y NMAD en dex — las mismas métricas del paper, para que
los números sean directamente comparables — junto a dos referencias obligatorias:

- **Baseline de predecir siempre la media.** En el paper da ~0.20 dex. Un modelo que
  no lo supere no está extrayendo nada de las imágenes, por bonito que sea su R².
- **Wu & Boada (2019):** RMSE 0.085 dex, NMAD 0.067 dex.

`scripts/04_evaluar.py` genera en [figures/](figures/): la relación masa-metalicidad
de la muestra (control de sanidad del catálogo), las curvas de entrenamiento,
predicción vs. real, los residuos, el error por bin de metalicidad y una grilla de
cutouts con sus predicciones.

## Estructura

```
src/gasmet/
  config.py      carga de config.yaml
  catalog.py     descarga de MPA-JHU, cruce y cortes
  images.py      cutouts de SDSS y empaquetado a memmap
  dataset.py     splits, normalización y Dataset de PyTorch
  model.py       ResNet pre-entrenada y una CNN simple de referencia
  train.py       bucle de entrenamiento con early stopping
  evaluate.py    métricas (RMSE, NMAD, por bin) y baselines
  plots.py       figuras
scripts/         los cuatro pasos del pipeline
cluster/         submit scripts de SLURM
tests/           tests unitarios, incluidos los de regresión del bug original
docs/legacy/     el intento original y la auditoría de por qué falló
```

## Desarrollo

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ scripts/ tests/
```

Los tests no tocan la red ni los datos descargados: corren en segundos. Varios
son de regresión sobre los errores que invalidaron el intento original —
[test_images.py](tests/test_images.py) falla si alguien vuelve a poner un
parámetro `band` en la URL del cutout, y
[test_dataset.py](tests/test_dataset.py) falla si la normalización vuelve a ser
por imagen en vez de con estadísticas compartidas.

## Referencias

- Wu, J. F. & Boada, S. (2019). *Using convolutional neural networks to predict
  galaxy metallicity from three-color images.* MNRAS 484, 4683.
  [arXiv:1810.12913](https://arxiv.org/abs/1810.12913) ·
  [código](https://github.com/jwuphysics/galaxy-cnns)
- Tremonti, C. A. et al. (2004). *The origin of the mass-metallicity relation.*
  ApJ 613, 898.
- Catálogos MPA-JHU DR8: https://www.sdss4.org/dr17/spectro/galaxy_mpajhu/

## Créditos

Benjamín Mitchell, Ian Rivera y Esteban Ibarra — AS4501 Astroinformática,
Universidad de Chile.
