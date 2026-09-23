# Metodología

Este documento justifica cada decisión del pipeline. El [README](../README.md)
explica *qué* hace el proyecto; esto explica *por qué* lo hace así, y dónde se
aparta deliberadamente de [Wu & Boada (2019)](https://arxiv.org/abs/1810.12913),
que es la referencia que reproduce.

---

## 1. La pregunta

La metalicidad de fase gaseosa de una galaxia, expresada como `12 + log(O/H)`,
mide cuánto oxígeno hay en su medio interestelar respecto al hidrógeno. Es una
cantidad central en evolución de galaxias: registra la historia acumulada de
formación estelar, de vientos que expulsan metales y de acreción de gas
primordial.

Se mide del espectro, a partir de razones entre líneas de emisión. El problema
es de escala: la espectroscopía requiere apuntar el telescopio galaxia por
galaxia. Los surveys fotométricos que vienen — LSST entre ellos — van a
producir imágenes de miles de millones de galaxias para las que nunca habrá
espectro.

La pregunta, entonces: **¿cuánta información sobre la metalicidad hay en una
imagen de banda ancha?**

A priori uno esperaría que poca y solo indirecta, vía la relación
masa-metalicidad de [Tremonti et al. (2004)](https://doi.org/10.1086/423264):
las galaxias más masivas son más metálicas, y la masa sí se estima de la
fotometría. Wu & Boada muestran que hay bastante más que eso — su CNN predice
con 0.085 dex de RMSE, y el MZR construido con sus predicciones tiene una
dispersión *igual o menor* que el empírico, lo que no podría pasar si el
modelo solo estuviera estimando masa y aplicando la relación.

---

## 2. El target

Se usa `OH_P50` del catálogo **MPA-JHU**: la mediana de la distribución de
verosimilitud de metalicidad que el pipeline ajusta siguiendo a Tremonti+04,
comparando las líneas observadas contra una grilla de modelos de fotoionización.

Tres cosas que conviene tener claras sobre este target:

**No es una medición directa.** Es el resultado de un ajuste a modelos. Su
incertidumbre sistemática típica es de ~0.03 dex, lo que fija un piso: ningún
modelo puede predecir mejor que el ruido de su propia etiqueta.

**Está en escala de Tremonti+04.** Los distintos calibradores de metalicidad
(Tremonti, Pettini & Pagel, Kobulnicky & Kewley…) difieren entre sí hasta en
~0.7 dex en el cero absoluto. Los números de este proyecto no son comparables
con trabajos que usen otro calibrador.

**Solo es válido para galaxias con formación estelar.** El calibrador supone
que las líneas de emisión vienen de regiones HII ionizadas por estrellas
jóvenes. Si hay un núcleo activo, el campo de radiación es completamente
distinto y el valor ajustado no significa nada. De ahí el corte por BPT.

---

## 3. Selección de la muestra

Los tres catálogos de MPA-JHU DR8 son **paralelos**: misma longitud, misma fila
para el mismo espectro. Se unen por posición de fila, no con un `merge` sobre
`SPECOBJID`, por dos razones: es innecesario, y ~370.000 filas de DR8 traen ese
campo en blanco, lo que en un join las cruzaría todas contra todas (en la
práctica, una petición de 1023 GiB de memoria). La alineación se verifica
explícitamente en `_unir`; si los archivos fueran de releases distintos, el
pipeline falla en vez de asignar en silencio la metalicidad de una galaxia a la
imagen de otra.

La cadena de cortes, con las galaxias que sobreviven a cada uno:

| Corte | Quedan | Por qué |
|---|---:|---|
| catálogo completo | 1.843.200 | |
| `SPECOBJID` válido | 1.472.581 | filas sin contraparte espectroscópica |
| `Z_WARNING == 0` | 1.382.698 | el redshift del pipeline es confiable |
| `RELIABLE != 0` | 920.019 | bandera de calidad de MPA-JHU |
| `SN_MEDIAN > 10` | 588.719 | espectro con suficiente señal |
| S/N > 5 en 4 líneas | 189.323 | ver abajo |
| `BPTCLASS == 1` | 124.009 | solo star-forming |
| 7.6 < 12+log(O/H) < 9.5 | 123.735 | rango físicamente plausible |
| 0.02 < z < 0.38 | 118.103 | ver abajo |

**S/N > 5 en Hα, Hβ, [NII]6584 y [OIII]5007.** Son las cuatro líneas que
alimentan el estimador. Si alguna es ruidosa, `OH_P50` hereda ese ruido aunque
el ajuste igual converja. Es el corte más agresivo de la cadena — se lleva dos
tercios de la muestra — y es el que más protege la calidad de las etiquetas.

**`BPTCLASS == 1`.** El diagrama BPT separa galaxias con formación estelar de
las dominadas por un AGN, usando las razones [OIII]/Hβ y [NII]/Hα. Wu & Boada
**no** aplican este corte; nosotros sí, porque como se explicó arriba el
calibrador de metalicidad no es válido en presencia de un AGN. Es una mejora
deliberada respecto al paper, al costo de una muestra algo menor.

**0.02 < z < 0.38.** Bajo z = 0.02 la fibra de 3″ de SDSS cubre una fracción
muy pequeña de la galaxia, y la metalicidad medida corresponde solo al núcleo,
que es sistemáticamente más metálico que el promedio. Sobre z = 0.38 la imagen
ya casi no resuelve estructura. El rango coincide con el de Wu & Boada.

Sobreviven **118.103 galaxias**, notablemente cerca de las 116.429 del paper
pese a partir de un data release distinto y aplicar cortes distintos. La
muestra final se submuestrea al azar a 100.000, sin tocar la forma de la
distribución.

### Sobre el muestreo

Se usa la **distribución natural** de metalicidad, que está fuertemente
concentrada en torno a 8.9 (σ = 0.188 dex). Es tentador reequilibrarla —
tomar N galaxias por bin de metalicidad para que el modelo vea el rango
completo — pero tiene dos costos que no valen la pena:

1. **Rompe la comparabilidad.** Un dataset uniforme tiene mucha más varianza en
   el target que el real, así que el R² y el RMSE dejan de ser comparables con
   la literatura y con el baseline.
2. **Bota la mayor parte de los datos.** Los bins extremos tienen pocos objetos,
   así que el tamaño de la muestra queda limitado por el bin más pobre.

El desbalance es real y hay que reportarlo, no esconderlo: por eso la
evaluación incluye métricas desglosadas por bin de metalicidad, que muestran
explícitamente dónde el modelo es peor.

---

## 4. Las imágenes

Un cutout de **128 × 128 px a 0.296″/px**, o sea 38″ × 38″ de cielo, desde el
servicio `ImgCutout` del SkyServer. Mismos valores que el paper.

El servicio devuelve un **JPEG RGB**: una composición de las bandas *g*, *r* e
*i* según el algoritmo de [Lupton et al. (2004)](https://doi.org/10.1086/382245),
que aplica un stretch asinh y asigna R=*i*, G=*r*, B=*g*. Son **tres canales
con información distinta**, no tres copias.

> ### ⚠️ El error que invalidó el intento original
>
> El endpoint `getjpeg` **no acepta un parámetro `band`**. Si se le pasa
> `band=u`, `band=g`, etc., lo ignora en silencio y devuelve siempre la misma
> imagen, con código 200 y sin ningún warning.
>
> El primer desarrollo de este proyecto pedía las cinco bandas *ugriz* así y
> recibía cinco archivos byte-idénticos. El preprocesamiento además tomaba
> `img[:,:,0]` de cada uno, de modo que el tensor `(5, 64, 64)` que alimentaba
> la red eran **cinco copias del mismo canal en escala de grises**.
>
> El bug no producía ningún síntoma: los archivos se guardaban, el modelo
> entrenaba y devolvía un R² positivo. Ver [docs/legacy/](legacy/).
>
> Para *ugriz* real hay que bajar los FITS por banda del SAS y recortar. Es
> posible, pero mucho más pesado, y el paper muestra que *gri* basta para llegar
> a 0.085 dex.

### Por qué el tamaño angular es fijo

38″ es un tamaño **angular**, no físico: a z = 0.02 abarca ~15 kpc y a z = 0.38
unos ~200 kpc. Es una limitación heredada del paper, y la razón de acotar el
rango de redshift. Una alternativa sería recortar un tamaño físico fijo, pero
eso introduce su propio sesgo, porque la resolución efectiva pasaría a depender
del redshift.

---

## 5. Preprocesamiento

**Normalización por canal, con estadísticas fijas.** Se calculan la media y la
desviación de cada canal una sola vez, sobre el split de entrenamiento, y se
aplican igual a train, val y test.

Lo que hay que **no** hacer es normalizar cada imagen por separado (min-max o
z-score por galaxia). Eso reescala cada objeto a un rango común y **borra el
brillo y el color absolutos**, que son precisamente la señal. El paper lo
cuantifica con su ablación de bandas:

| Bandas | RMSE |
|---|---|
| solo *r* | 0.138 dex |
| *gr* | 0.0915 dex |
| *gri* | 0.0851 dex |

Toda esa ganancia es información de color. Normalizar por imagen la destruye.

Calcular las estadísticas solo con train tampoco es cosmético: usar el dataset
completo filtraría información de validación y test al entrenamiento.

**No se vuelve a aplicar un stretch.** El JPEG ya trae el asinh de Lupton
aplicado. Un segundo stretch comprimiría aún más el rango dinámico sin ganar
nada. Por eso `preprocesamiento.stretch` es `null` en la configuración, y hay
un test que lo verifica.

**Augmentation: solo rotaciones de 90° y flips.** La orientación de una galaxia
en el cielo es un accidente de perspectiva y no aporta información física sobre
su metalicidad, así que el grupo diédrico es una simetría legítima del problema.
Zoom o recortes aleatorios **no** se usan: alterarían el tamaño aparente, que
correlaciona con la masa y por tanto con la metalicidad — se estaría destruyendo
señal real.

**Splits 70/15/15**, aleatorios y no estratificados, para que los tres conjuntos
compartan la distribución natural. El test se aparta al construir el dataset y
solo lo toca `scripts/04_evaluar.py`.

---

## 6. Modelo

**ResNet-34 pre-entrenada en ImageNet**, con la capa final reemplazada por una
cabeza de regresión (dropout + lineal a una salida, sin activación). Es la misma
arquitectura del paper.

El transfer learning importa más de lo que parece. Los filtros de bajo nivel —
bordes, gradientes de color, texturas — son los mismos para fotos de gatos que
para galaxias, y aprenderlos desde cero exigiría muchos más datos. El repo
incluye una `cnn_simple` entrenada desde cero justamente para poder medir
cuánto aporta.

**El target se entrega estandarizado** (media 0, σ 1 según el split de train).
Sin esto la red parte prediciendo valores cercanos a 0 contra objetivos de ~8.9
y gasta las primeras épocas aprendiendo nada más que el offset. En el intento
original esto era visible: el entrenamiento partía con un MSE de 7.64, que es
aproximadamente 8.8².

Optimizador AdamW con weight decay, learning rate en coseno, precisión mixta, y
early stopping sobre el RMSE de validación. El paper entrena 10 épocas con
one-cycle; acá se permiten hasta 60 con paciencia de 10, que es más conservador.

---

## 7. Métricas

Se reportan **RMSE y NMAD en dex**, las mismas del paper, para que los números
sean directamente comparables.

El **NMAD** (desviación absoluta mediana normalizada, `1.4826 × mediana|x − mediana(x)|`)
aproxima σ en datos gaussianos pero es insensible a outliers. Reportarlo junto
al RMSE permite distinguir "el modelo es malo en general" de "el modelo es bueno
salvo por unos pocos casos patológicos".

**El R² se reporta pero no se usa como métrica principal.** Depende de la
varianza del target, así que cambia si uno cambia la composición de la muestra —
exactamente el problema que tenía el intento original, cuyo R² = 0.45 estaba
calculado sobre un dataset reequilibrado y no significaba lo que parecía.

Dos referencias acompañan siempre el resultado:

- **Baseline de predecir siempre la media del train.** Su RMSE es aproximadamente
  σ del target: ~0.188 dex en nuestra muestra, ~0.20 dex en la del paper. Un
  modelo que no lo supere no está extrayendo *nada* de las imágenes. El código
  lo verifica y emite un warning explícito si eso ocurre.
- **Wu & Boada (2019):** RMSE 0.085 dex, NMAD 0.067 dex.

Además se calculan métricas **por bin de metalicidad**, porque con la
distribución natural una cifra global esconde el desempeño en los extremos.

---

## 8. Diferencias deliberadas con el paper

| | Wu & Boada (2019) | Este repo |
|---|---|---|
| Data release | DR7 | DR8 |
| Corte de calidad | `rChi2 < 2` | `SN_MEDIAN > 10` + S/N > 5 en 4 líneas |
| AGN | sin filtrar | `BPTCLASS == 1` |
| Cortes fotométricos | 10 < *ugriz* < 25, 0 < *u−r* < 6, petroMag_r < 18 | ninguno |
| Muestra | 116.429 | 100.000 (de 118.103) |
| Splits | 80/20 + test de 20.466 aparte | 70/15/15 |
| Entrenamiento | 10 épocas, one-cycle | hasta 60, coseno + early stopping |

La ausencia de cortes fotométricos es la diferencia menos deseable: los
catálogos `galSpec*` no traen fotometría, así que habría que cruzar con
`photoObj`. El corte por `SN_MEDIAN` cubre parcialmente el mismo objetivo
(descartar objetos demasiado débiles), pero no es equivalente.

---

## 9. Limitaciones conocidas

**El JPEG no es flujo calibrado.** Trae un stretch no lineal y está cuantizado a
8 bits por canal. Se pierde rango dinámico respecto a los FITS. El paper acepta
la misma limitación y aun así llega a 0.085 dex, pero es una vía clara de mejora.

**Solo tres bandas.** Las bandas *u* y *z* aportan información sobre formación
estelar y masa estelar respectivamente. El propio paper nota que un random
forest sobre fotometría *ugriz* de cinco bandas es competitivo con su CNN de
tres bandas.

**La muestra no es representativa del universo.** Es el resultado de una cadena
de cortes que favorece galaxias brillantes, con formación estelar activa y
líneas de emisión fuertes. Cualquier conclusión aplica a esa población, no a
las galaxias en general.

**El techo no es cero.** La incertidumbre sistemática de las etiquetas es de
~0.03 dex, y la dispersión intrínseca del MZR ronda los 0.10 dex. El paper
discute en su §6.2 que su RMSE de 0.085 dex ya está cerca del presupuesto de
error disponible.

---

## Referencias

- Wu, J. F. & Boada, S. (2019). *Using convolutional neural networks to predict
  galaxy metallicity from three-color images.* MNRAS 484, 4683.
  [arXiv:1810.12913](https://arxiv.org/abs/1810.12913)
- Tremonti, C. A. et al. (2004). *The origin of the mass-metallicity relation.*
  ApJ 613, 898.
- Lupton, R. et al. (2004). *Preparing red-green-blue images from CCD data.*
  PASP 116, 133.
- Baldwin, Phillips & Terlevich (1981). *Classification parameters for the
  emission-line spectra of extragalactic objects.* PASP 93, 5. (el diagrama BPT)
- Catálogos MPA-JHU: https://www.sdss4.org/dr17/spectro/galaxy_mpajhu/
