# El intento original (julio 2025)

Este directorio guarda los notebooks del primer desarrollo del proyecto y la
auditoría de por qué no funcionó. Los `.ipynb` no se versionan (pesan ~9 MB con
los outputs embebidos); viven en el Drive del curso. Lo que sí se versiona es
este documento, que es la parte reutilizable.

Se conserva porque el error que lo invalidó es instructivo y difícil de ver.

## Qué se hizo

El catálogo se construyó bien: MPA-JHU DR8, cruce por `SPECOBJID`, corte de
S/N > 5 en seis líneas de emisión, `BPTCLASS == 1`, rango físico de metalicidad.
De 1.843.200 espectros quedaron 84.123 galaxias. **Esa parte se reutilizó casi
tal cual en la reescritura.**

De ahí se muestrearon 100 galaxias por bin de metalicidad de 0.1 dex entre 8.2 y
9.2 — 1000 en total — y se bajaron cutouts de 64×64 px a 0.396″/px, supuestamente
uno por cada banda *ugriz*. Preprocesamiento: crop central, normalización min-max
por imagen, stack a un tensor `(5, 64, 64)`.

Se entrenaron tres modelos: regresión lineal sobre los píxeles crudos, una CNN en
PyTorch y una CNN en Keras.

| Modelo | MAE | R² |
|---|---|---|
| CNN Keras | 0.173 | 0.451 |
| Regresión lineal | 0.332 | — |
| CNN PyTorch | 0.787 | **−7.66** |

## Por qué falló

**Las imágenes nunca fueron multibanda.** Se descargaron con

```
https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=...&dec=...&band=u
```

y ese endpoint **no tiene** un parámetro `band`. Lo ignora en silencio y siempre
devuelve el mismo JPEG RGB. Verificado: las cinco peticiones con `band=u,g,r,i,z`
y la petición sin `band` devuelven un archivo **byte-idéntico**, mismo md5.

Se agrava en el preprocesamiento, que hacía `img = img[:, :, 0]` — el canal rojo —
para cada "banda". El tensor `(5, 64, 64)` que alimentó a la red eran **cinco
copias del mismo canal en escala de grises**.

Dado que el color es la señal dominante (el paper: RMSE 0.138 con solo *r*, 0.085
con *gri*), eso solo explica el resultado.

Los demás problemas, por impacto:

1. **Normalización min-max por imagen.** Reescala cada galaxia a [0,1] por
   separado, borrando el brillo y el color absolutos. Aunque los canales hubieran
   sido distintos, esto habría destruido la señal igual.
2. **1000 galaxias contra 116.429 del paper**, entrenando desde cero en vez de
   hacer transfer learning. La CNN de PyTorch partió con un train loss de 7.64,
   o sea prediciendo ~0 contra objetivos de ~8.8: ni siquiera había aprendido el
   offset.
3. **Muestreo uniforme por bins.** Infla la varianza del target, así que ese
   R² = 0.45 no es comparable con el paper. El MAE de 0.173 equivale a un RMSE de
   ~0.21 dex, que es aproximadamente lo que Wu & Boada reportan como el *peor
   resultado posible*: predecir siempre la media (~0.20 dex).
4. **No hubo conjunto de test.** La celda de evaluación final falla con
   `NameError: test_loader is not defined`.

## Qué cambió en la reescritura

| Antes | Ahora |
|---|---|
| `band=u,g,r,i,z` → 5 copias del mismo archivo | un cutout *gri*, sus 3 canales reales |
| `img[:, :, 0]`, un canal replicado | los 3 canales RGB |
| min-max por imagen | media/σ por canal, fijas, calculadas solo en train |
| 64 px a 0.396″/px (25″) | 128 px a 0.296″/px (38″), igual que el paper |
| 1000 galaxias, uniforme por bins | distribución natural, ~100k |
| CNN desde cero | ResNet-34 pre-entrenada en ImageNet |
| sin test | train/val/test 70/15/15, test intocado hasta el final |
| R² sobre un target reequilibrado | RMSE y NMAD en dex, contra el baseline de la media |

## La lección

El bug no producía ningún síntoma. La API respondía 200, guardaba cinco archivos
con nombres distintos, el modelo entrenaba y devolvía un R² positivo. Sin un
`md5sum` o una inspección visual de las bandas lado a lado, no había forma de
notarlo desde el código.

De ahí que `images.py` valide el tamaño y el modo de cada imagen recibida antes de
aceptarla, y que `evaluate.py` avise explícitamente cuando el modelo no supera al
baseline de predecir la media.
