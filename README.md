# DeepLearning Experimentos

Colección de scripts independientes para tareas de clasificación, regresión e imágenes, con un enfoque educativo. Cada carpeta se ejecuta de forma autónoma y demuestra un patrón distinto (Dense, CNN, LSTM, multi‑salida, wrappers sklearn, etc.).

## 1. Tabla de scripts

| Carpeta | Tipo | Modelo / Arquitectura | Dataset | Notas |
|---------|------|-----------------------|---------|-------|
| 01-wine | Clasificación multiclase | MLP denso (Input + Dense capas) | Wine | Salida ahora con resumen legible y predicciones formateadas |
| 02-mushrooms | Clasificación binaria | MLP denso | Mushrooms | One-hot extensivo, ideal para ver sparsidad |
| 03-house-prices | Regresión | MLP (varias variantes) | King County | Incluye versión con GridSearch (`scikeras`) y otra scikit wrapper |
| 04-videogames | Regresión multi‑salida | Modelo funcional multi‑cabeza | Ventas VG | Predice ventas NA/EU/JP simultáneamente |
| 05-sq-tri | Clasificación imágenes | CNN simple (Keras) | Formas (imágenes) | Usa utilidades `Matrix_CV_ML.py` |
| 06-bolts-nuts | Clasificación imágenes | CNN simple | Imágenes 3D (tornillo/tuerca) | Utilidad `Matrix_CV_ML3D.py` |
| 07-new-houses-london | Serie temporal | LSTM stack | New houses London | Normaliza y modela secuencia |
| 08-temperature | Serie temporal | LSTM stack | Temperature | Similar al anterior, distinta serie |

## 2. Dependencias

Archivo `requirements.txt` (solo dependencias de primer nivel): numpy, pandas, scikit-learn, scipy, matplotlib, opencv-python, tensorflow, scikeras. Transitivos (gast, protobuf, etc.) se instalan automáticamente. Si TensorFlow falla con Python 3.13, crea un entorno alterno 3.11/3.12.

## 3. Entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Verificación rápida

```bash
source venv/bin/activate
python -c "import numpy, pandas, sklearn, cv2, tensorflow; print('OK')"
```

## 5. Ejecución normal

```bash
source venv/bin/activate
python 01-wine/01-wine.py
python 02-mushrooms/02-mushrooms.py
python 03-house-prices/03-kingcounty-house-prices.py
python 03-house-prices/03-kingcounty-house-prices-grid.py
python 03-house-prices/03-kingcounty-house-prices-scikit.py
python 04-videogames/04-videogames.py
python 05-sq-tri/05-sq-tri-keras.py
python 06-bolts-nuts/06-bolts-nuts.py
python 07-new-houses-london/07-new-houses-london.py
python 08-temperature/08-temperature.py
```

## 6. Ejecución rápida (sanity check)

Todos los scripts aceptan la variable de entorno `EPOCHS_OVERRIDE` para reducir épocas y validar que corren sin entrenar a fondo:

```bash
export EPOCHS_OVERRIDE=1  # ó 5 para algo mínimamente informativo
python 01-wine/01-wine.py > quickruns/logs/wine.txt 2>&1
```

En algunos scripts de predicciones puedes limitar ejemplos usando `PRED_SAMPLE_LIMIT`. Revisa los logs en `quickruns/logs/`.

### 6.1 Script único / filtro por alias

Se añadió `scripts/run.sh` para ejecutar uno o varios scripts por alias.

Alias disponibles: `01 02 03 03-grid 03-scikit 04 05 06 07 08`.

Ejemplos:

```bash
./scripts/run.sh

# Sólo wine
./scripts/run.sh 01

# House prices (versión principal) y videogames
./scripts/run.sh 03,04

# Con más épocas y límite de predicciones diferente
EPOCHS_OVERRIDE=5 PRED_SAMPLE_LIMIT=8 ./scripts/run.sh 01,02

# Ayuda
./scripts/run.sh --help
```

Los logs ahora se guardan con el nombre del script: `quickruns/logs/01-wine.txt`, `quickruns/logs/03-kingcounty-house-prices-grid.txt`, etc. (antes se usaba `<alias>.txt`). Se genera además `summary.tsv`.

## 7. Logs y artefactos

`quickruns/logs/` almacena salidas de ejecuciones rápidas. Convención: cada script `X/Nombre.py` produce `quickruns/logs/Nombre.txt`. Artefactos de entrenamiento (si se guardan) deben añadirse a `.gitignore` si crecen (modelos grandes, checkpoints, etc.).

## 8. Estructura de datos

Cada carpeta tiene `data/` con CSV original. Algunos scripts generan `trainpred.csv`, `testpred.csv` o similares en la raíz para inspección rápida.

## 9. Extender / mejoras

Sugerencias:

- Añadir early stopping y callbacks de TensorBoard.
- Integrar validación train/val split (actualmente algunos entrenan sobre todo el dataset para simplicidad).
- Unificar pipeline de preprocesado (normalización, splits) en módulo común.
- Añadir tests simples (shape, pérdida < umbral tras pocas épocas).
- Script `run_all.sh` para reproducciones rápidas.

## 10. Git (opcional inicialización)

```bash
git init
git add .
git commit -m "Inicializa proyecto DeepLearning"
git branch -M main
git remote add origin git@github.com:<USUARIO>/deeplearning.git
git push -u origin main
```

## 11. Licencia

Elegir MIT / Apache-2.0 / otra y crear `LICENSE`.

---
Contribuciones: abre una issue o PR.
