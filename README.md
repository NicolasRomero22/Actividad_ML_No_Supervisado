# Segmentación de inmuebles en arriendo — ML No Supervisado

Universidad de Cundinamarca · ESP. Inteligencia Artificial · CADI – Introducción a Machine Learning · 2026-2

Segmentación de 1000 inmuebles en arriendo con K-Means y PCA (scikit-learn, Matplotlib).

| Archivo | Contenido |
|---|---|
| `1_KMeans_Inmuebles.py` | K-Means con K = 2, 3 y 4; método del codo y silueta |
| `2_PCA_Inmuebles.py` | PCA, varianza explicada, cargas, K-Means sobre variables vs. sobre componentes |
| `DATASET_Inmuebles.xlsx` | Dataset (1000 registros) |

## Ejecución (PyScripter)
1. `pip install -r requirements.txt`
2. Dejar `DATASET_Inmuebles.xlsx` en la misma carpeta que los scripts.
3. Abrir un script en PyScripter y ejecutarlo (Run). Los resultados salen por consola y en ventanas de gráficas; no se crean archivos.

## Notas sobre los datos
- `Tipo_inmueble` se descarta: es redundante con `Localidad`.
- `Estrato.1` es una columna duplicada sin relación con el resto y se descarta.
- `ID_inmueble` se conserva para identificar registros pero no se usa para calcular distancias.
