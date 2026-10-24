# %%
# Laboratorio 2 — Reducción de dimensionalidad con PCA + K-Means
# Universidad de Cundinamarca · ESP. Inteligencia Artificial · CADI – Introducción a Machine Learning · 2026-2
#
# Modifica el ejemplo Algoritmo_InmueblesPCA.py para leer el dataset Excel (en lugar de un CSV),
# analiza la varianza explicada, interpreta los componentes y compara K-Means sobre las variables
# preprocesadas frente a K-Means sobre los componentes principales.

# %%
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

NOMBRE_EXCEL = "DATASET_Inmuebles.xlsx"
fmt_cop = lambda x, _=None: f"${x/1e6:.1f}M"
COLORES = ["#3b6fb6", "#f2a541", "#d1495b", "#2a9d8f", "#7a5195"]


def localizar_dataset(nombre=NOMBRE_EXCEL):
    """Busca el Excel junto al script o en el directorio actual."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    for ruta in [os.path.join(base, nombre), 
                 os.path.join(os.getcwd(), nombre), ]:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(f"No se encontró {nombre}. Colócalo en la misma carpeta que este script.")


def mostrar():
    """Muestra la gráfica en pantalla (no se guarda ningún archivo)."""
    plt.tight_layout()
    plt.show()


# %%
# 1. Cargar el dataset
# El identificador se conserva para interpretar resultados pero no entra al cálculo de distancias.
# Se descartan Tipo_inmueble (redundante: cada localidad tiene un único tipo, Chapinero→Casa, Suba→Apto,
# Usaquén→ApartaEstudio) y la columna duplicada Estrato.1.

# %%
df = pd.read_excel(localizar_dataset())
df = df.rename(columns={
    "Inmueble": "ID_inmueble", "Área m²": "Area_m2", "Precio arriendo": "Precio_arriendo",
    "Baños": "Banos", "Antig[uedad": "Antiguedad_anios", "Tipo Inmueble": "Tipo_inmueble",
}).drop(columns=["Estrato.1"])

print("Redundancia Localidad–Tipo_inmueble:")
print(pd.crosstab(df["Localidad"], df["Tipo_inmueble"]))

ids = df["ID_inmueble"]

variables_numericas = ["Precio_arriendo", "Area_m2", "Habitaciones", "Banos", "Antiguedad_anios", "Estrato"]
variables_categoricas = ["Localidad"]

X = df[variables_numericas + variables_categoricas]
print("\nDimensiones antes del preprocesamiento (variables de entrada):", X.shape)

# %%
# 2. Preprocesamiento (escalado + codificación one-hot)

# %%
preprocesamiento = ColumnTransformer(transformers=[
    ("numericas", StandardScaler(), variables_numericas),
    ("categoricas", OneHotEncoder(handle_unknown="ignore"), variables_categoricas),
])
X_preparado = preprocesamiento.fit_transform(X)
nombres_features = list(variables_numericas) + [
    f"Loc_{c}" for c in preprocesamiento.named_transformers_["categoricas"].categories_[0]]

print("Dimensiones después del preprocesamiento:", X_preparado.shape)
print("Variables resultantes:", nombres_features)

# %%
# 3. PCA completo: ¿cuánta varianza explica cada componente?

# %%
pca_full = PCA().fit(X_preparado)
var = pca_full.explained_variance_ratio_
acum = np.cumsum(var)

tabla_var = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(len(var))],
    "Varianza explicada (%)": (var * 100).round(2),
    "Varianza acumulada (%)": (acum * 100).round(2),
})
print(tabla_var.to_string(index=False))
n90 = int(np.argmax(acum >= 0.90) + 1)
print(f"\nComponentes necesarios para conservar >= 90% de la varianza: {n90}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(1, len(var) + 1), var * 100, color="#3b6fb6")
ax[0].set(title="Varianza explicada por componente (scree plot)", xlabel="Componente principal",
          ylabel="% de varianza")
ax[1].plot(range(1, len(var) + 1), acum * 100, "o-", color="#d1495b")
ax[1].axhline(90, ls="--", color="gray")
ax[1].set(title="Varianza acumulada", xlabel="Número de componentes", ylabel="% acumulado")
for a in ax:
    a.grid(alpha=.3)
mostrar()

# %%
# 4. PCA con 2 componentes (el del ejemplo del docente)

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preparado)

print("Dimensiones originales después del preprocesamiento:", X_preparado.shape)
print("Dimensiones después de PCA:", X_pca.shape)
print("Varianza explicada por componente:", pca.explained_variance_ratio_.round(4))
print("Varianza acumulada (PC1+PC2):", round(pca.explained_variance_ratio_.sum(), 4))
print("Varianza NO capturada por 2 componentes:", round(1 - pca.explained_variance_ratio_.sum(), 4))

# Error de reconstrucción: información que se pierde al comprimir a 2 dimensiones
X_rec = pca.inverse_transform(X_pca)
mse = np.mean((X_preparado - X_rec) ** 2)
print("Error cuadrático medio de reconstrucción con 2 componentes:", round(mse, 4))

# %%
# 5. Interpretación de los componentes (cargas o *loadings*)

# %%
cargas = pd.DataFrame(pca.components_.T, index=nombres_features, columns=["PC1", "PC2"])
print(cargas.round(3))

fig, ax = plt.subplots(figsize=(7, 4.5))
cargas.plot.barh(ax=ax, color=["#3b6fb6", "#f2a541"])
ax.axvline(0, color="black", lw=.8)
ax.set(title="Cargas de las variables en PC1 y PC2", xlabel="Peso en el componente")
ax.grid(alpha=.3)
mostrar()

# %%
# 6. Experimento A — K-Means sobre los datos preprocesados (K = 3) y gráfica en PC1–PC2

# %%
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
grupos = kmeans.fit_predict(X_preparado)

# Reordenar para que el grupo 0 sea el de menor precio promedio
orden = df.assign(g=grupos).groupby("g")["Precio_arriendo"].mean().sort_values().index
mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
grupos = pd.Series(grupos).map(mapa).values
df["Grupo_KMeans"] = grupos

print("Silueta (espacio preprocesado):", round(silhouette_score(X_preparado, grupos), 3))

plt.figure(figsize=(8, 5.5))
for g in sorted(df["Grupo_KMeans"].unique()):
    sel = grupos == g
    plt.scatter(X_pca[sel, 0], X_pca[sel, 1], label=f"Grupo {g}", alpha=.7,
                s=14, color=COLORES[g])
plt.xlabel(f"Componente principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Componente principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Segmentación inmobiliaria: K-Means + PCA")
plt.legend()
plt.grid(True, alpha=.3)
mostrar()

# %%
# 7. Perfil comercial de los grupos

# %%
print("Perfil promedio por grupo:")
perfil = df.groupby("Grupo_KMeans")[variables_numericas].mean().round(2)
perfil.insert(0, "inmuebles", df.groupby("Grupo_KMeans").size())
print(perfil)

print("\nDistribución por localidad dentro de cada grupo:")
print(pd.crosstab(df["Grupo_KMeans"], df["Localidad"], normalize="index").round(2))

print("\nSeparación de grupos en PC1 / PC2 (promedio ± desviación):")
print(pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "g": grupos}).groupby("g").agg(["mean", "std"]).round(2))

# %%
# 8. Experimento B — K-Means sobre los componentes principales
# Se compara con el Experimento A usando el *Adjusted Rand Index* (1 = agrupaciones idénticas).

# %%
pca90 = PCA(n_components=n90).fit(X_preparado)
X_pca90 = pca90.transform(X_preparado)

comparacion = []
for nombre, Z in [("Preprocesado (9 dim)", X_preparado),
                  ("PCA 2 componentes", X_pca),
                  (f"PCA {n90} componentes (>=90% var.)", X_pca90)]:
    g = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Z)
    comparacion.append({
        "Espacio de entrada": nombre, "Dimensiones": Z.shape[1],
        "ARI vs. Experimento A": round(adjusted_rand_score(grupos, g), 3),
        "% inmuebles con igual grupo": None,
        "Silueta (en su espacio)": round(silhouette_score(Z, g), 3),
    })
    # concordancia exacta tras alinear etiquetas por precio
    ordg = df.assign(g=g).groupby("g")["Precio_arriendo"].mean().sort_values().index
    g2 = pd.Series(g).map({v: n for n, v in enumerate(ordg)}).values
    comparacion[-1]["% inmuebles con igual grupo"] = round((g2 == grupos).mean() * 100, 1)
    if "2 componentes" in nombre:
        g_pca2 = g2

print(pd.DataFrame(comparacion).to_string(index=False))
print("\nMatriz de contingencia (filas = A preprocesado, columnas = B sobre PCA 2D):")
print(pd.crosstab(grupos, g_pca2, rownames=["A"], colnames=["B"]))

fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for a, etiquetas, titulo in [(ax[0], grupos, "A: K-Means sobre variables preprocesadas"),
                             (ax[1], g_pca2, "B: K-Means sobre PC1–PC2")]:
    for g in range(3):
        s = etiquetas == g
        a.scatter(X_pca[s, 0], X_pca[s, 1], s=12, alpha=.7, color=COLORES[g], label=f"Grupo {g}")
    a.set(title=titulo, xlabel="PC1")
    a.grid(alpha=.3)
    a.legend()
ax[0].set_ylabel("PC2")
mostrar()

# %%
# 9. Experimento C — solo variables informativas (Precio, Área, Habitaciones, Estrato)
# En el Experimento A, Banos y Antiguedad_anios casi no se correlacionan con el resto (|r| < 0.07) y
# acaparan PC2. Aquí se repite PCA + K-Means sin ellas para ver cuánto cambia la segmentación.

# %%
nucleo = ["Precio_arriendo", "Area_m2", "Habitaciones", "Estrato"]
X_nuc = StandardScaler().fit_transform(df[nucleo])
pca_nuc = PCA(n_components=2).fit(X_nuc)
Z_nuc = pca_nuc.transform(X_nuc)
print("Varianza explicada (4 variables) PC1, PC2:", pca_nuc.explained_variance_ratio_.round(4),
      "| acumulada:", round(pca_nuc.explained_variance_ratio_.sum(), 4))
print(pd.DataFrame(pca_nuc.components_.T, index=nucleo, columns=["PC1", "PC2"]).round(3))

g_nuc = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_nuc)
ordn = df.assign(g=g_nuc).groupby("g")["Precio_arriendo"].mean().sort_values().index
g_nuc = pd.Series(g_nuc).map({v: n for n, v in enumerate(ordn)}).values
df["Grupo_nucleo"] = g_nuc
print("Silueta (4 variables):", round(silhouette_score(X_nuc, g_nuc), 3))
print("ARI entre Experimento A y C:", round(adjusted_rand_score(grupos, g_nuc), 3))
print(df.groupby("Grupo_nucleo")[nucleo].mean().round(2).assign(inmuebles=df.groupby("Grupo_nucleo").size()))

plt.figure(figsize=(8, 5.5))
for g, nom in zip(range(3), ["Económico", "Intermedio", "Premium"]):
    s = g_nuc == g
    plt.scatter(Z_nuc[s, 0], Z_nuc[s, 1], s=14, alpha=.7, color=COLORES[g], label=f"Grupo {g} · {nom}")
plt.xlabel(f"PC1 ({pca_nuc.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca_nuc.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("K-Means + PCA con variables informativas")
plt.legend()
plt.grid(alpha=.3)
mostrar()

# %%
# 10. Pipeline reutilizable (preprocesamiento → PCA → K-Means)

# %%
pipeline = Pipeline([
    ("prep", preprocesamiento),
    ("pca", PCA(n_components=n90)),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10)),
])
etiquetas_pipe = pipeline.fit_predict(X)
print("Pipeline ajustado. Inmuebles por grupo:", np.bincount(etiquetas_pipe))

