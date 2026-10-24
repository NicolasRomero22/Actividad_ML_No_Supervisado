# %%
# Laboratorio 1 — Segmentación de inmuebles en arriendo con K-Means
# Universidad de Cundinamarca · ESP. Inteligencia Artificial · CADI – Introducción a Machine Learning · 2026-2
#
# %%
# ============================================================
# 0. Librerías y utilidades
# ============================================================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

NOMBRE_EXCEL = "DATASET_Inmuebles.xlsx"

# Formato de pesos colombianos para los ejes
fmt_cop = lambda x, _=None: f"${x/1e6:.1f}M"


def localizar_dataset(nombre=NOMBRE_EXCEL):
    """Busca el Excel junto al script o en el directorio actual."""
    try:
        base = os.path.dirname(os.path.abspath(__file__))   # PyScripter / consola
    except NameError:
        base = os.getcwd()
    candidatos = [os.path.join(base, nombre), os.path.join(os.getcwd(), nombre)]
    for ruta in candidatos:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(f"No se encontró {nombre}. Colócalo en la misma carpeta que este script.")


def mostrar():
    """Muestra la gráfica en pantalla (no se guarda ningún archivo)."""
    plt.tight_layout()
    plt.show()


# %%
# 1. Carga y revisión del dataset
# Se estandarizan los nombres de columnas al formato del enunciado. La columna ID_inmueble se conserva
# para identificar los registros, pero no se usa para calcular distancias.

# %%
ruta = localizar_dataset()
df = pd.read_excel(ruta)

print("Columnas originales:", list(df.columns))

# Renombrar al formato del enunciado. 'Estrato.1' es una segunda columna 'Estrato'
# duplicada en el archivo (valores 1-4, sin relación con las demás variables): se descarta.
df = df.rename(columns={
    "Inmueble": "ID_inmueble", "Área m²": "Area_m2", "Precio arriendo": "Precio_arriendo",
    "Baños": "Banos", "Antig[uedad": "Antiguedad_anios", "Tipo Inmueble": "Tipo_inmueble",
})
df = df.drop(columns=["Estrato.1"])

print("\nDimensiones:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nValores nulos:", int(df.isna().sum().sum()), "| Duplicados:", int(df.duplicated().sum()))
print("\nEstadísticas descriptivas:")
print(df.describe().round(2))

# %%
# 2. Exploración: relación entre las variables
# El precio de arriendo debería moverse junto con el área, las habitaciones y el estrato.

# %%
variables = ["Area_m2", "Habitaciones", "Estrato", "Precio_arriendo"]   # las 4 del ejemplo del docente
print("Matriz de correlación:")
print(df[variables + ["Banos", "Antiguedad_anios"]].corr().round(2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["Precio_arriendo"], bins=30, color="#3b6fb6", edgecolor="white")
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(fmt_cop))
ax[0].set(title="Distribución del precio de arriendo", xlabel="Precio de arriendo", ylabel="Inmuebles")
ax[1].scatter(df["Area_m2"], df["Precio_arriendo"], s=10, alpha=.5, color="#3b6fb6")
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(fmt_cop))
ax[1].set(title="Área vs. precio", xlabel="Área (m²)", ylabel="Precio de arriendo")
mostrar()

# %%
# 3. Escalamiento
# K-Means usa distancias euclidianas; el precio (millones) dominaría sobre habitaciones (1–5) si no se escala.

# %%
X = df[variables]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Media tras escalar:", X_scaled.mean(axis=0).round(3), "| Desv.:", X_scaled.std(axis=0).round(3))

# %%
# 4. ¿Cuántos clusters? Método del codo y coeficiente de silueta (K = 2 … 8)

# %%
resultados = []
for k in range(2, 9):
    m = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    resultados.append({
        "K": k,
        "Inercia": m.inertia_,
        "Silueta": silhouette_score(X_scaled, m.labels_),
        "Davies-Bouldin": davies_bouldin_score(X_scaled, m.labels_),
        "Calinski-Harabasz": calinski_harabasz_score(X_scaled, m.labels_),
    })
metricas = pd.DataFrame(resultados).set_index("K")
print(metricas.round(3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(metricas.index, metricas["Inercia"], "o-", color="#3b6fb6")
ax[0].set(title="Método del codo", xlabel="Número de clusters (K)", ylabel="Inercia (WCSS)")
ax[1].plot(metricas.index, metricas["Silueta"], "o-", color="#d1495b")
ax[1].set(title="Coeficiente de silueta", xlabel="Número de clusters (K)", ylabel="Silueta promedio")
for a in ax:
    a.grid(alpha=.3)
mostrar()

# %%
# 5. K-Means con K = 3 (modelo del ejemplo) — Económico / Intermedio / Premium
# Se reordenan las etiquetas para que el cluster 0 sea siempre el de menor precio promedio.

# %%
def ajustar_kmeans(k):
    """Ajusta K-Means, ordena los clusters por precio promedio y devuelve (modelo, etiquetas, centroides)."""
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(X_scaled)
    centros = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=variables)
    orden = centros["Precio_arriendo"].argsort().values          # de menor a mayor precio
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
    etiquetas = pd.Series(modelo.labels_).map(mapa).values
    centros = centros.iloc[orden].reset_index(drop=True)
    return modelo, etiquetas, centros


modelo3, df["cluster_k3"], centros3 = ajustar_kmeans(3)

print("Centroides en escala original (K=3):")
print(centros3.round(1))

resumen3 = df.groupby("cluster_k3").agg(
    inmuebles=("ID_inmueble", "count"),
    area_m2=("Area_m2", "mean"), habitaciones=("Habitaciones", "mean"),
    estrato=("Estrato", "mean"), precio_arriendo=("Precio_arriendo", "mean"),
).round(2)
resumen3["% portafolio"] = (resumen3["inmuebles"] / len(df) * 100).round(1)
print("\nResumen por cluster (K=3):")
print(resumen3)
print("\nSilueta K=3:", round(silhouette_score(X_scaled, df["cluster_k3"]), 3))

# %%
# 6. Comparación K = 2, K = 3 y K = 4 (actividad 3.a y 3.b)

# %%
etiquetas_k, centros_k = {}, {}
for k in (2, 3, 4):
    _, etiquetas_k[k], centros_k[k] = ajustar_kmeans(k)
    df[f"cluster_k{k}"] = etiquetas_k[k]

colores = ["#3b6fb6", "#f2a541", "#d1495b", "#2a9d8f"]
fig, ejes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, k in zip(ejes, (2, 3, 4)):
    for c in range(k):
        s = etiquetas_k[k] == c
        ax.scatter(df.loc[s, "Area_m2"], df.loc[s, "Precio_arriendo"], s=12, alpha=.6,
                   color=colores[c], label=f"Cluster {c}")
    ax.scatter(centros_k[k]["Area_m2"], centros_k[k]["Precio_arriendo"], marker="X", s=180,
               color="black", edgecolor="white", label="Centroides")
    sil = silhouette_score(X_scaled, etiquetas_k[k])
    ax.set(title=f"K-Means con K = {k} (silueta = {sil:.3f})", xlabel="Área del inmueble (m²)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
ejes[0].set_ylabel("Precio de arriendo")
ejes[0].yaxis.set_major_formatter(plt.FuncFormatter(fmt_cop))
mostrar()

for k in (2, 3, 4):
    print(f"\n===== Perfil promedio por cluster — K = {k} =====")
    perfil = df.groupby(f"cluster_k{k}").agg(
        inmuebles=("ID_inmueble", "count"), area_m2=("Area_m2", "mean"),
        habitaciones=("Habitaciones", "mean"), estrato=("Estrato", "mean"),
        precio_arriendo=("Precio_arriendo", "mean")).round(2)
    print(perfil)

# %%
# 7. Perfil por localidad y tipo dentro de cada segmento (K = 3)
# Las variables categóricas no se usaron para agrupar; sirven para interpretar los segmentos.

# %%
print("Distribución por localidad (proporción por cluster):")
print(pd.crosstab(df["cluster_k3"], df["Localidad"], normalize="index").round(2))

nombres = {0: "Económico", 1: "Intermedio", 2: "Premium"}
fig, ax = plt.subplots(figsize=(7, 4))
df.boxplot(column="Precio_arriendo", by="cluster_k3", ax=ax, grid=False)
ax.set_xticklabels([f"{c} · {nombres[c]}" for c in range(3)])
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_cop))
ax.set(title="Precio de arriendo por segmento (K=3)", xlabel="", ylabel="Precio de arriendo")
plt.suptitle("")
mostrar()

# %%
# 8. Conclusión

# %%

mejor_k = metricas["Silueta"].idxmax()
print(f"\nMayor silueta: K = {mejor_k} ({metricas.loc[mejor_k, 'Silueta']:.3f}). "
      "Para la inmobiliaria, K = 3 equilibra interpretabilidad comercial (Económico / Intermedio / Premium) "
      "y calidad de separación; K = 2 es más simple pero mezcla segmentos; K = 4 fragmenta el portafolio.")
