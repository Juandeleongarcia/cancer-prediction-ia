"""
_build_notebook.py
Genera analysis.ipynb completo para el proyecto Cancer Prediction.
Ejecutar: python _build_notebook.py
"""
import json, uuid
from pathlib import Path

def cid():
    return str(uuid.uuid4())[:8]

def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": src.strip("\n"),
    }

def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": src.strip("\n"),
    }

cells = []

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 0: TÍTULO
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"""# Predicción de Diagnóstico de Cáncer — Estudio Comparativo ML vs MLP

**Universidad Alfonso X el Sabio**
Asignatura: Bases de Datos e Inteligencia Artificial · Curso 2025-2026

---

| Sección | Contenido |
|---|---|
| 1 | Configuración e importaciones |
| 2 | Carga y auditoría de los 6 CSV |
| 3 | Fusión de colecciones por `paciente_id` |
| 4 | Validación del dataset unido |
| 5 | Persistencia del dataset procesado |
| 6–13 | Análisis Exploratorio de Datos (EDA) |
| 14–18 | Preprocessing y pipeline sklearn |
| 19–25 | Modelos ML clásicos (LR · RF · XGB · LGB · CB) |
| 26–30 | Red neuronal MLP |
| 31–34 | Comparativa final, ranking y métricas |"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 1: SECCIÓN 1 — IMPORTS Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 1. Configuración e importaciones"))

cells.append(code(
r"""import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd

# matplotlib DEBE usar backend no-interactivo ANTES de cualquier import de pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks as tf_callbacks

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(42)
tf.random.set_seed(42)

# Estilo global de gráficas
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi":     120,
    "savefig.bbox":   "tight",
    "savefig.dpi":    150,
})

# ── Rutas del proyecto ────────────────────────────────────────────────────────
PROJECT_ROOT  = Path().resolve()
RAW_DIR       = PROJECT_ROOT / "base de datos"
PROCESSED_DIR = PROJECT_ROOT / "data"    / "processed"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR   = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"

for _d in [PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH   = PROCESSED_DIR / "cancer_merged.csv"
PIPELINE_PATH = PROCESSED_DIR / "preprocess_pipeline.pkl"
COL_CFG_PATH  = PROCESSED_DIR / "column_config.json"

CSV_FILES = {
    "bioquimicos":       "CASOCANCER_01_BIOQUIMICOS.csv",
    "clinicos":          "CASOCANCER_02_CLINICOS.csv",
    "geneticos":         "CASOCANCER_03_GENETICOS.csv",
    "economicos":        "CASOCANCER_04_ECONOMICOS.csv",
    "generales":         "CASOCANCER_05_GENERALES.csv",
    "sociodemograficos": "CASOCANCER_06_SOCIODEMOGRAFICOS.csv",
}

print(f"Proyecto  : {PROJECT_ROOT}")
print(f"Datos raw : {RAW_DIR}")
print(f"Figuras   : {FIGURES_DIR}")
print(f"Métricas  : {METRICS_DIR}")
print(f"Modelos   : {MODELS_DIR}")
print("Imports OK ✓")"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 3: SECCIÓN 2 — CARGA Y AUDITORÍA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 2. Carga y auditoría individual de cada CSV"))

cells.append(code(
r"""SEPARATOR = "=" * 70

def audit_dataframe(name: str, df: pd.DataFrame) -> None:
    # Imprime el informe de auditoría de un DataFrame.
    print(SEPARATOR)
    print(f"  COLECCIÓN: {name.upper()}")
    print(SEPARATOR)
    print(f"\n[Dimensiones] {df.shape[0]:,} filas × {df.shape[1]} columnas")

    dtypes_info = df.dtypes.reset_index()
    dtypes_info.columns = ["columna", "tipo"]
    print("\n[Columnas y tipos de datos]")
    print(dtypes_info.to_string(index=False))

    nulls = df.isnull().sum()
    print("\n[Valores nulos por columna]")
    if nulls.sum() == 0:
        print("  Sin valores nulos.")
    else:
        print(nulls[nulls > 0].to_string())

    if "paciente_id" in df.columns:
        n_dup = df["paciente_id"].duplicated().sum()
        print(f"\n[Duplicados en paciente_id] {n_dup}")
    else:
        print("\n[ADVERTENCIA] 'paciente_id' NO está presente.")

    if "cancer" in df.columns:
        vc   = df["cancer"].value_counts()
        prev = df["cancer"].mean() * 100
        print(f"\n[Variable objetivo 'cancer'] Dist: {vc.to_dict()}  Prevalencia: {prev:.2f}%")

    print("\n[Primeras 3 filas]")
    print(df.head(3).to_string())
    print()"""
))

cells.append(code(
r"""# Carga y auditoría de cada CSV
dataframes = {}
for key, filename in CSV_FILES.items():
    filepath = RAW_DIR / filename
    if not filepath.exists():
        print(f"[FICHERO NO ENCONTRADO] {filepath}\n  → Colección omitida.\n")
        continue
    df = pd.read_csv(filepath)
    dataframes[key] = df
    audit_dataframe(key, df)

print(f"\n{len(dataframes)} colecciones cargadas correctamente.")"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 6: SECCIÓN 3 — FUSIÓN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 3. Fusión de colecciones por `paciente_id`\n\nInner join secuencial sobre la clave `paciente_id`."))

cells.append(code(
r"""if not dataframes:
    raise RuntimeError("No se cargó ningún DataFrame. Revisa las rutas de los CSV.")

keys_loaded = list(dataframes.keys())
df_merged   = dataframes[keys_loaded[0]].copy()
print(f"Base de fusión: '{keys_loaded[0]}' — {df_merged.shape[0]:,} filas")

for key in keys_loaded[1:]:
    n_antes   = df_merged.shape[0]
    df_merged = df_merged.merge(dataframes[key], on="paciente_id", how="inner")
    n_despues = df_merged.shape[0]
    perdidas  = n_antes - n_despues
    status    = f"  [{key}] {n_antes:,} → {n_despues:,} filas"
    if perdidas > 0:
        status += f"  ⚠ {perdidas:,} filas perdidas"
    print(status)

print(f"\nDataset unido final: {df_merged.shape[0]:,} filas × {df_merged.shape[1]} columnas")"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 8: SECCIÓN 4 — VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 4. Validación del dataset unido"))

cells.append(code(
r"""print("=" * 70)
print("  VALIDACIÓN DEL DATASET UNIDO")
print("=" * 70)
print(f"\n[Dimensiones] {df_merged.shape[0]:,} filas × {df_merged.shape[1]} columnas")

nulls_total = df_merged.isnull().sum()
print("\n[Valores nulos]")
if nulls_total.sum() == 0:
    print("  Sin valores nulos.")
else:
    print(nulls_total[nulls_total > 0].to_string())

n_dup_final = df_merged["paciente_id"].duplicated().sum()
print(f"\n[Duplicados en paciente_id] {n_dup_final}")

if "cancer" in df_merged.columns:
    vc   = df_merged["cancer"].value_counts()
    prev = df_merged["cancer"].mean() * 100
    print(f"\n[Variable objetivo 'cancer'] PRESENTE ✓")
    print(f"  Distribución : {vc.to_dict()}")
    print(f"  Prevalencia  : {prev:.2f}%")
else:
    print("\n[ERROR] 'cancer' NO está en el dataset unido.")

print("\n[Estadísticas descriptivas — numéricas]")
print(df_merged.describe().round(3).to_string())"""
))

cells.append(code(
r"""# Resumen de variables categóricas
cat_cols_full = df_merged.select_dtypes(include=["object"]).columns.tolist()
cat_features_full = [c for c in cat_cols_full if c != "paciente_id"]

if cat_features_full:
    print("[Variables categóricas]")
    for col in cat_features_full:
        vc = df_merged[col].value_counts()
        print(f"\n  {col} — {df_merged[col].nunique()} categorías:")
        print(vc.to_string())
else:
    print("No hay columnas object adicionales.")"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 11: SECCIÓN 5 — GUARDADO
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 5. Persistencia del dataset procesado"))

cells.append(code(
r"""df_merged.to_csv(OUTPUT_PATH, index=False)
df_check = pd.read_csv(OUTPUT_PATH)
assert df_check.shape == df_merged.shape, "Error: dimensiones distintas tras guardar."

print(f"Dataset guardado correctamente:")
print(f"  Ruta    : {OUTPUT_PATH}")
print(f"  Tamaño  : {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
print(f"  Filas   : {df_check.shape[0]:,}")
print(f"  Columnas: {df_check.shape[1]}")"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDAS EDA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"""---
## 6–13. Análisis Exploratorio de Datos (EDA)

Distribuciones, correlaciones y detección de variables con riesgo de data leakage."""
))

# fig_01: Target distribution
cells.append(code(
r"""# ── fig_01: Distribución de la variable objetivo ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

counts = df_merged["cancer"].value_counts().sort_index()
labels = ["Sin cáncer (0)", "Con cáncer (1)"]
colors = ["#4C8BDA", "#E05C4B"]

axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
for bar, n in zip(axes[0].patches, counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
        f"{n:,}", ha="center", fontsize=11, fontweight="bold"
    )
axes[0].set_title("Distribución de la variable objetivo", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Pacientes")
axes[0].grid(axis="y", lw=0.4, alpha=0.5)

axes[1].pie(
    counts.values, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=90, pctdistance=0.75,
    textprops={"fontsize": 11},
)
axes[1].set_title("Proporción de clases", fontsize=13, fontweight="bold")

prev = df_merged["cancer"].mean() * 100
fig.suptitle(
    f"Prevalencia de cáncer: {prev:.2f}%  |  Desbalance ≈ {(100-prev)/prev:.1f}:1",
    fontsize=11, y=1.02,
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_01_target_distribution.png")
plt.close(fig)
print("fig_01_target_distribution.png guardada")"""
))

# fig_02: Nulls heatmap
cells.append(code(
r"""# ── fig_02: Mapa de valores nulos por colección ──────────────────────────────
collections_cols = {
    "bioquimicos":       list(dataframes["bioquimicos"].columns),
    "clinicos":          list(dataframes["clinicos"].columns),
    "geneticos":         list(dataframes["geneticos"].columns),
    "economicos":        list(dataframes["economicos"].columns),
    "generales":         list(dataframes["generales"].columns),
    "sociodemograficos": list(dataframes["sociodemograficos"].columns),
}

null_dict = {}
for coll_name, cols in collections_cols.items():
    cols_in = [c for c in cols if c in df_merged.columns and c != "paciente_id"]
    null_dict[coll_name] = df_merged[cols_in].isnull().sum().to_dict()

null_df = pd.DataFrame(null_dict).T.fillna(0)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    null_df, annot=True, fmt=".0f", cmap="YlOrRd",
    linewidths=0.5, ax=ax, annot_kws={"size": 8},
)
ax.set_title("Valores nulos por colección y variable", fontsize=13, fontweight="bold")
ax.set_ylabel("Colección")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_02_nulls_by_collection.png")
plt.close(fig)
print("fig_02_nulls_by_collection.png guardada")"""
))

# fig_03: Biochemical distributions
cells.append(code(
r"""# ── fig_03: Distribuciones de variables bioquímicas ──────────────────────────
bioq_cols = ["glucosa", "colesterol", "trigliceridos", "hemoglobina",
             "leucocitos", "plaquetas", "creatinina"]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(bioq_cols):
    axes[i].hist(df_merged[col], bins=50, color="#4C8BDA", edgecolor="white", alpha=0.8)
    axes[i].axvline(
        df_merged[col].mean(), color="#E05C4B", lw=1.5, linestyle="--",
        label=f"μ={df_merged[col].mean():.1f}",
    )
    axes[i].set_title(col.replace("_", " ").title(), fontsize=11)
    axes[i].legend(fontsize=8)
    axes[i].grid(axis="y", lw=0.4, alpha=0.5)

axes[-1].set_visible(False)
fig.suptitle("Distribuciones de variables bioquímicas", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_03_bioq_distributions.png")
plt.close(fig)
print("fig_03_bioq_distributions.png guardada")"""
))

# fig_04: Continuous by cancer
cells.append(code(
r"""# ── fig_04: Variables continuas por diagnóstico ──────────────────────────────
num_cols_eda = ["glucosa", "colesterol", "trigliceridos", "hemoglobina",
                "leucocitos", "plaquetas", "creatinina", "edad",
                "distancia_hospital_km", "num_hijos"]

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols_eda):
    for val, color, label in [(0, "#4C8BDA", "Sano"), (1, "#E05C4B", "Cáncer")]:
        subset = df_merged[df_merged["cancer"] == val][col]
        axes[i].hist(subset, bins=40, color=color, alpha=0.6,
                     label=label, density=True, edgecolor="white")
    axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
    axes[i].legend(fontsize=8, framealpha=0.7)
    axes[i].grid(axis="y", lw=0.4, alpha=0.5)

fig.suptitle("Variables continuas por diagnóstico de cáncer", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_04_continuous_by_cancer.png")
plt.close(fig)
print("fig_04_continuous_by_cancer.png guardada")"""
))

# fig_05: Binary by cancer
cells.append(code(
r"""# ── fig_05: Variables binarias vs cáncer ─────────────────────────────────────
bin_cols_eda = ["fumador", "diabetes", "hipertension", "obesidad",
                "enfermedad_cardiaca", "asma", "epoc",
                "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
                "mut_PIK3CA", "mut_ALK", "mut_BRAF"]

rates = []
for col in bin_cols_eda:
    rate_0 = df_merged[df_merged[col] == 0]["cancer"].mean()
    rate_1 = df_merged[df_merged[col] == 1]["cancer"].mean()
    rates.append({"feature": col, "sin_factor": rate_0, "con_factor": rate_1})

rates_df = pd.DataFrame(rates).sort_values("con_factor", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = range(len(rates_df))
ax.barh([y - 0.2 for y in y_pos], rates_df["sin_factor"] * 100,
        height=0.4, color="#4C8BDA", label="Variable = 0", alpha=0.85)
ax.barh([y + 0.2 for y in y_pos], rates_df["con_factor"] * 100,
        height=0.4, color="#E05C4B", label="Variable = 1", alpha=0.85)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(rates_df["feature"].tolist(), fontsize=10)
ax.set_xlabel("Tasa de cáncer (%)")
ax.set_title("Prevalencia de cáncer por variable binaria", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="x", lw=0.4, alpha=0.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_05_binary_by_cancer.png")
plt.close(fig)
print("fig_05_binary_by_cancer.png guardada")"""
))

# fig_06: Categorical by cancer
cells.append(code(
r"""# ── fig_06: Variables categóricas vs cáncer ──────────────────────────────────
cat_cols_eda = ["actividad_fisica", "nivel_educativo", "nivel_ingresos",
                "zona", "estado_civil"]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, col in enumerate(cat_cols_eda):
    grp = df_merged.groupby(col)["cancer"].mean().reset_index()
    grp = grp.sort_values("cancer", ascending=False)
    axes[i].bar(grp[col], grp["cancer"] * 100,
                color="#5B9BD5", edgecolor="white", alpha=0.9)
    axes[i].set_title(col.replace("_", " ").title(), fontsize=10)
    if i == 0:
        axes[i].set_ylabel("Tasa de cáncer (%)")
    axes[i].set_xticklabels(grp[col].tolist(), rotation=30, ha="right", fontsize=8)
    axes[i].grid(axis="y", lw=0.4, alpha=0.5)

fig.suptitle("Tasa de cáncer por categoría", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_06_categorical_by_cancer.png")
plt.close(fig)
print("fig_06_categorical_by_cancer.png guardada")"""
))

# fig_07: Correlation matrix
cells.append(code(
r"""# ── fig_07: Matriz de correlación ────────────────────────────────────────────
corr_cols = [
    "glucosa", "colesterol", "trigliceridos", "hemoglobina",
    "leucocitos", "plaquetas", "creatinina",
    "edad", "distancia_hospital_km", "num_hijos",
    "fumador", "diabetes", "hipertension", "obesidad",
    "enfermedad_cardiaca", "asma", "epoc",
    "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
    "mut_PIK3CA", "mut_ALK", "mut_BRAF", "cancer",
]

corr_matrix = df_merged[corr_cols].corr()

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, linewidths=0.3,
    annot_kws={"size": 6}, ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title(
    "Matriz de correlación — variables predictoras y target",
    fontsize=14, fontweight="bold", pad=15,
)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_07_correlation_matrix.png")
plt.close(fig)
print("fig_07_correlation_matrix.png guardada")"""
))

# fig_08: Leakage detection
cells.append(code(
r"""# ── fig_08: Detección de variables con riesgo de data leakage ────────────────
leakage_candidates = ["coste_total", "coste_farmaco", "num_ingresos",
                       "dias_hospital", "vive"]

df_leakage = df_merged[leakage_candidates + ["cancer"]].copy()

# Normalizar coma decimal en columnas de coste
for col in ["coste_total", "coste_farmaco"]:
    df_leakage[col] = (
        df_leakage[col].astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

corr_leakage = df_leakage.corr()["cancer"].drop("cancer")

fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = ["#E05C4B" if abs(v) > 0.15 else "#4C8BDA" for v in corr_leakage.values]
bars = ax.barh(corr_leakage.index.tolist(), corr_leakage.values,
               color=bar_colors, alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.axvline( 0.15, color="red", lw=1, linestyle="--", label="|r|>0.15 → riesgo leakage")
ax.axvline(-0.15, color="red", lw=1, linestyle="--")

for bar, val in zip(bars, corr_leakage.values):
    offset = 0.005 if val >= 0 else -0.005
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9)

ax.set_xlabel("Correlación de Pearson con 'cancer'")
ax.set_title("Variables excluidas — riesgo de data leakage", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="x", lw=0.4, alpha=0.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_08_leakage_correlation.png")
plt.close(fig)

print("fig_08_leakage_correlation.png guardada")
print("\nCorrelaciones con target:")
print(corr_leakage.sort_values(ascending=False).to_string())"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDAS PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"""---
## 14–18. Preprocessing y Pipeline sklearn

**Protocolo anti-data-leakage**: pipeline ajustado SOLO sobre train; umbral optimizado en validación; test evaluado UNA SOLA VEZ."""
))

# Feature selection
cells.append(code(
r"""# ── Definición de features ────────────────────────────────────────────────────
# Variables excluidas por leakage o nula varianza
LEAKAGE_VARS = [
    "coste_total", "coste_farmaco", "num_ingresos",
    "dias_hospital", "vive", "alcohol", "tipo_seguro",
]

TARGET = "cancer"
ID_COL = "paciente_id"

# Tipos de feature
NUM_COLS = [
    "glucosa", "colesterol", "trigliceridos", "hemoglobina",
    "leucocitos", "plaquetas", "creatinina",
    "edad", "distancia_hospital_km", "num_hijos",
]
BIN_COLS = [
    "fumador",
    "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
    "mut_PIK3CA", "mut_ALK", "mut_BRAF",
    "diabetes", "hipertension", "obesidad",
    "enfermedad_cardiaca", "asma", "epoc",
]
ORD_COLS = ["actividad_fisica"]
ORD_CATS = [["Baja", "Moderada", "Alta"]]
CAT_COLS = ["nivel_educativo", "nivel_ingresos", "zona", "estado_civil"]

ALL_FEATURES = NUM_COLS + BIN_COLS + ORD_COLS + CAT_COLS

# Verificar existencia
missing_f = [f for f in ALL_FEATURES if f not in df_merged.columns]
assert len(missing_f) == 0, f"Features no encontradas: {missing_f}"

print(f"Features activas: {len(ALL_FEATURES)}")
print(f"  Numéricas  ({len(NUM_COLS)}) : {NUM_COLS}")
print(f"  Binarias   ({len(BIN_COLS)}) : {BIN_COLS}")
print(f"  Ordinales  ({len(ORD_COLS)}) : {ORD_COLS}")
print(f"  Nominales  ({len(CAT_COLS)}) : {CAT_COLS}")
print(f"\nVariables excluidas (leakage/constante): {LEAKAGE_VARS}")"""
))

# Train/val/test split
cells.append(code(
r"""# ── Split estratificado 64 / 16 / 20 ─────────────────────────────────────────
X = df_merged[ALL_FEATURES].copy()
y = df_merged[TARGET].copy()

# Asegurar tipos numéricos en columnas numéricas
for c in NUM_COLS:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# 80 / 20 → train+val vs test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# 80 / 20 del trainval → train vs val  (= 64 / 16 del total)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
)

print(f"Train : {X_train.shape[0]:,} ({X_train.shape[0]/len(X)*100:.1f}%)  cáncer={y_train.mean()*100:.2f}%")
print(f"Val   : {X_val.shape[0]:,} ({X_val.shape[0]/len(X)*100:.1f}%)  cáncer={y_val.mean()*100:.2f}%")
print(f"Test  : {X_test.shape[0]:,} ({X_test.shape[0]/len(X)*100:.1f}%)  cáncer={y_test.mean()*100:.2f}%")"""
))

# Pipeline construction + fit
cells.append(code(
r"""# ── Construcción del ColumnTransformer ───────────────────────────────────────
numeric_pipe = Pipeline([("scaler", StandardScaler())])
binary_pipe  = Pipeline([("pass",   "passthrough")])
ordinal_pipe = Pipeline([
    ("enc", OrdinalEncoder(
        categories=ORD_CATS,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ))
])
nominal_pipe = Pipeline([
    ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, NUM_COLS),
        ("bin", binary_pipe,  BIN_COLS),
        ("ord", ordinal_pipe, ORD_COLS),
        ("cat", nominal_pipe, CAT_COLS),
    ],
    remainder="drop",
)

# Ajustar SOLO sobre train — protocolo anti-leakage
preprocessor.fit(X_train)

X_train_proc = preprocessor.transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

# Class weights calculados sobre train únicamente
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
n_total = len(y_train)
class_weight_dict = {
    0: n_total / (2 * n_neg),
    1: n_total / (2 * n_pos),
}

print(f"Dimensiones post-preprocessing:")
print(f"  Train : {X_train_proc.shape}")
print(f"  Val   : {X_val_proc.shape}")
print(f"  Test  : {X_test_proc.shape}")
print(f"Class weights : {class_weight_dict}")"""
))

# Save pipeline + fig_09, fig_10, fig_11
cells.append(code(
r"""# ── Guardar pipeline y configuración de columnas ─────────────────────────────
joblib.dump(preprocessor, PIPELINE_PATH)

col_config = {
    "num_cols":       NUM_COLS,
    "bin_cols":       BIN_COLS,
    "ord_cols":       ORD_COLS,
    "ord_categories": ORD_CATS,
    "cat_cols":       CAT_COLS,
    "all_features":   ALL_FEATURES,
    "target":         TARGET,
    "id_col":         ID_COL,
}
with open(COL_CFG_PATH, "w", encoding="utf-8") as f:
    json.dump(col_config, f, ensure_ascii=False, indent=2)

print(f"Pipeline guardado : {PIPELINE_PATH}")
print(f"Column config     : {COL_CFG_PATH}")

# ── fig_09: Distribución de clases en los splits ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, y_split) in zip(axes, [("Train", y_train), ("Val", y_val), ("Test", y_test)]):
    counts_s = y_split.value_counts().sort_index()
    ax.bar(["Sano (0)", "Cáncer (1)"], counts_s.values,
           color=["#4C8BDA", "#E05C4B"], edgecolor="white", width=0.5)
    for bar, n in zip(ax.patches, counts_s.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{n:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"{name} ({len(y_split):,} muestras\n{y_split.mean()*100:.1f}% cáncer)",
                 fontsize=11)
    ax.set_ylabel("Pacientes" if name == "Train" else "")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
fig.suptitle("Distribución de clases en los tres splits", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_09_splits_distribution.png")
plt.close(fig)

# ── fig_10: Muestra de datos procesados ──────────────────────────────────────
rng = np.random.RandomState(42)
idx_sample = rng.choice(X_train_proc.shape[0], size=min(50, X_train_proc.shape[0]), replace=False)
sample_data = X_train_proc[idx_sample, :20]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(sample_data, cmap="RdBu_r", center=0, linewidths=0.2,
            ax=ax, cbar_kws={"shrink": 0.6})
ax.set_title("Muestra de datos procesados (primeras 20 features, 50 pacientes)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Feature (post-pipeline)")
ax.set_ylabel("Paciente (muestra)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_10_processed_sample.png")
plt.close(fig)

# ── fig_11: Pesos de clase ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Clase 0 (sano)", "Clase 1 (cáncer)"],
       [class_weight_dict[0], class_weight_dict[1]],
       color=["#4C8BDA", "#E05C4B"], edgecolor="white", width=0.5)
for bar, val in zip(ax.patches, [class_weight_dict[0], class_weight_dict[1]]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Pesos de clase para compensar el desbalance", fontsize=13, fontweight="bold")
ax.set_ylabel("Peso")
ax.grid(axis="y", lw=0.4, alpha=0.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_11_class_weights.png")
plt.close(fig)

print("fig_09, fig_10, fig_11 guardadas")"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDAS ML
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"""---
## 19–25. Modelos ML Clásicos

5 modelos entrenados con barrido de umbral sobre **validación**.
El **test set se evalúa una única vez** al final de esta sección."""
))

# Helper functions
cells.append(code(
r"""# ── Funciones auxiliares ──────────────────────────────────────────────────────
def sweep_threshold(model, X_proc, y_true, n_steps: int = 99):
    # Barrido de umbral en [0.01, 0.99]. Maximiza score compuesto en val.
    thresholds = np.linspace(0.01, 0.99, n_steps)
    proba = model.predict_proba(X_proc)[:, 1]
    auc_roc = roc_auc_score(y_true, proba)
    auc_pr  = average_precision_score(y_true, proba)

    best_score, best_thr = -1.0, 0.5
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        if preds.sum() == 0:
            continue
        r  = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        s  = 0.35 * r + 0.30 * f1 + 0.20 * auc_pr + 0.15 * auc_roc
        if s > best_score:
            best_score, best_thr = s, thr
    return float(best_thr)


def evaluate_on_test(model, X_proc, y_true, threshold: float, model_name: str) -> dict:
    # Evalúa modelo en test con el umbral óptimo. Devuelve dict de métricas.
    proba = model.predict_proba(X_proc)[:, 1]
    preds = (proba >= threshold).astype(int)
    r       = recall_score(y_true, preds, zero_division=0)
    p       = precision_score(y_true, preds, zero_division=0)
    f1      = f1_score(y_true, preds, zero_division=0)
    auc_roc = roc_auc_score(y_true, proba)
    auc_pr  = average_precision_score(y_true, proba)
    score   = 0.35 * r + 0.30 * f1 + 0.20 * auc_pr + 0.15 * auc_roc
    return {
        "Modelo":     model_name,
        "Recall":     r,
        "Precision":  p,
        "F1":         f1,
        "AUC-ROC":    auc_roc,
        "AUC-PR":     auc_pr,
        "Threshold":  threshold,
        "Score":      score,
        "proba_test": proba,
    }

# Diccionario acumulador de resultados ML
ml_results = []
model_objects = {}

print("Funciones auxiliares cargadas ✓")"""
))

# Logistic Regression
cells.append(code(
r"""# ── Logistic Regression ───────────────────────────────────────────────────────
print("Entrenando Logistic Regression…")
lr_model = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=42, C=1.0
)
lr_model.fit(X_train_proc, y_train)

lr_thr = sweep_threshold(lr_model, X_val_proc, y_val)
lr_res = evaluate_on_test(lr_model, X_test_proc, y_test, lr_thr, "Logistic Regression")
ml_results.append(lr_res)
model_objects["Logistic Regression"] = lr_model

joblib.dump(lr_model, MODELS_DIR / "ml_logistic_regression.pkl")
print(f"  Threshold={lr_thr:.2f}  Recall={lr_res['Recall']:.4f}  "
      f"F1={lr_res['F1']:.4f}  AUC-ROC={lr_res['AUC-ROC']:.4f}")"""
))

# Random Forest
cells.append(code(
r"""# ── Random Forest ────────────────────────────────────────────────────────────
print("Entrenando Random Forest…")
rf_model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    max_depth=12, min_samples_leaf=5,
    random_state=42, n_jobs=-1,
)
rf_model.fit(X_train_proc, y_train)

rf_thr = sweep_threshold(rf_model, X_val_proc, y_val)
rf_res = evaluate_on_test(rf_model, X_test_proc, y_test, rf_thr, "Random Forest")
ml_results.append(rf_res)
model_objects["Random Forest"] = rf_model

joblib.dump(rf_model, MODELS_DIR / "ml_random_forest.pkl")
print(f"  Threshold={rf_thr:.2f}  Recall={rf_res['Recall']:.4f}  "
      f"F1={rf_res['F1']:.4f}  AUC-ROC={rf_res['AUC-ROC']:.4f}")"""
))

# XGBoost
cells.append(code(
r"""# ── XGBoost ──────────────────────────────────────────────────────────────────
print("Entrenando XGBoost…")
scale_pos = int((y_train == 0).sum()) / int((y_train == 1).sum())

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos,
    random_state=42, verbosity=0,
    eval_metric="logloss",
)
xgb_model.fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
    verbose=False,
)

xgb_thr = sweep_threshold(xgb_model, X_val_proc, y_val)
xgb_res = evaluate_on_test(xgb_model, X_test_proc, y_test, xgb_thr, "XGBoost")
ml_results.append(xgb_res)
model_objects["XGBoost"] = xgb_model

joblib.dump(xgb_model, MODELS_DIR / "ml_xgboost.pkl")
print(f"  Threshold={xgb_thr:.2f}  Recall={xgb_res['Recall']:.4f}  "
      f"F1={xgb_res['F1']:.4f}  AUC-ROC={xgb_res['AUC-ROC']:.4f}")"""
))

# LightGBM
cells.append(code(
r"""# ── LightGBM ─────────────────────────────────────────────────────────────────
print("Entrenando LightGBM…")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.05,
    class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1,
)
lgb_model.fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=-1),
    ],
)

lgb_thr = sweep_threshold(lgb_model, X_val_proc, y_val)
lgb_res = evaluate_on_test(lgb_model, X_test_proc, y_test, lgb_thr, "LightGBM")
ml_results.append(lgb_res)
model_objects["LightGBM"] = lgb_model

joblib.dump(lgb_model, MODELS_DIR / "ml_lightgbm.pkl")
print(f"  Threshold={lgb_thr:.2f}  Recall={lgb_res['Recall']:.4f}  "
      f"F1={lgb_res['F1']:.4f}  AUC-ROC={lgb_res['AUC-ROC']:.4f}")"""
))

# CatBoost
cells.append(code(
r"""# ── CatBoost ──────────────────────────────────────────────────────────────────
print("Entrenando CatBoost…")
cb_model = CatBoostClassifier(
    iterations=300, depth=6, learning_rate=0.05,
    class_weights={0: class_weight_dict[0], 1: class_weight_dict[1]},
    random_seed=42, verbose=0, eval_metric="F1",
)
cb_model.fit(
    X_train_proc, y_train,
    eval_set=(X_val_proc, y_val),
    use_best_model=True,
    verbose=False,
)

cb_thr = sweep_threshold(cb_model, X_val_proc, y_val)
cb_res = evaluate_on_test(cb_model, X_test_proc, y_test, cb_thr, "CatBoost")
ml_results.append(cb_res)
model_objects["CatBoost"] = cb_model

joblib.dump(cb_model, MODELS_DIR / "ml_catboost.pkl")
print(f"  Threshold={cb_thr:.2f}  Recall={cb_res['Recall']:.4f}  "
      f"F1={cb_res['F1']:.4f}  AUC-ROC={cb_res['AUC-ROC']:.4f}")"""
))

# ML figures + save
cells.append(code(
r"""# ── Tabla de resultados ML ────────────────────────────────────────────────────
ml_df = pd.DataFrame([{k: v for k, v in r.items() if k != "proba_test"}
                       for r in ml_results])
ml_df = ml_df.sort_values("Score", ascending=False).reset_index(drop=True)

print("\n=== RESULTADOS ML (test set) ===")
print(ml_df[["Modelo", "Recall", "Precision", "F1",
             "AUC-ROC", "AUC-PR", "Threshold", "Score"]].to_string(index=False))

# Mejor modelo ML
best_ml_name = ml_df.iloc[0]["Modelo"]
best_ml_res  = next(r for r in ml_results if r["Modelo"] == best_ml_name)
joblib.dump(model_objects[best_ml_name], MODELS_DIR / "best_ml_model.pkl")
print(f"\nMejor modelo ML : {best_ml_name}  Score={ml_df.iloc[0]['Score']:.4f}")

# Guardar CSVs ML
ml_df.drop(columns=["Score"]).to_csv(METRICS_DIR / "ml_results.csv", index=False)
ml_thr_df = ml_df[["Modelo", "Threshold"]].rename(columns={"Threshold": "Threshold_optimo"})
ml_thr_df.to_csv(METRICS_DIR / "ml_thresholds.csv", index=False)

# ── fig_12: Barrido de umbral del mejor modelo ────────────────────────────────
best_model_obj   = model_objects[best_ml_name]
thresholds_sweep = np.linspace(0.01, 0.99, 99)
proba_val_best   = best_model_obj.predict_proba(X_val_proc)[:, 1]
auc_roc_v = roc_auc_score(y_val, proba_val_best)
auc_pr_v  = average_precision_score(y_val, proba_val_best)

recalls_sw, f1s_sw, scores_sw = [], [], []
for thr in thresholds_sweep:
    p_sw = (proba_val_best >= thr).astype(int)
    r_sw = recall_score(y_val, p_sw, zero_division=0)
    f_sw = f1_score(y_val, p_sw, zero_division=0)
    s_sw = 0.35 * r_sw + 0.30 * f_sw + 0.20 * auc_pr_v + 0.15 * auc_roc_v
    recalls_sw.append(r_sw); f1s_sw.append(f_sw); scores_sw.append(s_sw)

opt_idx = int(np.argmax(scores_sw))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds_sweep, recalls_sw, color="#4C8BDA", label="Recall", lw=2)
ax.plot(thresholds_sweep, f1s_sw,     color="#E05C4B", label="F1",     lw=2)
ax.plot(thresholds_sweep, scores_sw,  color="#27AE60", label="Score compuesto", lw=2, linestyle="--")
ax.axvline(thresholds_sweep[opt_idx], color="gray", lw=1.5, linestyle=":",
           label=f"Umbral óptimo = {thresholds_sweep[opt_idx]:.2f}")
ax.set_xlabel("Umbral de decisión"); ax.set_ylabel("Métrica")
ax.set_title(f"Barrido de umbral — {best_ml_name} (validación)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(lw=0.4, alpha=0.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_12_threshold_scan.png")
plt.close(fig)

# ── fig_13: Curvas ROC ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
colors_ml = ["#4C8BDA", "#E05C4B", "#27AE60", "#F39C12", "#9B59B6"]
for res, col in zip(ml_results, colors_ml):
    fpr, tpr, _ = roc_curve(y_test, res["proba_test"])
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f"{res['Modelo']} (AUC={res['AUC-ROC']:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curvas ROC — modelos ML (test set)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right"); ax.grid(lw=0.4, alpha=0.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_13_roc_curves.png")
plt.close(fig)

# ── fig_14: Curvas Precision-Recall ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for res, col in zip(ml_results, colors_ml):
    prec, rec, _ = precision_recall_curve(y_test, res["proba_test"])
    ax.plot(rec, prec, color=col, lw=2,
            label=f"{res['Modelo']} (AP={res['AUC-PR']:.4f})")
ax.axhline(y_test.mean(), color="gray", lw=1, linestyle="--",
           label=f"Baseline ({y_test.mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Curvas Precision-Recall — modelos ML (test set)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(lw=0.4, alpha=0.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_14_pr_curves.png")
plt.close(fig)

# ── fig_15: Matriz de confusión del mejor modelo ─────────────────────────────
preds_best_test = (best_ml_res["proba_test"] >= best_ml_res["Threshold"]).astype(int)
cm_best = confusion_matrix(y_test, preds_best_test)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_best,
                               display_labels=["Sano (0)", "Cáncer (1)"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Matriz de confusión — {best_ml_name}\n(umbral={best_ml_res['Threshold']:.2f})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_15_confusion_matrix_best.png")
plt.close(fig)

# ── fig_16: Comparativa de métricas ML ───────────────────────────────────────
metrics_cols = ["Recall", "Precision", "F1", "AUC-ROC", "AUC-PR"]
fig, axes = plt.subplots(1, len(metrics_cols), figsize=(16, 5))
for ax, metric in zip(axes, metrics_cols):
    vals   = ml_df[metric].values
    models = ml_df["Modelo"].values
    bars   = ax.barh(models, vals, color=colors_ml[:len(models)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel(metric); ax.set_xlim(0, 1.18)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    if ax is not axes[0]:
        ax.set_yticklabels([])

fig.suptitle("Comparativa de métricas — modelos ML (test set)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_16_metrics_comparison.png")
plt.close(fig)

print("Figuras ML (fig_12 – fig_16) guardadas ✓")"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDAS MLP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"""---
## 26–30. Red Neuronal MLP

Arquitectura `Dense(256→128→64)` + `BatchNormalization` + `Dropout`.
Regularización: `EarlyStopping`, `ReduceLROnPlateau`, `class_weight` balanceado."""
))

# fig_17: Architecture diagram
cells.append(code(
r"""# ── fig_17: Diagrama de arquitectura MLP ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

layer_defs = [
    (1.0, "Input\n(n_features)",        "#D5E8D4", "#82B366"),
    (3.0, "Dense(256)\nBatchNorm\nDropout(0.25)", "#DAE8FC", "#6C8EBF"),
    (5.5, "Dense(128)\nBatchNorm\nDropout(0.25)", "#DAE8FC", "#6C8EBF"),
    (8.0, "Dense(64)\nBatchNorm\nDropout(0.20)",  "#DAE8FC", "#6C8EBF"),
    (10.5, "Output(1)\nSigmoid",         "#FFE6CC", "#D6B656"),
]

for x, label, fc, ec in layer_defs:
    ax.add_patch(matplotlib.patches.FancyBboxPatch(
        (x - 0.75, 1.5), 1.5, 2.0,
        boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=1.8,
    ))
    ax.text(x, 2.5, label, ha="center", va="center",
            fontsize=9, fontweight="bold", multialignment="center")

for i in range(len(layer_defs) - 1):
    x1 = layer_defs[i][0]   + 0.75
    x2 = layer_defs[i+1][0] - 0.75
    ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

ax.text(5.75, 0.7,
        "Optimizador: Adam (lr=0.001)  |  Pérdida: binary_crossentropy  |  class_weight: balanced",
        ha="center", fontsize=9, color="gray", style="italic")
ax.set_title("Arquitectura de la Red Neuronal MLP", fontsize=14, fontweight="bold", y=0.96)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_17_mlp_architecture.png")
plt.close(fig)
print("fig_17_mlp_architecture.png guardada")"""
))

# MLP build & train
cells.append(code(
r"""# ── Construcción y entrenamiento del MLP ─────────────────────────────────────
n_features = X_train_proc.shape[1]

def build_mlp(n_in: int) -> tf.keras.Model:
    inp = layers.Input(shape=(n_in,))
    x   = layers.Dense(256, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.25)(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.25)(x)
    x   = layers.Dense(64,  activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.20)(x)
    out = layers.Dense(1,   activation="sigmoid")(x)
    m   = tf.keras.Model(inputs=inp, outputs=out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return m

mlp_model = build_mlp(n_features)
mlp_model.summary()

cb_list = [
    tf_callbacks.EarlyStopping(
        monitor="val_recall", patience=15,
        restore_best_weights=True, mode="max", min_delta=0.001,
    ),
    tf_callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=7, factor=0.5, min_lr=1e-6, verbose=0,
    ),
]

print("\nEntrenando MLP…")
mlp_history = mlp_model.fit(
    X_train_proc, y_train.values,
    validation_data=(X_val_proc, y_val.values),
    epochs=100,
    batch_size=512,
    class_weight=class_weight_dict,
    callbacks=cb_list,
    verbose=1,
)
n_epochs = len(mlp_history.history["loss"])
print(f"\nEntrenamiento completado en {n_epochs} épocas")

mlp_model.save(str(MODELS_DIR / "mlp_model.keras"))
print(f"MLP guardado: {MODELS_DIR / 'mlp_model.keras'}")"""
))

# fig_18: Learning curves
cells.append(code(
r"""# ── fig_18: Curvas de aprendizaje del MLP ────────────────────────────────────
hist = mlp_history.history
epochs_r = range(1, len(hist["loss"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Loss
axes[0].plot(epochs_r, hist["loss"],     color="#4C8BDA", lw=2, label="Train")
axes[0].plot(epochs_r, hist["val_loss"], color="#E05C4B", lw=2, linestyle="--", label="Val")
axes[0].set_title("Pérdida (binary_crossentropy)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(lw=0.4, alpha=0.5)

# Recall
axes[1].plot(epochs_r, hist["recall"],     color="#4C8BDA", lw=2, label="Train")
axes[1].plot(epochs_r, hist["val_recall"], color="#E05C4B", lw=2, linestyle="--", label="Val")
axes[1].set_title("Recall (sensibilidad)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Recall")
axes[1].legend(); axes[1].grid(lw=0.4, alpha=0.5)

# AUC-ROC
axes[2].plot(epochs_r, hist["auc_roc"],     color="#4C8BDA", lw=2, label="Train")
axes[2].plot(epochs_r, hist["val_auc_roc"], color="#E05C4B", lw=2, linestyle="--", label="Val")
axes[2].set_title("AUC-ROC", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Época"); axes[2].set_ylabel("AUC")
axes[2].legend(); axes[2].grid(lw=0.4, alpha=0.5)

fig.suptitle("Curvas de aprendizaje — MLP", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_18_mlp_learning_curves.png")
plt.close(fig)
print("fig_18_mlp_learning_curves.png guardada")"""
))

# fig_19: MLP threshold
cells.append(code(
r"""# ── fig_19: Barrido de umbral MLP (validación) ───────────────────────────────
proba_mlp_val = mlp_model.predict(X_val_proc, verbose=0).flatten()
thresholds_mlp = np.linspace(0.01, 0.99, 99)

auc_roc_mlp_v = roc_auc_score(y_val, proba_mlp_val)
auc_pr_mlp_v  = average_precision_score(y_val, proba_mlp_val)

recalls_m, f1s_m, scores_m = [], [], []
for thr in thresholds_mlp:
    p_m = (proba_mlp_val >= thr).astype(int)
    r_m = recall_score(y_val, p_m, zero_division=0)
    f_m = f1_score(y_val, p_m, zero_division=0)
    s_m = 0.35 * r_m + 0.30 * f_m + 0.20 * auc_pr_mlp_v + 0.15 * auc_roc_mlp_v
    recalls_m.append(r_m); f1s_m.append(f_m); scores_m.append(s_m)

mlp_opt_idx = int(np.argmax(scores_m))
mlp_thr     = float(thresholds_mlp[mlp_opt_idx])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds_mlp, recalls_m, color="#4C8BDA", label="Recall", lw=2)
ax.plot(thresholds_mlp, f1s_m,     color="#E05C4B", label="F1",     lw=2)
ax.plot(thresholds_mlp, scores_m,  color="#27AE60", label="Score compuesto", lw=2, linestyle="--")
ax.axvline(mlp_thr, color="gray", lw=1.5, linestyle=":",
           label=f"Umbral óptimo = {mlp_thr:.2f}")
ax.set_xlabel("Umbral de decisión"); ax.set_ylabel("Métrica")
ax.set_title("Barrido de umbral — MLP (validación)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(lw=0.4, alpha=0.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_19_mlp_threshold_scan.png")
plt.close(fig)
print(f"fig_19_mlp_threshold_scan.png guardada  —  Umbral óptimo MLP: {mlp_thr:.2f}")"""
))

# fig_20, fig_21: MLP evaluation
cells.append(code(
r"""# ── Evaluación MLP en test (una sola vez) ────────────────────────────────────
proba_mlp_test = mlp_model.predict(X_test_proc, verbose=0).flatten()
preds_mlp_test = (proba_mlp_test >= mlp_thr).astype(int)

mlp_recall    = recall_score(y_test, preds_mlp_test, zero_division=0)
mlp_precision = precision_score(y_test, preds_mlp_test, zero_division=0)
mlp_f1        = f1_score(y_test, preds_mlp_test, zero_division=0)
mlp_auc_roc   = roc_auc_score(y_test, proba_mlp_test)
mlp_auc_pr    = average_precision_score(y_test, proba_mlp_test)
mlp_score     = 0.35 * mlp_recall + 0.30 * mlp_f1 + 0.20 * mlp_auc_pr + 0.15 * mlp_auc_roc

mlp_res = {
    "Modelo":     "MLP",
    "Recall":     mlp_recall,
    "Precision":  mlp_precision,
    "F1":         mlp_f1,
    "AUC-ROC":    mlp_auc_roc,
    "AUC-PR":     mlp_auc_pr,
    "Threshold":  mlp_thr,
    "Score":      mlp_score,
    "proba_test": proba_mlp_test,
}

print("=== RESULTADOS MLP (test set) ===")
for k, v in mlp_res.items():
    if k != "proba_test":
        print(f"  {k:<12}: {v:.4f}" if isinstance(v, float) else f"  {k:<12}: {v}")

# Guardar métricas MLP
mlp_df = pd.DataFrame([{k: v for k, v in mlp_res.items() if k != "proba_test"}])
mlp_df.to_csv(METRICS_DIR / "mlp_results.csv", index=False)
with open(METRICS_DIR / "mlp_threshold.json", "w", encoding="utf-8") as f:
    json.dump({"mlp_threshold": mlp_thr}, f)

# ── fig_20: Curvas ROC y PR — MLP vs mejor ML ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC
fpr_m, tpr_m, _ = roc_curve(y_test, proba_mlp_test)
axes[0].plot(fpr_m, tpr_m, color="#9B59B6", lw=2.5,
             label=f"MLP (AUC={mlp_auc_roc:.4f})")
fpr_b, tpr_b, _ = roc_curve(y_test, best_ml_res["proba_test"])
axes[0].plot(fpr_b, tpr_b, color="#4C8BDA", lw=2, linestyle="--",
             label=f"{best_ml_name} (AUC={best_ml_res['AUC-ROC']:.4f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("Curva ROC — MLP vs mejor ML", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(lw=0.4, alpha=0.5)

# PR
prec_m, rec_m, _ = precision_recall_curve(y_test, proba_mlp_test)
axes[1].plot(rec_m, prec_m, color="#9B59B6", lw=2.5,
             label=f"MLP (AP={mlp_auc_pr:.4f})")
prec_b, rec_b, _ = precision_recall_curve(y_test, best_ml_res["proba_test"])
axes[1].plot(rec_b, prec_b, color="#4C8BDA", lw=2, linestyle="--",
             label=f"{best_ml_name} (AP={best_ml_res['AUC-PR']:.4f})")
axes[1].axhline(y_test.mean(), color="gray", lw=1, linestyle="--", label="Baseline")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Curva PR — MLP vs mejor ML", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(lw=0.4, alpha=0.5)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_20_mlp_roc_pr.png")
plt.close(fig)

# ── fig_21: Matriz de confusión MLP ──────────────────────────────────────────
cm_mlp = confusion_matrix(y_test, preds_mlp_test)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp,
                               display_labels=["Sano (0)", "Cáncer (1)"])
disp.plot(ax=ax, cmap="Purples", colorbar=False)
ax.set_title(f"Matriz de confusión — MLP\n(umbral={mlp_thr:.2f})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_21_mlp_confusion_matrix.png")
plt.close(fig)

print("fig_20, fig_21 guardadas ✓")"""
))

# ══════════════════════════════════════════════════════════════════════════════
# CELDAS FINAL RANKING
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"""---
## 31–34. Comparativa Final, Ranking y Métricas

**Score compuesto** = `0.35·Recall + 0.30·F1 + 0.20·AUC-PR + 0.15·AUC-ROC`"""
))

# Final ranking + fig_final
cells.append(code(
r"""# ── Ranking global (ML + MLP) ─────────────────────────────────────────────────
all_results = ml_results + [mlp_res]
all_df = pd.DataFrame([{k: v for k, v in r.items() if k != "proba_test"}
                        for r in all_results])
all_df = all_df.sort_values("Score", ascending=False).reset_index(drop=True)
all_df.index = all_df.index + 1  # ranking 1-based

print("=== RANKING FINAL (test set) ===")
print(all_df[["Modelo", "Recall", "Precision", "F1",
              "AUC-ROC", "AUC-PR", "Threshold", "Score"]].to_string())

# ── fig_final_metrics_comparison ─────────────────────────────────────────────
colors_all = ["#4C8BDA", "#E05C4B", "#27AE60", "#F39C12", "#9B59B6", "#E67E22"]
metrics_final = ["Recall", "Precision", "F1", "AUC-ROC", "AUC-PR"]

fig, axes = plt.subplots(1, len(metrics_final), figsize=(18, 6))
for ax, metric in zip(axes, metrics_final):
    vals   = all_df[metric].values
    models = all_df["Modelo"].values
    bars   = ax.barh(models, vals, color=colors_all[:len(models)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel(metric); ax.set_xlim(0, 1.18)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    if ax is not axes[0]:
        ax.set_yticklabels([])

fig.suptitle("Comparativa final — todos los modelos (test set)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_final_metrics_comparison.png")
plt.close(fig)

# ── fig_final_precision_recall_space ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
for i, (_, row) in enumerate(all_df.iterrows()):
    col = colors_all[i - 1]
    ax.scatter(row["Recall"], row["Precision"], s=220, color=col,
               zorder=5, edgecolors="white", lw=1.2,
               label=f"{row['Modelo']}  F1={row['F1']:.3f}")
    ax.annotate(row["Modelo"], (row["Recall"], row["Precision"]),
                textcoords="offset points", xytext=(9, 4),
                fontsize=9, color=col, fontweight="bold")

# Iso-curvas F1
rec_range = np.linspace(0.01, 1.0, 300)
for f1_val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    denom = 2 * rec_range - f1_val
    valid = denom > 0
    prec_curve = np.where(valid, f1_val * rec_range / denom, np.nan)
    mask = valid & (prec_curve <= 1.0)
    ax.plot(rec_range[mask], prec_curve[mask], "gray", lw=0.7, alpha=0.4, linestyle=":")
    idxs = np.where(mask)[0]
    if len(idxs) > 0:
        mid = idxs[len(idxs) // 2]
        ax.text(rec_range[mid], prec_curve[mid], f"F1={f1_val:.1f}",
                fontsize=7, color="gray", alpha=0.6)

ax.set_xlabel("Recall (sensibilidad)"); ax.set_ylabel("Precision")
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
ax.set_title("Espacio Precision-Recall — todos los modelos", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower left"); ax.grid(lw=0.4, alpha=0.4)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig_final_precision_recall_space.png")
plt.close(fig)

print("fig_final_metrics_comparison.png  +  fig_final_precision_recall_space.png guardadas ✓")"""
))

# Save all metrics
cells.append(code(
r"""# ── Guardar todos los artefactos de métricas ─────────────────────────────────
# all_models_metrics.csv
all_df.to_csv(METRICS_DIR / "all_models_metrics.csv", index=True)

# final_model_ranking.csv
ranking_df = all_df[["Modelo", "Recall", "Precision", "F1",
                      "AUC-ROC", "AUC-PR", "Threshold", "Score"]].copy()
ranking_df.to_csv(METRICS_DIR / "final_model_ranking.csv", index=True)

# executive_summary.json
best_row = all_df.iloc[0]
exec_summary = {
    "mejor_modelo":    str(best_row["Modelo"]),
    "mejor_recall":    float(best_row["Recall"]),
    "mejor_precision": float(best_row["Precision"]),
    "mejor_f1":        float(best_row["F1"]),
    "mejor_auc_roc":   float(best_row["AUC-ROC"]),
    "mejor_auc_pr":    float(best_row["AUC-PR"]),
    "mejor_threshold": float(best_row["Threshold"]),
    "mejor_score":     float(best_row["Score"]),
    "n_modelos":       int(len(all_df)),
    "n_pacientes":     int(df_merged.shape[0]),
    "n_features":      int(len(ALL_FEATURES)),
    "prevalencia_pct": float(round(df_merged["cancer"].mean() * 100, 4)),
}
with open(METRICS_DIR / "executive_summary.json", "w", encoding="utf-8") as f:
    json.dump(exec_summary, f, ensure_ascii=False, indent=2)

print("Artefactos de métricas guardados:")
for p in sorted(METRICS_DIR.glob("*.csv")) + sorted(METRICS_DIR.glob("*.json")):
    print(f"  {p.name}")

print("\nModelos guardados:")
for p in sorted(MODELS_DIR.glob("*.pkl")) + sorted(MODELS_DIR.glob("*.keras")):
    print(f"  {p.name}")

print("\nFiguras generadas:")
for p in sorted(FIGURES_DIR.glob("fig_*.png")):
    print(f"  {p.name}")

print(f"\n{'='*60}")
print(f"RESUMEN EJECUTIVO")
print(f"{'='*60}")
print(f"Mejor modelo   : {exec_summary['mejor_modelo']}")
print(f"Recall (test)  : {exec_summary['mejor_recall']:.4f}")
print(f"F1 (test)      : {exec_summary['mejor_f1']:.4f}")
print(f"AUC-ROC (test) : {exec_summary['mejor_auc_roc']:.4f}")
print(f"AUC-PR (test)  : {exec_summary['mejor_auc_pr']:.4f}")
print(f"Umbral óptimo  : {exec_summary['mejor_threshold']:.2f}")
print(f"\nNotebook ejecutado completamente. Run All ✓")"""
))

# Final summary markdown
cells.append(md(
"""---
## Resumen de la ejecución completa

| Artefacto | Ubicación |
|---|---|
| Dataset procesado | `data/processed/cancer_merged.csv` |
| Pipeline sklearn | `data/processed/preprocess_pipeline.pkl` |
| Column config | `data/processed/column_config.json` |
| Mejor modelo ML | `outputs/models/best_ml_model.pkl` |
| Modelo MLP | `outputs/models/mlp_model.keras` |
| Métricas todos los modelos | `outputs/metrics/all_models_metrics.csv` |
| Ranking final | `outputs/metrics/final_model_ranking.csv` |
| Resumen ejecutivo | `outputs/metrics/executive_summary.json` |
| Figuras EDA | `outputs/figures/fig_01 – fig_08` |
| Figuras Preprocessing | `outputs/figures/fig_09 – fig_11` |
| Figuras ML clásicos | `outputs/figures/fig_12 – fig_16` |
| Figuras MLP | `outputs/figures/fig_17 – fig_21` |
| Figuras comparativa final | `outputs/figures/fig_final_*` |

> **Protocolo anti-data-leakage verificado**: pipeline ajustado solo en train · umbral optimizado en validación · test evaluado una sola vez."""
))

# ══════════════════════════════════════════════════════════════════════════════
# GENERAR NOTEBOOK JSON
# ══════════════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

output_path = Path(r"c:\inteligencia artificial\aa trabajo optativo\analysis.ipynb")
with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, ensure_ascii=False, indent=1)

n = len(cells)
print(f"OK Notebook generado: {output_path}")
print(f"OK Total de celdas  : {n}")
print(f"   ({sum(1 for c in cells if c['cell_type']=='code')} codigo  +  "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
