"""
app.py — Streamlit demo: Sistema de Predicción de Diagnóstico de Cáncer
Universidad Alfonso X el Sabio · Bases de Datos e IA · Curso 2025-2026

Uso:
    streamlit run app.py

Requiere artefactos generados por analysis.ipynb (secciones 22, 29, 42-46).
Si faltan, la app muestra avisos claros sin romper.
"""

# ── Imports estándar ─────────────────────────────────────────────────────────
import json
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Rutas de artefactos ──────────────────────────────────────────────────────
ROOT         = pathlib.Path(__file__).parent
PROC_DIR     = ROOT / "data"    / "processed"
METRICS_DIR  = ROOT / "outputs" / "metrics"
MODELS_DIR   = ROOT / "outputs" / "models"
FIGURES_DIR  = ROOT / "outputs" / "figures"

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CancerRisk · Sistema de Predicción",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS mínimo para tarjetas KPI y alertas clínicas ─────────────────────────
st.markdown(
    """
    <style>
    /* Fuente base */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* Tarjeta KPI */
    .kpi-card {
        background: #f7f9fc;
        border: 1px solid #dde3ed;
        border-left: 4px solid #2c5f8a;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .kpi-label  { font-size: 0.78rem; color: #6b7a8d; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value  { font-size: 1.65rem; font-weight: 700; color: #1a3a5c;
                  margin: 2px 0; }
    .kpi-sub    { font-size: 0.75rem; color: #8a96a3; }

    /* Resultado de predicción */
    .result-low    { background:#e8f5e9; border-left:5px solid #43a047;
                     border-radius:6px; padding:16px 20px; }
    .result-medium { background:#fff8e1; border-left:5px solid #fb8c00;
                     border-radius:6px; padding:16px 20px; }
    .result-high   { background:#fce4ec; border-left:5px solid #e53935;
                     border-radius:6px; padding:16px 20px; }

    /* Aviso ético */
    .ethics-box {
        background:#fff3cd; border:1px solid #ffc107;
        border-radius:6px; padding:12px 16px;
        font-size:0.85rem; color:#664d03;
    }

    /* Título de sección */
    .section-title {
        font-size:1.15rem; font-weight:700; color:#1a3a5c;
        border-bottom:2px solid #2c5f8a; padding-bottom:4px;
        margin-bottom:16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# CARGA DE ARTEFACTOS (con fallbacks)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando artefactos…")
def load_artifacts():
    """Carga todos los artefactos. Devuelve un dict con None donde falten."""
    arts = {}

    # Pipeline de preprocessing
    pip_path = PROC_DIR / "preprocess_pipeline.pkl"
    if pip_path.exists():
        import joblib
        arts["pipeline"] = joblib.load(pip_path)
    else:
        arts["pipeline"] = None

    # Mejor modelo ML
    model_path = MODELS_DIR / "best_ml_model.pkl"
    if model_path.exists():
        import joblib
        arts["model"] = joblib.load(model_path)
        arts["model_path"] = model_path.name
    else:
        arts["model"] = None
        arts["model_path"] = None

    # Ranking final
    rank_path = METRICS_DIR / "final_model_ranking.csv"
    arts["ranking"] = pd.read_csv(rank_path, index_col=0) if rank_path.exists() else None

    # Métricas todos los modelos
    all_metrics_path = METRICS_DIR / "all_models_metrics.csv"
    arts["all_metrics"] = (
        pd.read_csv(all_metrics_path, index_col=0) if all_metrics_path.exists() else None
    )

    # Thresholds ML
    thr_path = METRICS_DIR / "ml_thresholds.csv"
    arts["ml_thresholds"] = pd.read_csv(thr_path) if thr_path.exists() else None

    # Resumen ejecutivo
    summ_path = METRICS_DIR / "executive_summary.json"
    if summ_path.exists():
        arts["summary"] = json.loads(summ_path.read_text(encoding="utf-8"))
    else:
        arts["summary"] = None

    # Threshold del mejor modelo (desde ml_thresholds o summary)
    arts["best_threshold"] = None
    if arts["summary"]:
        arts["best_threshold"] = arts["summary"].get("mejor_threshold")
    if arts["best_threshold"] is None and arts["ml_thresholds"] is not None:
        arts["best_threshold"] = float(arts["ml_thresholds"]["Threshold_optimo"].iloc[0])

    # Configuración de columnas
    cfg_path = PROC_DIR / "column_config.json"
    if cfg_path.exists():
        arts["col_cfg"] = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        arts["col_cfg"] = None  # se reconstruirá desde el dataset

    # Dataset principal
    ds_path = PROC_DIR / "cancer_merged.csv"
    arts["dataset"] = pd.read_csv(ds_path) if ds_path.exists() else None

    return arts


@st.cache_data(show_spinner=False)
def get_column_config(_arts: dict) -> dict:
    """Devuelve configuración de columnas desde JSON o la infiere del dataset."""
    if _arts["col_cfg"]:
        return _arts["col_cfg"]

    # Fallback: reconstrucción desde el conocimiento del dataset
    return {
        "num_cols": [
            "glucosa", "colesterol", "trigliceridos", "hemoglobina",
            "leucocitos", "plaquetas", "creatinina",
            "edad", "distancia_hospital_km", "num_hijos",
        ],
        "bin_cols": [
            "fumador", "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
            "mut_PIK3CA", "mut_ALK", "mut_BRAF",
            "diabetes", "hipertension", "obesidad",
            "enfermedad_cardiaca", "asma", "epoc",
        ],
        "ord_cols": ["actividad_fisica"],
        "ord_categories": [["Baja", "Moderada", "Alta"]],
        "cat_cols": ["nivel_educativo", "nivel_ingresos", "zona", "estado_civil"],
        "all_features": [
            "glucosa", "colesterol", "trigliceridos", "hemoglobina",
            "leucocitos", "plaquetas", "creatinina",
            "edad", "distancia_hospital_km", "num_hijos",
            "fumador", "mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS",
            "mut_PIK3CA", "mut_ALK", "mut_BRAF",
            "diabetes", "hipertension", "obesidad",
            "enfermedad_cardiaca", "asma", "epoc",
            "actividad_fisica",
            "nivel_educativo", "nivel_ingresos", "zona", "estado_civil",
        ],
        "target": "cancer",
        "id_col": "paciente_id",
    }


# ── Metadatos clínicos para las variables ────────────────────────────────────
NUM_META = {
    "glucosa":              ("Glucosa en ayunas",        "mg/dL",   55.0,  179.0, 102.0),
    "colesterol":           ("Colesterol total",          "mg/dL",  120.0,  320.0, 194.0),
    "trigliceridos":        ("Triglicéridos",             "mg/dL",   50.0,  322.0, 156.0),
    "hemoglobina":          ("Hemoglobina",               "g/dL",     8.0,   18.0,  13.9),
    "leucocitos":           ("Leucocitos",                "×10³/µL",  2.0,   15.0,   7.1),
    "plaquetas":            ("Plaquetas",                 "×10³/µL", 100.0, 490.0, 255.0),
    "creatinina":           ("Creatinina sérica",         "mg/dL",    0.3,    2.1,   1.0),
    "edad":                 ("Edad",                      "años",    20.0,   90.0,  54.5),
    "distancia_hospital_km":("Distancia al hospital",    "km",       0.5,  250.0,  25.6),
    "num_hijos":            ("Número de hijos",           "",         0.0,    8.0,   1.5),
}
BIN_LABELS = {
    "fumador":           "Fumador activo",
    "mut_BRCA1":         "Mutación BRCA1",
    "mut_TP53":          "Mutación TP53",
    "mut_EGFR":          "Mutación EGFR",
    "mut_KRAS":          "Mutación KRAS",
    "mut_PIK3CA":        "Mutación PIK3CA",
    "mut_ALK":           "Mutación ALK",
    "mut_BRAF":          "Mutación BRAF",
    "diabetes":          "Diabetes mellitus",
    "hipertension":      "Hipertensión arterial",
    "obesidad":          "Obesidad (IMC ≥ 30)",
    "enfermedad_cardiaca":"Cardiopatía diagnosticada",
    "asma":              "Asma bronquial",
    "epoc":              "EPOC",
}
CAT_OPTIONS = {
    "actividad_fisica": ["Baja", "Moderada", "Alta"],
    "nivel_educativo":  ["Sin estudios", "Primaria", "Secundaria", "Universitario"],
    "nivel_ingresos":   ["Muy bajo", "Bajo", "Medio", "Alto"],
    "zona":             ["Rural", "Semiurbana", "Urbana"],
    "estado_civil":     ["Soltero", "Casado", "Divorciado", "Viudo"],
}
CAT_LABELS = {
    "actividad_fisica": "Actividad física",
    "nivel_educativo":  "Nivel educativo",
    "nivel_ingresos":   "Nivel de ingresos",
    "zona":             "Zona de residencia",
    "estado_civil":     "Estado civil",
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def ethics_banner() -> None:
    st.markdown(
        '<div class="ethics-box">'
        "⚠️ <strong>Aviso ético:</strong> Este sistema es una herramienta de demostración académica "
        "entrenada sobre datos <em>sintéticos</em>. "
        "<strong>No constituye un diagnóstico médico</strong> y no debe utilizarse para tomar "
        "decisiones clínicas reales. Cualquier decisión de salud debe tomarse con un profesional médico cualificado."
        "</div>",
        unsafe_allow_html=True,
    )


def artifact_warning(name: str) -> None:
    st.warning(
        f"⚙️ **{name}** no encontrado. "
        "Ejecuta `analysis.ipynb` de principio a fin para generar todos los artefactos.",
        icon="⚠️",
    )


def show_image_safe(path: pathlib.Path, caption: str = "", width: Optional[int] = None) -> bool:
    """Muestra una imagen si existe; devuelve True si tuvo éxito."""
    if path.exists():
        kw = {"caption": caption, "use_container_width": True}
        if width:
            kw = {"caption": caption, "width": width}
        st.image(str(path), **kw)
        return True
    return False


def predict_patient(arts: dict, patient_row: pd.Series, col_cfg: dict) -> Optional[dict]:
    """
    Aplica el pipeline y predice la probabilidad de cáncer para un paciente.
    Devuelve dict con probabilidad, predicción y umbral, o None si faltan artefactos.
    """
    if arts["pipeline"] is None or arts["model"] is None:
        return None

    features = col_cfg["all_features"]
    # Construir DataFrame de una fila con las features en el orden correcto
    row_df = pd.DataFrame([{f: patient_row.get(f) for f in features}])

    # Transformar con el pipeline serializado
    X_proc = arts["pipeline"].transform(row_df)

    # Predecir probabilidad
    proba = float(arts["model"].predict_proba(X_proc)[0, 1])
    thr   = arts["best_threshold"] if arts["best_threshold"] is not None else 0.5
    pred  = int(proba >= thr)

    # Nivel de riesgo
    if proba < 0.20:
        risk_level, risk_css = "Bajo", "result-low"
    elif proba < 0.50:
        risk_level, risk_css = "Medio", "result-medium"
    else:
        risk_level, risk_css = "Alto", "result-high"

    return {
        "proba":      proba,
        "prediction": pred,
        "threshold":  thr,
        "risk_level": risk_level,
        "risk_css":   risk_css,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CARGA (se ejecuta una sola vez por sesión)
# ═══════════════════════════════════════════════════════════════════════════
arts    = load_artifacts()
col_cfg = get_column_config(_arts=arts)
dataset = arts["dataset"]


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Navegación
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 CancerRisk")
    st.caption("Universidad Alfonso X el Sabio · IA 2025-2026")
    st.divider()

    page = st.radio(
        "Navegación",
        options=[
            "📊 Resumen del proyecto",
            "🧬 Simulador de paciente",
            "📈 Comparativa de modelos",
            "🖼️ Figuras del estudio",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Estado de artefactos en sidebar
    st.caption("**Estado de artefactos**")
    checks = {
        "Pipeline":       arts["pipeline"] is not None,
        "Modelo ML":      arts["model"] is not None,
        "Ranking":        arts["ranking"] is not None,
        "Métricas":       arts["all_metrics"] is not None,
        "Dataset":        dataset is not None,
    }
    for name, ok in checks.items():
        icon = "🟢" if ok else "🔴"
        st.caption(f"{icon} {name}")

    st.divider()
    st.caption("ℹ️ Solo demostración académica")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — Resumen del proyecto
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Resumen del proyecto":

    st.title("Sistema de Predicción de Diagnóstico de Cáncer")
    st.markdown(
        "Estudio comparativo de modelos clásicos de ML vs red neuronal MLP "
        "sobre un dataset sintético de **50 001 pacientes oncológicos**."
    )
    ethics_banner()
    st.divider()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Métricas del mejor modelo (test set)</div>',
                unsafe_allow_html=True)

    summ = arts["summary"]
    if summ:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("Mejor modelo", summ.get("mejor_modelo", "—"),
                     "por F1 + Recall ponderados")
        with c2:
            kpi_card("F1 (cáncer=1)", f'{summ.get("mejor_f1", 0):.4f}',
                     "media armónica P/R")
        with c3:
            kpi_card("Recall", f'{summ.get("mejor_recall", 0):.4f}',
                     "sensibilidad oncológica")
        with c4:
            kpi_card("AUC-ROC", f'{summ.get("mejor_auc_roc", 0):.4f}',
                     "capacidad discriminativa")
        with c5:
            kpi_card("Umbral óptimo", f'{summ.get("mejor_threshold", 0.5):.2f}',
                     "optimizado en validación")
    elif arts["ranking"] is not None:
        rk = arts["ranking"]
        best = rk.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi_card("Mejor modelo", str(best.get("Modelo", "—")))
        with c2: kpi_card("F1 (cáncer=1)", f'{float(best.get("F1", 0)):.4f}')
        with c3: kpi_card("Recall",        f'{float(best.get("Recall", 0)):.4f}')
        with c4: kpi_card("AUC-ROC",       f'{float(best.get("AUC-ROC", 0)):.4f}')
        with c5: kpi_card("Umbral",        f'{float(best.get("Threshold", 0.5)):.2f}')
    else:
        st.info("Ejecuta el notebook para generar las métricas del proyecto.")

    st.divider()

    # ── Descripción del proyecto ──────────────────────────────────────────────
    col_desc, col_data = st.columns([3, 2])

    with col_desc:
        st.markdown('<div class="section-title">Sobre el proyecto</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            Este trabajo implementa un **pipeline completo de Machine Learning** para
            la predicción binaria de diagnóstico de cáncer, abarcando:

            - **EDA** (15 secciones): análisis de distribuciones, correlaciones y detección
              de variables con riesgo de data leakage.
            - **Preprocessing** (8 secciones): pipeline sklearn con `ColumnTransformer`,
              4 sub-pipelines (numérico, binario, ordinal, nominal), split estratificado
              80/16/20 y compensación del desbalance con `class_weight='balanced'`.
            - **Modelos ML clásicos** (7 secciones): Regresión Logística, Random Forest,
              XGBoost, LightGBM y CatBoost con barrido de umbral sobre validación.
            - **Red neuronal MLP** (9 secciones): arquitectura Dense(256→128→64) +
              BatchNorm + Dropout, EarlyStopping y ReduceLROnPlateau.
            - **Comparativa final** (8 secciones): ranking compuesto, análisis ejecutivo
              clínico y recomendación de despliegue.
            - **Auditoría metodológica**: verificación de 10 criterios de rigor.

            **Protocolo anti-data-leakage estricto**: el pipeline se ajusta solo sobre
            train, el umbral se optimiza sobre validación y el test se evalúa una única vez.
            """
        )

    with col_data:
        st.markdown('<div class="section-title">Dataset</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            | Parámetro | Valor |
            |---|---|
            | Pacientes | 50 001 |
            | Variables totales | 38 |
            | Variables usadas | 29 |
            | Prevalencia cáncer | ≈ 19 % |
            | Desbalance | 4.3 : 1 |
            | Train / Val / Test | 64 / 16 / 20 % |
            | Colecciones | 6 (bioquímica, clínica, genética, económica, hábitos, sociodem.) |

            **Variables excluidas** (data leakage):
            `coste_total`, `coste_farmaco`, `num_ingresos`, `dias_hospital`, `vive`,
            `alcohol` (constante), `tipo_seguro`.
            """
        )

    # ── Figura distribución objetivo ─────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">Distribución de la variable objetivo</div>',
                unsafe_allow_html=True)

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        if not show_image_safe(FIGURES_DIR / "fig_01_target_distribution.png",
                               "Distribución de cáncer = 0 / 1"):
            # Mostrar con datos del dataset si la figura no existe
            if dataset is not None:
                import matplotlib.pyplot as plt
                counts = dataset["cancer"].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.bar(["Sin cáncer (0)", "Con cáncer (1)"],
                       counts.values,
                       color=["#4C8BDA", "#E05C4B"],
                       edgecolor="white", width=0.5)
                for bar, n in zip(ax.patches, counts.values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01,
                            f"{n:,}", ha="center", fontsize=10, fontweight="bold")
                ax.set_title("Distribución de la variable objetivo")
                ax.set_ylabel("Pacientes")
                ax.set_axisbelow(True)
                ax.grid(axis="y", lw=0.5, alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with fig_col2:
        show_image_safe(FIGURES_DIR / "fig_09_splits_distribution.png",
                        "Distribución de clases en train / val / test")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — Simulador de paciente
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧬 Simulador de paciente":

    st.title("🧬 Simulador de Riesgo Oncológico")
    st.markdown(
        "Selecciona un paciente del dataset o introduce datos manualmente. "
        "El sistema aplica el **pipeline de preprocessing** y el **mejor modelo ML** "
        "para estimar la probabilidad de diagnóstico de cáncer."
    )
    ethics_banner()
    st.divider()

    # ── Aviso si faltan artefactos críticos ───────────────────────────────────
    missing_critical = []
    if arts["pipeline"] is None:
        missing_critical.append("`data/processed/preprocess_pipeline.pkl`")
    if arts["model"] is None:
        missing_critical.append("`outputs/models/best_ml_model.pkl`")

    if missing_critical:
        st.error(
            "**Artefactos necesarios no encontrados:**\n\n"
            + "\n".join(f"- {m}" for m in missing_critical)
            + "\n\nEjecuta `analysis.ipynb` (secciones 22 y 29) y vuelve a cargar la app.",
            icon="🚫",
        )
        st.stop()

    # ── Selector de paciente ──────────────────────────────────────────────────
    col_sel, col_info = st.columns([2, 1])

    with col_sel:
        st.markdown('<div class="section-title">Selección de paciente</div>',
                    unsafe_allow_html=True)

        if dataset is not None:
            sample_ids = dataset["paciente_id"].tolist()
            selected_id = st.selectbox(
                "Seleccionar paciente del dataset",
                options=["— Introducir datos manualmente —"] + sample_ids[:2000],
                index=0,
                help="Puedes seleccionar cualquiera de los 50 001 pacientes del dataset.",
            )
            use_existing = selected_id != "— Introducir datos manualmente —"
        else:
            st.warning("Dataset `cancer_merged.csv` no disponible. Introduce datos manualmente.")
            use_existing = False
            selected_id = None

    with col_info:
        if arts["summary"]:
            st.markdown('<div class="section-title">Modelo activo</div>',
                        unsafe_allow_html=True)
            summ = arts["summary"]
            st.metric("Modelo", summ.get("mejor_modelo", "—"))
            st.metric("Umbral", f'{summ.get("mejor_threshold", 0.5):.2f}')
            st.metric("F1 test", f'{summ.get("mejor_f1", 0):.4f}')

    st.divider()

    # ── Formulario de datos del paciente ─────────────────────────────────────
    st.markdown('<div class="section-title">Datos del paciente</div>',
                unsafe_allow_html=True)

    # Valores iniciales: del paciente seleccionado o defaults clínicos
    defaults: dict = {}
    real_label: int = -1
    if use_existing and dataset is not None:
        row = dataset[dataset["paciente_id"] == selected_id].iloc[0]
        for col in col_cfg["all_features"]:
            defaults[col] = row.get(col, None)
        real_label = int(row.get("cancer", -1))
    else:
        for col, (label, unit, lo, hi, mean) in NUM_META.items():
            defaults[col] = mean
        for col in col_cfg["bin_cols"]:
            defaults[col] = 0
        defaults["actividad_fisica"] = "Moderada"
        defaults["nivel_educativo"]  = "Secundaria"
        defaults["nivel_ingresos"]   = "Medio"
        defaults["zona"]             = "Urbana"
        defaults["estado_civil"]     = "Casado"

    with st.form("patient_form"):
        # ── Variables bioquímicas ─────────────────────────────────────────────
        st.markdown("**Analítica sanguínea**")
        bio_cols = st.columns(4)
        bioq_vars = ["glucosa", "colesterol", "trigliceridos", "hemoglobina",
                     "leucocitos", "plaquetas", "creatinina"]
        patient_vals: dict = {}
        for i, col in enumerate(bioq_vars):
            meta = NUM_META[col]
            with bio_cols[i % 4]:
                patient_vals[col] = st.number_input(
                    f"{meta[0]} ({meta[1]})",
                    min_value=float(meta[2]),
                    max_value=float(meta[3]),
                    value=float(defaults.get(col, meta[4])),
                    step=0.1,
                    format="%.1f",
                )

        # ── Datos sociodemográficos ───────────────────────────────────────────
        st.markdown("**Datos sociodemográficos**")
        soc_cols = st.columns(4)
        with soc_cols[0]:
            patient_vals["edad"] = st.number_input(
                "Edad (años)",
                min_value=20, max_value=90,
                value=int(defaults.get("edad", 55)),
                step=1,
            )
        with soc_cols[1]:
            patient_vals["num_hijos"] = st.number_input(
                "Número de hijos",
                min_value=0, max_value=8,
                value=int(defaults.get("num_hijos", 1)),
                step=1,
            )
        with soc_cols[2]:
            patient_vals["distancia_hospital_km"] = st.number_input(
                "Distancia hospital (km)",
                min_value=0.5, max_value=250.0,
                value=float(defaults.get("distancia_hospital_km", 25.0)),
                step=0.5,
            )
        with soc_cols[3]:
            patient_vals["actividad_fisica"] = st.selectbox(
                "Actividad física",
                options=CAT_OPTIONS["actividad_fisica"],
                index=CAT_OPTIONS["actividad_fisica"].index(
                    str(defaults.get("actividad_fisica", "Moderada"))
                ),
            )

        # ── Categoriales ─────────────────────────────────────────────────────
        cat_cols_ui = st.columns(4)
        for i, col in enumerate(["nivel_educativo", "nivel_ingresos", "zona", "estado_civil"]):
            with cat_cols_ui[i]:
                opts = CAT_OPTIONS[col]
                def_val = str(defaults.get(col, opts[0]))
                if def_val not in opts:
                    def_val = opts[0]
                patient_vals[col] = st.selectbox(
                    CAT_LABELS[col], options=opts, index=opts.index(def_val)
                )

        # ── Variables binarias ────────────────────────────────────────────────
        st.markdown("**Factores de riesgo y comorbilidades**")
        bin_col1, bin_col2, bin_col3, bin_col4 = st.columns(4)
        bin_cols_ui = [bin_col1, bin_col2, bin_col3, bin_col4]
        for i, col in enumerate(col_cfg["bin_cols"]):
            with bin_cols_ui[i % 4]:
                label = BIN_LABELS.get(col, col)
                patient_vals[col] = int(
                    st.checkbox(label, value=bool(defaults.get(col, 0)))
                )

        # ── Botón ─────────────────────────────────────────────────────────────
        st.divider()
        submitted = st.form_submit_button(
            "🔍 Calcular riesgo oncológico",
            type="primary",
            use_container_width=True,
        )

    # ── Resultado de la predicción ────────────────────────────────────────────
    if submitted:
        patient_series = pd.Series(patient_vals)
        result = predict_patient(arts, patient_series, col_cfg)

        st.divider()
        st.markdown('<div class="section-title">Resultado del análisis</div>',
                    unsafe_allow_html=True)

        if result is None:
            st.error("No se pudo calcular la predicción. Verifica que los artefactos estén disponibles.")
        else:
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                proba_pct = result["proba"] * 100
                st.markdown(
                    f'<div class="{result["risk_css"]}">'
                    f"<h3 style='margin:0'>Riesgo {result['risk_level']}</h3>"
                    f"<p style='font-size:2rem;font-weight:700;margin:6px 0'>"
                    f"{proba_pct:.1f} %</p>"
                    f"<p style='margin:0'>Probabilidad estimada de cáncer = 1</p>"
                    f"<hr style='border:none;border-top:1px solid rgba(0,0,0,0.1);margin:10px 0'>"
                    f"<p><strong>Umbral de decisión:</strong> {result['threshold']:.2f} "
                    f"(optimizado en validación)</p>"
                    f"<p><strong>Predicción binaria:</strong> "
                    f"{'🔴 Positivo (cáncer=1)' if result['prediction']==1 else '🟢 Negativo (cáncer=0)'}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Gauge con st.progress
                st.markdown("")
                st.markdown(f"**Nivel de probabilidad:** {proba_pct:.1f} %")
                st.progress(min(result["proba"], 1.0))

                st.markdown(
                    '<div class="ethics-box" style="margin-top:12px">'
                    "⚕️ <strong>Recuerda:</strong> esta estimación procede de un modelo "
                    "entrenado sobre datos sintéticos. No reemplaza la evaluación clínica "
                    "de un médico. Consúltalo siempre con un profesional de la salud."
                    "</div>",
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.metric(
                    label="Probabilidad P(cáncer=1)",
                    value=f"{proba_pct:.2f} %",
                    delta=f"Umbral: {result['threshold']:.2f}",
                    delta_color="off",
                )
                if real_label >= 0:
                    real_txt  = "🔴 Sí (cáncer=1)" if real_label == 1 else "🟢 No (cáncer=0)"
                    pred_txt  = "🔴 Positivo" if result["prediction"] == 1 else "🟢 Negativo"
                    match_txt = "✅ Correcto" if real_label == result["prediction"] else "❌ Incorrecto"
                    st.markdown(
                        f"""
                        | Campo | Valor |
                        |---|---|
                        | **Etiqueta real** | {real_txt} |
                        | **Predicción** | {pred_txt} |
                        | **Resultado** | {match_txt} |
                        """
                    )


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — Comparativa de modelos
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Comparativa de modelos":

    st.title("📈 Comparativa de Modelos ML vs MLP")
    st.markdown(
        "Resultados en el **test set** (evaluado una única vez) con el umbral "
        "optimizado en validación para cada modelo."
    )
    ethics_banner()
    st.divider()

    # ── Ranking global ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Ranking global (score compuesto)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Score = 0.35·Recall + 0.30·F1 + 0.20·AUC-PR + 0.15·AUC-ROC "
        "— pesos por prioridad clínica (minimizar falsos negativos)"
    )

    if arts["ranking"] is not None:
        rk = arts["ranking"].copy()
        # Formatear columnas numéricas
        fmt_cols = ["Recall", "F1", "AUC-PR", "AUC-ROC", "Score", "Score_compuesto"]
        for c in fmt_cols:
            if c in rk.columns:
                rk[c] = rk[c].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")
        if "Delta_vs_mejor" in rk.columns:
            rk["Delta_vs_mejor"] = rk["Delta_vs_mejor"].apply(
                lambda x: f"{float(x):+.4f}" if pd.notna(x) else "—"
            )
        cols_show = [c for c in ["Modelo", "Tipo", "Recall", "F1", "AUC-PR",
                                  "AUC-ROC", "Score_compuesto", "Score", "Delta_vs_mejor"]
                     if c in rk.columns]
        st.dataframe(rk[cols_show], use_container_width=True, height=280)
    else:
        artifact_warning("final_model_ranking.csv")

    st.divider()

    # ── Tabla completa de métricas ────────────────────────────────────────────
    st.markdown('<div class="section-title">Tabla completa de métricas (test set)</div>',
                unsafe_allow_html=True)

    if arts["all_metrics"] is not None:
        am = arts["all_metrics"].copy()
        fmt_cols2 = ["Precision", "Recall", "F1", "AUC-ROC", "AUC-PR", "Accuracy"]
        for c in fmt_cols2:
            if c in am.columns:
                am[c] = am[c].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")
        if "Threshold" in am.columns:
            am["Threshold"] = am["Threshold"].apply(
                lambda x: f"{float(x):.2f}" if pd.notna(x) else "—"
            )
        if "Train_s" in am.columns:
            am["Train_s"] = am["Train_s"].apply(
                lambda x: f"{float(x):.1f} s" if pd.notna(x) else "—"
            )
        cols_show2 = [c for c in ["Modelo", "Tipo", "Threshold", "Precision",
                                   "Recall", "F1", "AUC-ROC", "AUC-PR",
                                   "Accuracy", "Train_s"]
                      if c in am.columns]
        st.dataframe(am[cols_show2], use_container_width=True, height=300)

        # Interpretación breve
        st.info(
            "📌 **Accuracy** aparece sólo como referencia. Con prevalencia ≈19 %, "
            "un clasificador trivial (siempre cáncer=0) alcanza ~81 % de accuracy "
            "sin detectar ningún cáncer. Las métricas relevantes son "
            "**Recall, F1 y AUC-PR**.",
            icon="ℹ️",
        )
    else:
        artifact_warning("all_models_metrics.csv")

    st.divider()

    # ── Figuras comparativas ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Visualizaciones comparativas</div>',
                unsafe_allow_html=True)

    tab_bar, tab_pr, tab_roc, tab_cm = st.tabs([
        "Barras de métricas", "Espacio Precision–Recall",
        "Curvas ROC", "Curvas PR",
    ])

    with tab_bar:
        showed = show_image_safe(
            FIGURES_DIR / "fig_final_metrics_comparison.png",
            "Comparativa global de métricas — todos los modelos",
        )
        if not showed:
            show_image_safe(
                FIGURES_DIR / "fig_16_metrics_comparison.png",
                "Comparativa de métricas ML clásicos",
            )
            if not (FIGURES_DIR / "fig_final_metrics_comparison.png").exists():
                st.info("Figura no disponible aún. Ejecuta las secciones 43 y 28 del notebook.")

    with tab_pr:
        showed = show_image_safe(
            FIGURES_DIR / "fig_final_precision_recall_space.png",
            "Espacio Precision–Recall — todos los modelos",
        )
        if not showed:
            st.info("Figura no disponible. Ejecuta la sección 44 del notebook.")

    with tab_roc:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            show_image_safe(FIGURES_DIR / "fig_13_roc_curves.png", "Curvas ROC — modelos ML")
        with col_r2:
            show_image_safe(FIGURES_DIR / "fig_20_mlp_roc_pr.png", "Curva ROC — MLP")
        if not (FIGURES_DIR / "fig_13_roc_curves.png").exists():
            st.info("Figuras ROC no disponibles. Ejecuta las secciones 28 y 37 del notebook.")

    with tab_cm:
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            show_image_safe(FIGURES_DIR / "fig_15_confusion_matrix_best.png",
                            "Matriz de confusión — mejor modelo ML")
        with col_cm2:
            show_image_safe(FIGURES_DIR / "fig_21_mlp_confusion_matrix.png",
                            "Matriz de confusión — MLP")
        if not (FIGURES_DIR / "fig_15_confusion_matrix_best.png").exists():
            st.info("Matrices de confusión no disponibles. Ejecuta secciones 28 y 37.")

    # ── Thresholds ───────────────────────────────────────────────────────────
    if arts["ml_thresholds"] is not None:
        st.divider()
        st.markdown('<div class="section-title">Umbrales óptimos por modelo (validación)</div>',
                    unsafe_allow_html=True)
        thr_df = arts["ml_thresholds"].copy()
        if "Threshold_optimo" in thr_df.columns:
            thr_df["Threshold_optimo"] = thr_df["Threshold_optimo"].apply(
                lambda x: f"{float(x):.2f}"
            )
        if "F1_validacion" in thr_df.columns:
            thr_df["F1_validacion"] = thr_df["F1_validacion"].apply(
                lambda x: f"{float(x):.4f}"
            )
        st.dataframe(thr_df, use_container_width=True, hide_index=True)
        st.caption(
            "El umbral se eligió maximizando F1(cáncer=1) sobre el conjunto de validación. "
            "El test set nunca se usó en este proceso."
        )


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — Figuras del estudio
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Figuras del estudio":

    st.title("🖼️ Figuras del Estudio")
    st.markdown(
        "Galería de las 21+ figuras generadas durante el análisis. "
        "Las figuras se crean al ejecutar `analysis.ipynb`."
    )
    ethics_banner()
    st.divider()

    # ── Catálogo de figuras por sección ──────────────────────────────────────
    FIGURE_CATALOG = [
        # (nombre_archivo, título, sección)
        ("fig_01_target_distribution.png",    "Distribución variable objetivo",         "EDA"),
        ("fig_02_nulls_by_collection.png",       "Mapa de valores nulos",                  "EDA"),
        ("fig_03_bioq_distributions.png",     "Distribuciones bioquímicas",             "EDA"),
        ("fig_04_continuous_by_cancer.png",   "Variables continuas por clase",          "EDA"),
        ("fig_05_binary_by_cancer.png",       "Variables binarias — prevalencia",       "EDA"),
        ("fig_06_categorical_by_cancer.png",  "Variables categóricas por clase",        "EDA"),
        ("fig_07_correlation_matrix.png",     "Matriz de correlación",                  "EDA"),
        ("fig_08_leakage_correlation.png",    "Variables con riesgo de leakage",        "EDA"),
        ("fig_09_splits_distribution.png",    "Distribución clases en splits",          "Preprocessing"),
        ("fig_10_processed_sample.png",       "Muestra datos procesados",               "Preprocessing"),
        ("fig_11_class_weights.png",          "Desbalance y pesos de clase",            "Preprocessing"),
        ("fig_12_threshold_scan.png",         "Barrido de umbral — modelos ML",         "ML"),
        ("fig_13_roc_curves.png",             "Curvas ROC — modelos ML",                "ML"),
        ("fig_14_pr_curves.png",              "Curvas Precision-Recall — ML",           "ML"),
        ("fig_15_confusion_matrix_best.png",  "Matriz de confusión mejor ML",           "ML"),
        ("fig_16_metrics_comparison.png",     "Comparativa barras — ML clásicos",       "ML"),
        ("fig_17_mlp_architecture.png",       "Diagrama arquitectura MLP",              "MLP"),
        ("fig_18_mlp_learning_curves.png",    "Curvas de aprendizaje MLP",              "MLP"),
        ("fig_19_mlp_threshold_scan.png",     "Barrido umbral MLP",                     "MLP"),
        ("fig_20_mlp_roc_pr.png",             "Curvas ROC y PR — MLP",                  "MLP"),
        ("fig_21_mlp_confusion_matrix.png",   "Matriz de confusión MLP",                "MLP"),
        ("fig_final_metrics_comparison.png",  "Comparativa global métricas",            "Final"),
        ("fig_final_precision_recall_space.png", "Espacio Precision–Recall",            "Final"),
    ]

    sections = ["Todas"] + sorted(set(s for _, _, s in FIGURE_CATALOG))
    selected_sec = st.selectbox("Filtrar por sección:", sections)

    # Contar disponibles
    available = [f for f, _, s in FIGURE_CATALOG
                 if (FIGURES_DIR / f).exists()
                 and (selected_sec == "Todas" or s == selected_sec)]

    total_figs = len([f for f, _, s in FIGURE_CATALOG
                      if selected_sec == "Todas" or s == selected_sec])

    st.caption(
        f"**{len(available)}/{total_figs}** figuras disponibles "
        f"{'en esta sección' if selected_sec != 'Todas' else 'en total'}. "
        "Ejecuta el notebook para generarlas todas."
    )

    if not available:
        st.info(
            "No hay figuras disponibles aún. "
            "Ejecuta `analysis.ipynb` completo para generarlas.",
            icon="ℹ️",
        )
    else:
        # Mostrar en grid de 2 columnas
        filtered = [
            (f, title, sec) for f, title, sec in FIGURE_CATALOG
            if (FIGURES_DIR / f).exists()
            and (selected_sec == "Todas" or sec == selected_sec)
        ]

        for i in range(0, len(filtered), 2):
            row = st.columns(2)
            for j, (fname, title, sec) in enumerate(filtered[i:i+2]):
                with row[j]:
                    st.image(
                        str(FIGURES_DIR / fname),
                        caption=f"[{sec}] {title}",
                        use_container_width=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "🏥 **CancerRisk Demo** · Universidad Alfonso X el Sabio · "
    "Bases de Datos e Inteligencia Artificial 2025-2026 · "
    "Solo uso académico · No válido para decisiones clínicas"
)
