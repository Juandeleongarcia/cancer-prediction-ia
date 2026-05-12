# Predicción de Diagnóstico de Cáncer — ML vs MLP

**Universidad Alfonso X el Sabio**  
Asignatura: Bases de Datos e Inteligencia Artificial · Curso 2025-2026

Estudio comparativo de modelos clásicos de Machine Learning (Regresión Logística,
Random Forest, XGBoost, LightGBM, CatBoost) frente a una red neuronal MLP sobre
un dataset sintético de **50 001 pacientes oncológicos**.

---

## Estructura del proyecto

```
.
├── analysis.ipynb              # Notebook principal (§1–§48)
├── app.py                      # Aplicación Streamlit de demostración
├── requirements.txt            # Dependencias Python
├── README.md
│
├── base de datos/              # Datos crudos (6 colecciones CSV)
│   ├── CASOCANCER_01_BIOQUIMICOS.csv
│   ├── CASOCANCER_02_CLINICOS.csv
│   ├── CASOCANCER_03_GENETICOS.csv
│   ├── CASOCANCER_04_ECONOMICOS.csv
│   ├── CASOCANCER_05_GENERALES.csv
│   ├── CASOCANCER_06_SOCIODEMOGRAFICOS.csv
│   └── metadata_dataset_cancer.md
│
├── data/
│   └── processed/              # Generado por el notebook (§§ 3–22)
│       ├── cancer_merged.csv
│       ├── preprocess_pipeline.pkl
│       └── column_config.json
│
└── outputs/
    ├── figures/                # 21+ figuras PNG (generadas por el notebook)
    ├── metrics/                # CSVs y JSONs de métricas
    │   ├── ml_results.csv
    │   ├── ml_thresholds.csv
    │   ├── mlp_results.csv
    │   ├── mlp_threshold.json
    │   ├── final_model_ranking.csv
    │   ├── all_models_metrics.csv
    │   └── executive_summary.json
    └── models/
        ├── best_ml_model.pkl
        ├── mlp_model.keras
        └── ml_*.pkl            # Modelos individuales
```

---

## Instalación

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Instalar dependencias
pip install -r requirements.txt
```

> **Nota**: TensorFlow y los boosters (XGBoost, LightGBM, CatBoost) son opcionales
> para la app Streamlit. Son necesarios para ejecutar el notebook completo.

---

## Ejecución

### 1. Generar los artefactos (notebook)

El notebook debe ejecutarse **en orden** de principio a fin para generar el pipeline,
los modelos y las métricas que usa la app:

```bash
jupyter notebook analysis.ipynb
# o con JupyterLab:
jupyter lab analysis.ipynb
```

Ejecutar **Kernel → Restart & Run All**. El proceso completo tarda ~5–15 min
según hardware (entrenamiento MLP incluido).

### 2. Lanzar la aplicación Streamlit

```bash
streamlit run app.py
```

La app se abrirá automáticamente en `http://localhost:8501`.

> Si los artefactos del notebook aún no existen, la app muestra avisos claros
> pero no falla — puedes explorar la interfaz aunque falte algún artefacto.

---

## Protocolo anti-data-leakage

| Etapa | Regla aplicada |
|---|---|
| **Feature selection** | 7 variables excluidas: consecuencia del diagnóstico o constantes |
| **Pipeline fit** | Solo sobre `X_train` (nunca sobre val ni test) |
| **Umbral de decisión** | Optimizado barriendo `X_val`; test bloqueado hasta evaluación final |
| **Evaluación test** | Una única vez por modelo, con el umbral fijado |
| **Pesos de clase** | Calculados solo sobre `y_train` |

---

## Métricas de evaluación

La métrica prioritaria es **Recall(cáncer=1)** — en cribado oncológico,
los falsos negativos (cánceres no detectados) tienen coste clínico mayor.

Orden de prioridad: **Recall > F1 > AUC-PR > AUC-ROC > Accuracy**

La Accuracy no se usa como métrica principal: con prevalencia ≈19 %,
un clasificador trivial (siempre predice 0) alcanza ~81 % sin detectar ningún cáncer.

---

## Aviso ético

> Este proyecto utiliza datos **sintéticos** generados mediante un modelo logístico
> calibrado. Los resultados **no son generalizables** a pacientes reales sin validación
> externa en cohortes clínicas independientes.
>
> La aplicación Streamlit es una herramienta de **demostración académica** y
> **no debe utilizarse para tomar decisiones clínicas reales**.
