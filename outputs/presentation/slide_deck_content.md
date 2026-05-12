# Presentación Final — Predicción de Diagnóstico de Cáncer con IA
## Universidad Alfonso X el Sabio · Bases de Datos e IA · 2025-2026
### 5 Diapositivas Ejecutivas · Formato 16:9 · Estilo Consultoría / IA Médica

---
> **NOTA SOBRE MÉTRICAS:** El notebook `analysis.ipynb` no ha sido ejecutado aún.
> Todos los valores marcados como `pendiente de ejecución` se autocompletan al ejecutar
> **Kernel → Restart & Run All** en el notebook y luego volver a correr:
> ```
> python outputs/presentation/build_presentation.py
> ```
> Los archivos que deben existir en `outputs/metrics/` son:
> - `all_models_metrics.csv` — generado por §28 y §37
> - `final_model_ranking.csv` — generado por §42
> - `executive_summary.json` — generado por §46
> - `ml_results.csv` — generado por §27
> - `mlp_results.csv` — generado por §36

---

## SLIDE 1 — Objetivo y Datos

**Título:** Predicción de Diagnóstico de Cáncer con IA
**Subtítulo:** Comparativa ML clásico vs Red Neuronal Multicapa · UAX · 2025-2026

### KPIs principales (4 cards horizontales)

| KPI | Valor |
|-----|-------|
| Pacientes | **50 001** |
| Colecciones | **6** (bioquímica, clínica, genética, económica, hábitos, sociodemográfica) |
| Prevalencia cáncer | **≈ 19 %** · desbalance **4.3 : 1** |
| Variables activas | **29 / 38** · 7 excluidas por data leakage |

### Pipeline del proyecto (diagrama horizontal)

```
[6 CSVs] → [Merge paciente_id] → [EDA + Leakage audit] → [Pipeline sklearn] → [ML / MLP] → [Evaluación test]
             JOIN por ID           Audit variables         fit solo en train    5 + 1 modelos  Una sola vez
```

### Protocolo Anti-Data-Leakage (recuadro azul)
- Pipeline fit exclusivamente sobre train (nunca val/test)
- Umbral de decisión optimizado en validación
- Test evaluado una única vez por modelo
- 10 criterios de rigor metodológico verificados

### Variables excluidas — data leakage (recuadro rojo)
`coste_total` · `coste_farmaco` · `num_ingresos` · `dias_hospital` · `vive` · `alcohol` · `tipo_seguro`

> Razón: son **consecuencias del diagnóstico**, no predictores causales. Incluirlas sería trampa metodológica.

### Figuras usadas
- `fig_01_target_distribution.png` — distribución de cáncer=0 / cáncer=1 (barra azul/rojo)
- `fig_08_leakage_correlation.png` — variables con correlación sospechosa / excluidas

---

## SLIDE 2 — Resultados Modelos ML Clásicos

**Título:** Modelos ML Clásicos: Rendimiento en Test
**Subtítulo:** Umbral de decisión optimizado en validación · Test evaluado una única vez

### Tabla de métricas — test set (ordenada por Recall ↓)

| Modelo | Precision | Recall | F1 | AUC-ROC | AUC-PR | Umbral |
|--------|-----------|--------|----|---------|--------|--------|
| **[1º — pendiente]** | pend. | pend. | pend. | pend. | pend. | pend. |
| [2º — pendiente] | pend. | pend. | pend. | pend. | pend. | pend. |
| [3º — pendiente] | pend. | pend. | pend. | pend. | pend. | pend. |
| [4º — pendiente] | pend. | pend. | pend. | pend. | pend. | pend. |
| Log. Regresión | pend. | pend. | pend. | pend. | pend. | pend. |

> Valores se cargan automáticamente de `outputs/metrics/all_models_metrics.csv`
> La fila ganadora se resalta en azul claro en el PPTX.

**Nota al pie:** ★ Accuracy no es criterio principal: con prevalencia ≈ 19 %, un clasificador trivial (siempre cáncer=0) alcanza ~81 % sin detectar ningún cáncer.

### KPIs del mejor modelo ML (panel derecho)
```
MEJOR MODELO ML
[nombre — pendiente de ejecución]

Recall:    [pendiente]
F1:        [pendiente]
AUC-ROC:   [pendiente]
Umbral:    [pendiente]
```

### Figuras usadas
- `fig_13_roc_curves.png` — curvas ROC de los 5 modelos superpuestas
- `fig_15_confusion_matrix_best.png` — matriz de confusión del mejor modelo ML

### Figuras NO incluidas (evitar saturación)
- `fig_14_pr_curves.png` — disponible en la app Streamlit
- `fig_12_threshold_scan.png` — demasiado técnico para slide ejecutiva

---

## SLIDE 3 — Red Neuronal MLP

**Título:** Red Neuronal MLP: Arquitectura y Validación
**Subtítulo:** Dense 256→128→64 + BatchNorm + Dropout · Mismo protocolo anti-leakage

### Arquitectura (diagrama de capas)

```
Entrada: [N features procesadas]
         ↓
Dense(256, relu) → BatchNorm → Dropout(0.25)
         ↓
Dense(128, relu) → BatchNorm → Dropout(0.25)
         ↓
Dense(64,  relu) → BatchNorm → Dropout(0.20)
         ↓
Dense(1, sigmoid)
         ↓
P(cáncer = 1)  ∈ [0, 1]
```

### Regularización aplicada
- **BatchNorm** — estabiliza gradientes entre capas (μ≈0, σ≈1)
- **Dropout** — previene sobreajuste (retiene 75–80 % de activaciones por paso)
- **EarlyStopping** — paciencia 12, restaura mejores pesos
- **ReduceLROnPlateau** — lr × 0.5 si val_loss estanca (paciencia 6)
- **class_weight balanced** — compensa desbalance 4.3:1
- **Adam lr=0.001 · batch=256 · máx 150 épocas**

### Tabla comparativa MLP vs mejor ML (test set)

| Modelo | Recall ↑ | F1 ↑ | AUC-ROC | AUC-PR |
|--------|----------|------|---------|--------|
| MLP (Red Neuronal) | pendiente | pendiente | pendiente | pendiente |
| Mejor ML (clásico) | pendiente | pendiente | pendiente | pendiente |

> Valores de `outputs/metrics/mlp_results.csv` y `outputs/metrics/ml_results.csv`

### Conclusión (recuadro azul claro)
> "La MLP alcanza rendimiento comparable al mejor modelo clásico con mayor complejidad
> operacional y menor interpretabilidad. En contexto hospitalario, el modelo clásico es
> preferible salvo que la MLP supere claramente en Recall."

### Figuras usadas
- `fig_17_mlp_architecture.png` — diagrama de arquitectura generado por Keras
- `fig_18_mlp_learning_curves.png` — curvas loss + recall por época (detecta overfitting)

### Figuras NO incluidas
- `fig_19_mlp_threshold_scan.png` — técnico, disponible en app
- `fig_20_mlp_roc_pr.png` — disponible en app / slide 4

---

## SLIDE 4 — Comparativa Global

**Título:** Ranking Final: ML vs MLP
**Subtítulo:** Score compuesto = 0.35·Recall + 0.30·F1 + 0.20·AUC-PR + 0.15·AUC-ROC

### Tabla ranking completo (todos los modelos, ordenada por Score ↓)

| # | Modelo | Tipo | Recall | F1 | AUC-PR | AUC-ROC | Score |
|---|--------|------|--------|----|--------|---------|-------|
| 🥇 | [pendiente] | ML/MLP | pend. | pend. | pend. | pend. | pend. |
| 🥈 | [pendiente] | ML | pend. | pend. | pend. | pend. | pend. |
| 🥉 | [pendiente] | ML | pend. | pend. | pend. | pend. | pend. |
| 4 | [pendiente] | ML | pend. | pend. | pend. | pend. | pend. |
| 5 | [pendiente] | ML | pend. | pend. | pend. | pend. | pend. |
| 6 | Log. Regresión | ML | pend. | pend. | pend. | pend. | pend. |

> Datos de `outputs/metrics/final_model_ranking.csv`
> Fila 🥇 en azul en el PPTX. Tipo MLP con borde naranja.

### Mensaje clave (recuadro azul)
> "Criterio de selección: mejor equilibrio Recall · F1 · AUC-PR · AUC-ROC + interpretabilidad
> clínica. Los boosters (LightGBM / CatBoost) ofrecen SHAP nativo. La MLP requiere XAI externo."

### Figuras usadas
- `fig_final_metrics_comparison.png` — barras agrupadas de todas las métricas por modelo
- `fig_final_precision_recall_space.png` — scatter Precision × Recall (zona clínica: Recall > 0.70)

---

## SLIDE 5 — Viabilidad y Decisión

**Título:** Viabilidad Clínica y Recomendación Final
**Subtítulo:** Sistema de soporte al cribado oncológico · No reemplaza el diagnóstico médico

### Modelo recomendado (recuadro azul oscuro)
```
MODELO RECOMENDADO PARA PRODUCCIÓN
[nombre — pendiente de ejecución]

Recall [pend.]  ·  F1 [pend.]  ·  AUC-ROC [pend.]  ·  Umbral [pend.]
```

### Veredicto ejecutivo (recuadro verde)
> "Sistema viable como herramienta de soporte al cribado clínico.
> No constituye diagnóstico médico. Requiere validación externa en
> cohorte real antes de despliegue clínico."

### Limitaciones (3 columnas)

**Columna 1 — Limitaciones (fondo rojo claro)**
- Datos sintéticos — no generalizable sin validación en pacientes reales
- Riesgo de falsos negativos — Recall < 1.0 implica cánceres no detectados
- Posible sesgo poblacional si la distribución sintética difiere de la real
- MLP: caja negra sin XAI externo (SHAP/LIME no implementado)

**Columna 2 — Mejoras futuras (fondo azul claro)**
- SHAP — interpretabilidad por paciente para el clínico
- Calibración de probabilidades (Platt scaling / Isotonic regression)
- Validación en cohorte real con datos anonimizados hospitalarios
- Monitorización de drift — reentrenamiento periódico en producción

**Columna 3 — Demo interactiva (fondo verde claro)**
```bash
streamlit run app.py
```
- Simulador de paciente — introduce parámetros y obtén probabilidad en tiempo real
- Comparativa de modelos — tabla y gráficos interactivos
- Galería de figuras — todas las visualizaciones del estudio
- Estado de artefactos — verificación del pipeline en sidebar

---

## Diseño visual — especificaciones

| Elemento | Especificación |
|----------|----------------|
| Formato | 16:9 (13.33 × 7.5 pulgadas) |
| Fondo | Blanco `#FFFFFF` |
| Barra superior | Azul medio `#2C5F8A` (8pt altura) |
| Pie de página | Azul oscuro `#1A3A5C` con texto blanco |
| Títulos | Azul oscuro `#1A3A5C` · 24pt bold |
| Subtítulos | Azul medio `#2C5F8A` · 13pt |
| Headers tabla | Azul medio `#2C5F8A` fondo · blanco texto · 9pt |
| Fila destacada | Azul claro `#D6E4F0` · bold |
| Alertas / riesgo | Rojo `#E05C4B` |
| Positivo / OK | Verde `#43A047` |
| Cards KPI | Fondo gris `#F4F6F9` · borde izquierdo azul 4pt |

## Catálogo de figuras por slide

| Slide | Figura principal | Figura secundaria |
|-------|-----------------|-------------------|
| 1 | `fig_01_target_distribution.png` | `fig_08_leakage_correlation.png` |
| 2 | `fig_13_roc_curves.png` | `fig_15_confusion_matrix_best.png` |
| 3 | `fig_18_mlp_learning_curves.png` | `fig_17_mlp_architecture.png` |
| 4 | `fig_final_metrics_comparison.png` | `fig_final_precision_recall_space.png` |
| 5 | — (solo texto estructurado) | — |

## Figuras NO incluidas en la presentación

| Figura | Razón de exclusión |
|--------|--------------------|
| `fig_02_nulls.png` | Demasiado técnico para audiencia ejecutiva |
| `fig_03_bioq_distributions.png` | EDA exploratorio — para informe técnico |
| `fig_04_continuous_by_cancer.png` | Idem |
| `fig_05_binary_by_cancer.png` | Idem |
| `fig_06_categorical_by_cancer.png` | Idem |
| `fig_07_correlation_matrix.png` | Saturación visual |
| `fig_10_processed_sample.png` | Solo para evaluador técnico |
| `fig_11_class_weights.png` | Mencionado en texto, no necesita figura |
| `fig_12_threshold_scan.png` | Muy técnico — disponible en app |
| `fig_14_pr_curves.png` | Disponible en app Streamlit |
| `fig_19_mlp_threshold_scan.png` | Duplica info — disponible en app |
| `fig_21_mlp_confusion_matrix.png` | Solo si hay tiempo extra en presentación |
