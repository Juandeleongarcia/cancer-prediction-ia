# Guion del Presentador — 5 Slides · ~45-60 segundos por slide
## Predicción de Diagnóstico de Cáncer con IA · UAX 2025-2026

---

> **Instrucciones de uso:**
> Cada bloque está calibrado para 45–60 segundos de presentación oral.
> Los fragmentos entre [corchetes] se rellenan con los valores reales tras ejecutar el notebook.
> Habla despacio, pausa en los números clave, mira al público en las frases en negrita.

---

## SLIDE 1 — Objetivo y Datos (45 seg.)

**Apertura impactante:**
"Este proyecto responde a una pregunta clínica concreta: ¿puede un sistema de IA detectar un
diagnóstico de cáncer antes de que lo haga el proceso clínico habitual?"

**Contexto del dataset:**
"Trabajamos con 50.001 pacientes sintéticos, distribuidos en seis colecciones de datos que
cubren desde marcadores bioquímicos hasta mutaciones genéticas y factores socioeconómicos.
La variable objetivo tiene una prevalencia de solo el 19 por ciento — un desbalance de 4.3 a 1
que obliga a un tratamiento metodológico riguroso."

**Pipeline y anti-leakage:**
"Todo el flujo parte de seis archivos CSV que se unen por ID de paciente, pasando por una
auditoría explícita de data leakage. Siete variables fueron excluidas del modelo porque son
consecuencias del diagnóstico, no predictores: los costes hospitalarios, los días de ingreso
o si el paciente sobrevivió. Incluirlas habría inflado artificialmente las métricas."

**Cierre:**
"El pipeline de preprocessing se ajustó exclusivamente sobre el conjunto de entrenamiento.
El test set nunca se tocó hasta la evaluación final."

---

## SLIDE 2 — Resultados ML Clásicos (55 seg.)

**Apertura:**
"Entrenamos cinco modelos clásicos de Machine Learning: Regresión Logística como baseline,
Random Forest, XGBoost, LightGBM y CatBoost."

**Explicación de métricas:**
"La métrica prioritaria es el Recall — en cribado oncológico, un falso negativo, un cáncer
no detectado, tiene un coste clínico inaceptable. Por eso ordenamos los modelos por Recall,
no por accuracy. Con un 19 por ciento de prevalencia, un clasificador que siempre predice
'sin cáncer' alcanza el 81 por ciento de accuracy sin detectar un solo caso positivo."

**Umbral de decisión:**
"Cada modelo tiene un umbral de decisión optimizado en validación, nunca en test. Esto es
crítico: si optimizas el umbral en test, estás ajustando el modelo a datos que luego presentas
como 'no vistos', lo que invalida la evaluación."

**Resultado:**
"El mejor modelo clásico es [nombre], con un Recall de [valor], F1 de [valor] y AUC-ROC
de [valor], con un umbral óptimo de [valor]. Las curvas ROC muestran que todos los boosters
superan claramente a la regresión logística."

**Cierre:**
"Este es nuestro baseline fuerte antes de comparar con la red neuronal."

---

## SLIDE 3 — Red Neuronal MLP (55 seg.)

**Apertura:**
"La red neuronal MLP tiene una arquitectura de tres capas ocultas: 256, 128 y 64 neuronas,
con activación ReLU. La capa de salida usa sigmoid para obtener una probabilidad directa
de que el paciente tenga cáncer."

**Regularización:**
"Aplicamos cuatro mecanismos de regularización: BatchNormalization entre capas para
estabilizar el entrenamiento, Dropout para prevenir sobreajuste, EarlyStopping con paciencia
de 12 épocas que restaura los mejores pesos, y ReduceLROnPlateau que reduce el learning
rate cuando la pérdida de validación se estanca. Además, usamos class_weight para compensar
el desbalance de clases."

**Comparativa:**
"En el test set, la MLP obtiene un Recall de [valor] frente al [valor] del mejor modelo
clásico. [Ajusta según resultado: 'La red neuronal supera al mejor ML en Recall por X puntos'
o 'El rendimiento es comparable, con diferencia inferior al 2 por ciento'.]"

**Conclusión:**
"La red neuronal añade complejidad operacional — requiere TensorFlow, mayor tiempo de
entrenamiento y es una caja negra sin interpretabilidad directa. La diferencia de rendimiento
debe justificar ese coste en un entorno hospitalario."

---

## SLIDE 4 — Comparativa Global (50 seg.)

**Apertura:**
"El ranking final combina cuatro métricas con pesos clínicos: el 35 por ciento va al Recall,
el 30 al F1, el 20 al AUC-PR y el 15 al AUC-ROC. Esta ponderación refleja que en cribado
oncológico, minimizar los falsos negativos es la prioridad número uno."

**Lectura del ranking:**
"El modelo ganador es [nombre] con un score compuesto de [valor]. [Segundo y tercero].
La Regresión Logística, como baseline, cierra el ranking pero sirve como referencia de
cuánto aportan los modelos más complejos."

**Precision-Recall:**
"El gráfico de la derecha sitúa cada modelo en el espacio Precision-Recall. La línea roja
marca el umbral clínico de Recall superior a 0.70 — los modelos que no cruzan esa línea
no son candidatos para producción en cribado oncológico."

**Interpretabilidad:**
"Un punto crítico: los boosters como LightGBM y CatBoost ofrecen SHAP nativo — el clínico
puede ver qué variables impulsaron cada predicción individual. La MLP, sin XAI externo,
no puede dar esa explicación."

---

## SLIDE 5 — Viabilidad y Decisión (60 seg.)

**Apertura:**
"La pregunta final no es cuál modelo es mejor en métricas — es si el sistema es viable
en un hospital universitario real."

**Modelo recomendado:**
"Recomendamos [nombre del ganador] con un Recall de [valor], F1 de [valor] y umbral de [valor].
Es desplegable en Python con sklearn, interpretable con SHAP y tiene tiempos de inferencia
apropiados para uso clínico en tiempo real."

**Veredicto:**
"El sistema es viable como herramienta de soporte al cribado clínico de primer nivel.
No es un diagnóstico — es una señal de alerta que el clínico evalúa. Un resultado positivo
del modelo debe desencadenar un proceso de confirmación diagnóstica, no una decisión clínica
directa."

**Limitaciones honestas:**
"Tenemos tres limitaciones que debemos nombrar explícitamente: los datos son sintéticos y
el modelo no es generalizable sin validación en cohorte real; el Recall por debajo del 100 por
ciento implica cánceres no detectados, lo que es un riesgo clínico conocido; y existe posible
sesgo si la distribución del dataset no representa a la población objetivo del hospital."

**Cierre con demo:**
"Para cerrar: hemos desarrollado una aplicación Streamlit interactiva donde cualquier
clínico puede introducir los parámetros de un paciente y obtener la probabilidad de cáncer
en tiempo real, junto con el ranking de modelos y todas las visualizaciones del estudio.
El siguiente paso es llevar este prototipo a un entorno con datos reales."

---

## Timing total recomendado

| Slide | Duración | Enfoque |
|-------|----------|---------|
| 1 — Objetivo y datos | 45 seg. | Contexto y rigor metodológico |
| 2 — Resultados ML | 55 seg. | Números clave + explicar Recall |
| 3 — Red Neuronal MLP | 55 seg. | Arquitectura + comparativa directa |
| 4 — Comparativa global | 50 seg. | Ranking + interpretabilidad |
| 5 — Viabilidad | 60 seg. | Decisión ejecutiva + demo |
| **Total** | **~4:45** | + 1:15 min para preguntas |

---

## Preguntas frecuentes — respuestas preparadas

**"¿Por qué datos sintéticos y no reales?"**
> El dataset real de pacientes oncológicos requeriría aprobación del comité ético, anonimización
> y acuerdos de uso con el hospital. Para el alcance académico, el dataset sintético generado
> con un modelo logístico calibrado clínicamente permite demostrar el pipeline completo con
> total rigor metodológico.

**"¿Por qué Recall como métrica principal y no F1?"**
> En cribado oncológico, el coste asimétrico de los errores justifica priorizar Recall.
> Un falso negativo (cáncer no detectado) puede significar un diagnóstico tardío con
> pronóstico peor. Un falso positivo genera una prueba de confirmación adicional — coste
> económico y emocional, pero reversible. El F1 equilibra ambos; el Recall prioriza la
> sensibilidad.

**"¿Por qué no usar solo la MLP si es más potente?"**
> En un entorno hospitalario real, la interpretabilidad es un requisito regulatorio y ético,
> no un deseo. El clínico debe poder explicar al paciente y al comité médico por qué el
> sistema marcó un caso como positivo. LightGBM con SHAP cumple ese requisito. La MLP
> requeriría implementar una capa de XAI antes de ser considerada para producción.

**"¿Qué pasaría con datos reales?"**
> Esperamos degradación de rendimiento por ruido real, distribuciones distintas y variables
> faltantes. El pipeline está diseñado para ser re-entrenado con nuevos datos. El protocolo
> anti-leakage y la estructura de validación son robustos para datos reales.
