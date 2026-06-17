# Ética, Sesgos, Privacidad y Gobernanza

Sprint 4 · Olist Revenue Forecast (`monthly_revenue`)

Este documento evalúa el impacto ético del modelo de forecast de ingresos
mensuales y define el marco de gobernanza para su operación mensual.

---

## 1. Sesgos y limitaciones

| Sesgo / limitación | Descripción | Mitigación |
|---|---|---|
| **Agregación global** | El target es el revenue mensual agregado de todo el e-commerce. Esto oculta inequidades por estado, categoría y seller: un buen MAPE global puede convivir con errores grandes en segmentos pequeños. | Forecast estratificado por categoría/región como extensión; monitorear residuos por segmento. |
| **Cobertura temporal corta** | Datos de **ene-2017 a ago-2018** (~31 meses tras descartar meses incompletos). Solo ~1.5 ciclos anuales → la estacionalidad (pico de Nov/Black Friday) está poco muestreada. | Tratar el forecast >3 meses con cautela; ampliar histórico cuando esté disponible. |
| **Sesgo de supervivencia / filtro** | El revenue se calcula solo sobre órdenes `delivered` con `payment_value > 0`. Cancelaciones y órdenes no entregadas quedan fuera. | Documentado explícitamente; coherente con "revenue realizado", no "revenue bruto". |
| **Correlación entrega ↔ satisfacción** | `review_score` correlaciona con retrasos de entrega; no son variables independientes. | No interpretar coeficientes/SHAP como causalidad; usar SHAP solo como atribución. |
| **Dependencia de lags** | El modelo se apoya fuertemente en `revenue_lag_1` y medias móviles. Un shock atípico (p. ej. promoción única) se propaga a los meses siguientes. | Intervalos de predicción basados en residuos (factor de inflación 1.5) comunican la incertidumbre. |
| **Meses incompletos** | El primer y último mes se descartan por estar parcialmente poblados. | Implementado en `build_monthly_table` (`monthly.iloc[1:-1]`). |

---

## 2. Privacidad y manejo de datos

- **Dataset anonimizado en origen.** Olist publica el dataset con IDs sustituidos
  (customer, seller, order) y nombres reales reemplazados; no contiene PII directa
  (nombres, emails, documentos). Las reseñas de texto no se usan en el modelo.
- **Solo agregados mensuales.** El modelo final entrena sobre la **tabla mensual
  agregada** (`monthly_features.parquet`): no consume datos a nivel de individuo,
  por lo que la inferencia no expone información de clientes concretos.
- **Geolocalización.** El dataset de geolocalización se carga pero el modelo final
  usa, a lo sumo, agregados por estado; no se hace tracking individual.
- **Recomendaciones operativas:**
  - No reintroducir PII en futuras features sin base legal (LGPD en Brasil / análogos).
  - Mantener los CSVs crudos fuera de las imágenes de servicio (ver `.dockerignore`).
  - Restringir el acceso a `data/` y a los artefactos `.pkl` en despliegue.

---

## 3. Gobernanza del modelo

### Trazabilidad y versionado
- **MLflow** registra cada entrenamiento con tags (`version`, `model_type`),
  hiperparámetros, métricas (técnicas y de negocio) y artefactos (`.pkl` + lista de
  features). Ver `docs/MLFLOW.md`.
- Las **versiones de librerías** quedan en `final_model.pkl → metadata['versions']`
  (los `.pkl` son sensibles a la versión de scikit-learn/LightGBM).

### Criterio de aprobación (gate de promoción)
Un modelo solo se promueve a producción si:
1. **MAPE de backtest < 10 %** (`config.yaml → metrics.target_mape`).
2. **No degrada** respecto al campeón vigente (comparación en MLflow UI).
3. La selección de features pasó los filtros de estabilidad (PSI, correlación, RFECV).

### Monitoreo de drift
- El **PSI** (umbral 0.25, `src/features/selection.py`) detecta cambios de
  distribución entre train y datos recientes. Un PSI alto dispara reentrenamiento
  completo (flujo B en `docs/MLFLOW.md`).
- Seguimiento mensual del **MAPE real** una vez observado el revenue del mes.

### Explicabilidad
- **SHAP** (`src/evaluation/shap_analysis.py`, endpoint `/api/v1/shap`, pestaña del
  dashboard) expone el aporte de cada variable, tanto global como local. Permite
  auditar que el modelo se apoya en señales razonables (lags, órdenes, ticket) y no
  en artefactos espurios.

### Roles y responsabilidades (propuesta)
| Rol | Responsabilidad |
|---|---|
| Data/ML Engineer | Ejecuta el retraining mensual, revisa drift, mantiene el pipeline. |
| Analista de negocio | Valida coherencia del forecast y los KPIs de ROI. |
| Owner del modelo | Aprueba la promoción del modelo (gate) y custodia la gobernanza. |

---

## 4. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Sobre-confianza en el forecast | Decisiones de inventario/caja erróneas | Comunicar siempre el **intervalo** `[yhat_lower, yhat_upper]`, no solo el punto. |
| Deriva estacional no aprendida | Subestimar picos (Black Friday) | Reentrenamiento completo trimestral; ampliar histórico. |
| Dependencia de un único modelo | Punto único de fallo | Baselines y candidatos (Prophet, SARIMA) disponibles como respaldo. |
| Uso del modelo fuera de contexto | Aplicarlo a segmentos no validados | Documentar alcance: revenue **global** mensual, no por seller/categoría. |
| Degradación silenciosa | Pérdida de exactitud no detectada | Monitoreo de MAPE real + PSI; alertas cuando se cruce el umbral. |

---

## 5. Conclusión

El modelo opera sobre datos anonimizados y agregados, con bajo riesgo de privacidad.
Los principales riesgos son de **representatividad** (histórico corto, agregación
global) y de **interpretación** (no causalidad, incertidumbre). El marco de
gobernanza —gate de MAPE, versionado MLflow, monitoreo PSI y explicabilidad SHAP—
mantiene el sistema auditable y seguro para su uso mensual en decisiones de negocio.
