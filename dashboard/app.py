"""
Dashboard Streamlit – Olist Revenue Forecast.

Secciones:
  1. KPIs de negocio
  2. Serie temporal histórica + forecast
  3. Análisis de features (importancia, correlaciones)
  4. Métricas del modelo (técnicas y de negocio)
  5. Simulación "What If"

Iniciar:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils.helpers import load_config, load_model, load_dataframe


# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Revenue Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = load_config()


# ─────────────────────────────────────────────────────────────
#  Carga de artefactos (con caché)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        forecaster    = load_model("lgbm_forecaster",   cfg)
        cleaning_pipe = load_model("cleaning_pipeline", cfg)
        return forecaster, cleaning_pipe
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_monthly():
    try:
        return load_dataframe("monthly_features", cfg)
    except FileNotFoundError:
        return None


def run_forecast(horizon: int) -> pd.DataFrame | None:
    """Ejecuta el pipeline de predicción."""
    try:
        from src.pipeline.predict_pipeline import PredictPipeline
        pipe = PredictPipeline(cfg)
        return pipe.run(horizon=horizon)
    except Exception as e:
        st.error(f"Error al generar forecast: {e}")
        return None


@st.cache_data(show_spinner="Calculando valores SHAP…")
def compute_shap(horizon: int = 1):
    """
    Calcula SHAP sobre el histórico mensual limpio.
    Cacheado porque el cálculo es costoso. Retorna (summary_df, local_df, base).
    """
    from src.features.cleaning import clean_monthly_table
    from src.evaluation.shap_analysis import (
        compute_shap_values, shap_summary_df, local_contributions_df,
    )
    forecaster, cleaning_pipe = load_artifacts()
    if forecaster is None or cleaning_pipe is None:
        return None, None, None

    monthly_clean, _ = clean_monthly_table(
        monthly, cfg, fit=False, pipeline=cleaning_pipe
    )
    X = monthly_clean[forecaster.feature_cols_]
    shap_values, fnames, base = compute_shap_values(forecaster, X, horizon=horizon)
    return (
        shap_summary_df(shap_values, fnames),
        local_contributions_df(shap_values, fnames, index=-1),
        base,
    )


# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
    st.title("Olist Forecast")
    st.caption("Revenue Mensual – Series Temporales")
    st.divider()

    section = st.radio(
        "Sección",
        ["📊 Overview", "🔮 Forecast", "🧩 Features", "🔍 SHAP",
         "📐 Métricas", "🎯 What-If"]
    )
    st.divider()
    # El modelo es direct multi-step: solo predice hasta `horizon_months` meses.
    max_h = int(cfg["project"].get("horizon_months", 3))
    horizon = st.slider("Horizonte de predicción (meses)", 1, max_h, max_h)
    st.caption(f"Modelo: **{cfg['models']['final_model']}** (máx. {max_h} meses)")
    st.caption(f"MAPE objetivo: **< {cfg['metrics']['target_mape']}%**")


# ─────────────────────────────────────────────────────────────
#  Cargar datos
# ─────────────────────────────────────────────────────────────
monthly = load_monthly()
forecaster, cleaning_pipe = load_artifacts()

if monthly is None:
    st.error("⚠️ No se encontraron datos. Ejecuta el pipeline de entrenamiento primero.")
    st.code("python -c \"from src.pipeline.train_pipeline import TrainPipeline; TrainPipeline().run(tune=False)\"")
    st.stop()


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 1: Overview
# ─────────────────────────────────────────────────────────────
if section == "📊 Overview":
    st.title("📊 Olist Revenue Forecast – Dashboard")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Revenue Total",
        f"R$ {monthly['monthly_revenue'].sum():,.0f}",
        help="Suma total del período histórico"
    )
    col2.metric(
        "Revenue Promedio Mensual",
        f"R$ {monthly['monthly_revenue'].mean():,.0f}",
        f"Mediana: R$ {monthly['monthly_revenue'].median():,.0f}"
    )
    col3.metric(
        "Meses históricos",
        f"{len(monthly)}",
        help="Meses disponibles para entrenamiento"
    )
    col4.metric(
        "MAPE Objetivo",
        "< 10%",
        help="Objetivo del modelo final"
    )

    st.divider()

    # Serie temporal
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Revenue Mensual (BRL)", "Órdenes Mensuales",
        "Ticket Promedio (BRL)", "Crecimiento MoM (%)"
    ])

    fig.add_trace(go.Scatter(
        x=monthly["ds"], y=monthly["monthly_revenue"],
        mode="lines+markers", name="Revenue",
        line=dict(color="#2196F3", width=2)
    ), row=1, col=1)

    if "monthly_orders" in monthly.columns:
        fig.add_trace(go.Bar(
            x=monthly["ds"], y=monthly["monthly_orders"],
            name="Órdenes", marker_color="#4CAF50"
        ), row=1, col=2)

    if "avg_ticket" in monthly.columns:
        fig.add_trace(go.Scatter(
            x=monthly["ds"], y=monthly["avg_ticket"],
            mode="lines+markers", name="Avg Ticket",
            line=dict(color="#FF9800", width=2)
        ), row=2, col=1)

    if "revenue_mom_growth" in monthly.columns:
        colors = ["#F44336" if v < 0 else "#4CAF50"
                  for v in monthly["revenue_mom_growth"].fillna(0)]
        fig.add_trace(go.Bar(
            x=monthly["ds"],
            y=(monthly["revenue_mom_growth"] * 100).round(2),
            name="MoM %", marker_color=colors
        ), row=2, col=2)

    fig.update_layout(height=500, showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 2: Forecast
# ─────────────────────────────────────────────────────────────
elif section == "🔮 Forecast":
    st.title("🔮 Forecast de Revenue Mensual")

    if forecaster is None:
        st.warning("El modelo no está entrenado. Ejecuta el training pipeline primero.")
    else:
        with st.spinner(f"Generando forecast para {horizon} meses…"):
            forecast = run_forecast(horizon)

        if forecast is not None:
            # Gráfico histórico + forecast
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=monthly["ds"], y=monthly["monthly_revenue"],
                mode="lines+markers", name="Histórico",
                line=dict(color="#2196F3", width=2)
            ))

            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#FF5722", width=2, dash="dash"),
                marker=dict(size=8)
            ))

            # Banda de confianza
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
                y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(255, 87, 34, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Intervalo de confianza (±15%)"
            ))

            fig.update_layout(
                title=f"Forecast de Revenue – Próximos {horizon} Meses",
                xaxis_title="Fecha",
                yaxis_title="Revenue (BRL)",
                height=450, hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabla de forecast
            st.subheader("Tabla de Predicciones")
            forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            forecast_display.columns = ["Mes", "Predicción (BRL)", "Límite Inferior", "Límite Superior"]
            forecast_display["Mes"] = forecast_display["Mes"].dt.strftime("%b %Y")
            for col in ["Predicción (BRL)", "Límite Inferior", "Límite Superior"]:
                forecast_display[col] = forecast_display[col].map("R$ {:,.2f}".format)
            st.dataframe(forecast_display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 3: Features
# ─────────────────────────────────────────────────────────────
elif section == "🧩 Features":
    st.title("🧩 Análisis de Features")

    if forecaster is not None:
        fi_df = forecaster.feature_importance().head(20)

        fig = px.bar(
            fi_df, x="importance", y="feature",
            orientation="h", color="importance",
            color_continuous_scale="Blues",
            title="Top 20 Features – Importancia (LightGBM)"
        )
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Correlación con target
    st.subheader("Correlación con monthly_revenue")
    num_cols = monthly.select_dtypes(include="number").columns.tolist()
    if "monthly_revenue" in num_cols:
        corr = monthly[num_cols].corr()["monthly_revenue"].drop("monthly_revenue")
        corr_df = corr.abs().sort_values(ascending=False).head(20).reset_index()
        corr_df.columns = ["Feature", "Correlación Absoluta"]

        fig2 = px.bar(corr_df, x="Correlación Absoluta", y="Feature",
                      orientation="h", color="Correlación Absoluta",
                      color_continuous_scale="Greens",
                      title="Top 20 Features por Correlación con Target")
        fig2.update_layout(height=450, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 3b: SHAP (explicabilidad)
# ─────────────────────────────────────────────────────────────
elif section == "🔍 SHAP":
    st.title("🔍 Explicabilidad del Modelo (SHAP)")
    st.caption(
        "SHAP cuantifica el aporte real de cada variable a las predicciones "
        "del modelo del horizonte h=1 (predicción a 1 mes)."
    )

    if forecaster is None:
        st.warning("El modelo no está entrenado aún.")
    else:
        try:
            summary_df, local_df, base = compute_shap(horizon=1)
        except Exception as e:
            st.error(f"No se pudo calcular SHAP: {e}")
            summary_df = None

        if summary_df is not None:
            st.subheader("Importancia global (media |SHAP|)")
            top = summary_df.head(20)
            fig = px.bar(
                top, x="mean_abs_shap", y="feature",
                orientation="h", color="mean_abs_shap",
                color_continuous_scale="Purples",
                title="Top 20 Features – Aporte SHAP promedio",
            )
            fig.update_layout(height=520, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Explicación local – último mes observado")
            st.caption(f"Valor base (esperado) del modelo: R$ {base:,.2f}")
            loc = local_df.head(15).copy()
            loc["color"] = loc["shap_value"].apply(
                lambda v: "Aumenta revenue" if v >= 0 else "Reduce revenue"
            )
            fig2 = px.bar(
                loc, x="shap_value", y="feature", orientation="h",
                color="color",
                color_discrete_map={
                    "Aumenta revenue": "#2ca02c",
                    "Reduce revenue": "#d62728",
                },
                title="Contribución de cada feature a la predicción del último mes",
            )
            fig2.update_layout(height=480, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 4: Métricas
# ─────────────────────────────────────────────────────────────
elif section == "📐 Métricas":
    st.title("📐 Métricas del Modelo")

    st.subheader("Modelos Baseline (Sprint 1)")
    baseline_data = {
        "Modelo": ["Naive", "Media Móvil 3M", "Seasonal Naive", "Tendencia Lineal"],
        "RMSE":   [139874, 96804, 691071, 423894],
        "MAPE":   [9.05,   7.12,  50.47,  29.87],
        "Estado": ["⚠️ Base", "✅ Mejor baseline", "❌ Débil", "❌ Débil"],
    }
    st.dataframe(pd.DataFrame(baseline_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Hipótesis Validadas (Sprint 1)")
    hyp_data = {
        "Hipótesis": ["H1 – Estacionalidad", "H2 – Órdenes vs Revenue",
                       "H3 – Pareto categorías", "H4 – Entrega vs Satisfacción",
                       "H5 – Tasa de recompra"],
        "Resultado": ["✅ Confirmada (Nov pico)", "✅ Confirmada (r=0.99)",
                       "✅ Confirmada (9 cats → 80%)", "✅ Confirmada (r=-0.23)",
                       "⚠️ Tasa muy baja (3%)"],
    }
    st.dataframe(pd.DataFrame(hyp_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("KPIs de Negocio")
    kpi_data = {
        "KPI": ["Revenue Total", "Clientes Únicos", "% Entregas a Tiempo",
                 "Review Promedio", "% Clientes Recurrentes", "Tasa Cancelación"],
        "Valor": ["R$ 19,881,945", "93,357", "92.7%", "4.08 / 5.0", "3.0%", "0.63%"],
    }
    st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("💰 Impacto de Negocio / ROI (Sprint 4)")
    try:
        from src.evaluation.metrics import compute_roi_metrics
        fc = run_forecast(horizon)
        roi = compute_roi_metrics(fc, monthly, cfg)

        c1, c2, c3 = st.columns(3)
        c1.metric("Ahorro de tiempo / año",
                  f"R$ {roi['time_saving_brl_per_year']:,.0f}",
                  f"{roi['time_saved_hours_per_year']:.0f} h/año")
        c2.metric("Ahorro por exactitud / año",
                  f"R$ {roi['accuracy_saving_brl_per_year']:,.0f}",
                  f"+{roi['mape_improvement_pts']:.2f} pts MAPE")
        c3.metric("Beneficio total estimado / año",
                  f"R$ {roi['total_benefit_brl_per_year']:,.0f}")

        st.caption(
            f"Revenue proyectado próximos {roi['forecast_horizon_months']} meses: "
            f"**R$ {roi['projected_revenue_horizon']:,.0f}**. "
            "Supuestos editables en `config.yaml → business`."
        )
    except Exception as e:
        st.info(f"ROI no disponible: {e}")


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 5: What-If
# ─────────────────────────────────────────────────────────────
elif section == "🎯 What-If":
    st.title("🎯 Simulación What-If")
    st.write("Ajusta los valores de las features para simular escenarios.")

    if forecaster is None:
        st.warning("El modelo no está entrenado aún.")
    else:
        last_row = monthly[forecaster.feature_cols_].fillna(0).iloc[-1]

        col1, col2 = st.columns(2)
        with col1:
            orders = st.slider(
                "Órdenes mensuales",
                min_value=1000, max_value=10000,
                value=int(last_row.get("monthly_orders", 5000))
                if "monthly_orders" in forecaster.feature_cols_ else 5000
            )
            lag1 = st.number_input(
                "Revenue mes anterior (BRL)",
                value=float(last_row.get("revenue_lag_1", 1200000)),
                step=50000.0,
                format="%.0f"
            )
        with col2:
            avg_ticket = st.slider(
                "Ticket promedio (BRL)",
                min_value=50, max_value=500,
                value=int(last_row.get("avg_ticket", 150))
                if "avg_ticket" in forecaster.feature_cols_ else 150
            )
            review = st.slider(
                "Review score promedio",
                min_value=1.0, max_value=5.0, step=0.1,
                value=float(last_row.get("avg_review_score", 4.0))
                if "avg_review_score" in forecaster.feature_cols_ else 4.0
            )

        if st.button("🔮 Simular", type="primary"):
            input_dict = last_row.to_dict()
            if "monthly_orders"  in input_dict: input_dict["monthly_orders"]  = orders
            if "avg_ticket"      in input_dict: input_dict["avg_ticket"]      = avg_ticket
            if "revenue_lag_1"   in input_dict: input_dict["revenue_lag_1"]   = lag1
            if "avg_review_score" in input_dict: input_dict["avg_review_score"] = review

            X_sim = pd.DataFrame([input_dict])
            y_sim = forecaster.predict(X_sim)

            st.divider()
            st.subheader("Resultado de la Simulación")
            for i, val in enumerate(y_sim, 1):
                st.metric(f"Mes +{i}", f"R$ {val:,.2f}")
