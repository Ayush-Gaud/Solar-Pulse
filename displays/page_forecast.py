"""
page_forecast.py
----------------
Page 4 — Energy Forecasting

Changes from v2 (ARIMA → SARIMAX):
  - SARIMAX replaces ARIMA for daily-horizon forecasting:
      • Captures weekly (S=7) solar seasonality natively.
      • Accepts optional exogenous regressors (daily mean irradiance,
        temperature) via `exog_train` / `exog_future` — builds them
        automatically from power_df when weather columns are present.
      • Grid-searches (p,1,q)(P,1,Q,7) by AIC; falls back to HW.
  - UI label "SARIMAX" replaces "ARIMA" everywhere.
  - Backtest window: 7 days (dataset is ~34 days).
  - History slider capped at min(len(series), 30).
  - 15-min ML panel: weather-aware RF unchanged.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from viz_utils import COLORS, forecast_chart, LAYOUT_BASE

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forecasting import (
    prepare_series, sarimax_forecast, holt_winters_forecast,
    compute_metrics, backtest, build_forecast_table,
    forecast_next_day_15min, backtest_15min, build_forecast_table_15min,
)


def render(data: dict):
    daily_df = data["daily_df"].copy()
    power_df = data["power_df"].copy()   # should include irradiation/module_temp/ambient_temp

    # ── Coerce dtypes ─────────────────────────────────────────────────────────
    daily_df["date"]      = pd.to_datetime(daily_df["date"])
    power_df["timestamp"] = pd.to_datetime(power_df["timestamp"])

    daily_df = daily_df.reset_index(drop=True)
    power_df = power_df.reset_index(drop=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <span class="logo">📈</span>
        <div>
            <h1>Energy & Power Forecasting</h1>
            <span class="sub">Holt-Winters · SARIMAX · Random Forest (15-Min)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    forecast_mode = st.radio(
        "Choose Target Resolution Mode:",
        ["Daily Horizon (Multi-Day)", "15-Minute Resolution Profile (Next-Day)"],
        horizontal=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # DAILY HORIZON
    # ══════════════════════════════════════════════════════════════════════════
    if forecast_mode == "Daily Horizon (Multi-Day)":
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            model_choice = st.selectbox(
                "Forecast Model",
                ["Holt-Winters", "SARIMAX"],
                help="SARIMAX captures weekly solar seasonality (S=7) and can "
                     "exploit  \n daily mean irradiance/temperature as exogenous regressors",
            )
        with col_s2:
            forecast_days = st.slider("Forecast Horizon (days)", 3, 14, 7)
        with col_s3:
            max_hist = max(14, len(daily_df))
            history_days = st.slider(
                "History to Display (days)",
                min_value=7,
                max_value=min(max_hist, 90),
                value=min(30, len(daily_df)),
                help=f"Dataset contains {len(daily_df)} days of history.",
            )

        series = prepare_series(daily_df, "energy_kwh")

        # ── Build daily exog from power_df when weather columns exist ─────
        WEATHER_DAILY_COLS = ["irradiance_wm2", "temperature_c"]
        has_daily_weather = all(c in power_df.columns for c in WEATHER_DAILY_COLS)

        exog_train_daily = None
        exog_future_daily = None

        if has_daily_weather and model_choice == "SARIMAX":
            try:
                # Aggregate 15-min rows → daily mean for each weather variable
                _wx = (
                    power_df.groupby(power_df["timestamp"].dt.date)[WEATHER_DAILY_COLS]
                    .mean()
                    .rename_axis("date")
                    .reset_index()
                )
                _wx["date"] = pd.to_datetime(_wx["date"])
                _wx = _wx.set_index("date").asfreq("D").ffill()

                # Align with the energy series index
                exog_train_daily = _wx.reindex(series.index).ffill().bfill()

                # Naive future exog: repeat the last `forecast_days` rows
                # (In production replace with a weather-API forecast.)
                exog_future_daily = pd.concat(
                    [exog_train_daily.tail(forecast_days)] * 1
                ).reset_index(drop=True)
            except Exception as _ex:
                st.caption(f"⚠️ Could not build daily exog: {_ex}")
                exog_train_daily = None
                exog_future_daily = None

        with st.spinner("Fitting daily model…"):
            if model_choice == "Holt-Winters":
                result = holt_winters_forecast(series, steps=forecast_days)
            else:
                result = sarimax_forecast(
                    series,
                    steps=forecast_days,
                    exog_train=exog_train_daily,
                    exog_future=exog_future_daily,
                )

        forecast_vals = result["forecast"]
        lower_vals    = result["lower"].clip(lower=0)
        upper_vals    = result["upper"]
        model_name    = result["model_name"]
        aic_val       = result["aic"]

        hist_series = series.tail(history_days)

        st.subheader(f"📊 {forecast_days}-Day Macro Forecast Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Model Used",           model_name.split("(")[0].strip())
        k2.metric("AIC Score",            f"{aic_val}" if aic_val else "N/A")
        k3.metric("Forecast Avg (kWh)",   f"{forecast_vals.mean():.1f}")
        k4.metric("Forecast Total (kWh)", f"{forecast_vals.sum():.0f}")

        st.divider()
        st.subheader("Forecast vs Historical Trend")
        fig = forecast_chart(
            dates_hist=hist_series.index.astype(str), actual=hist_series,
            dates_fore=forecast_vals.index.astype(str), forecast=forecast_vals,
            lower=lower_vals, upper=upper_vals,
            title=f"Energy Forecast — {model_name}",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📋 Forecast Values")
        fc_table = build_forecast_table(result)

        col_tbl, col_bar = st.columns([1, 2])
        with col_tbl:
            st.dataframe(fc_table, use_container_width=True, hide_index=True)
        with col_bar:
            avg_fc     = fc_table["Forecast (kWh)"].mean()
            bar_colors = [
                COLORS["primary"] if v >= avg_fc else COLORS["warn"]
                for v in fc_table["Forecast (kWh)"]
            ]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=fc_table["Date"],
                y=fc_table["Forecast (kWh)"],
                marker_color=bar_colors,
                error_y=dict(
                    type="data",
                    array=(fc_table["Upper CI"] - fc_table["Forecast (kWh)"]).tolist(),
                    arrayminus=(fc_table["Forecast (kWh)"] - fc_table["Lower CI"]).tolist(),
                    visible=True,
                    color=COLORS["muted"],
                ),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=320, showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Back-test (7-day hold-out) ─────────────────────────────────────
        st.divider()
        st.subheader("🧪 Back-test Evaluation")
        # st.caption(
        #     "Hold-out window reduced from 14 → 7 days because the dataset "
        #     "contains ~34 days of history; a 14-day hold-out left too few "
        #     "training points for reliable model estimation."
        # )

        bt_method = "holt_winters" if model_choice == "Holt-Winters" else "sarimax"
        bt = backtest(series, test_days=7, method=bt_method)
        m  = bt["metrics"]

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{m['MAE']} kWh" if m["MAE"]  else "N/A")
        m2.metric("RMSE", f"{m['RMSE']} kWh" if m["RMSE"] else "N/A")
        m3.metric("MAPE", f"{m['MAPE']}%"    if m["MAPE"] else "N/A")

    # ══════════════════════════════════════════════════════════════════════════
    # 15-MINUTE RESOLUTION
    # ══════════════════════════════════════════════════════════════════════════
    else:
        # Check whether weather columns are available in power_df
        has_weather = all(
            c in power_df.columns
            for c in ["irradiance_wm2", "module_temp_c", "temperature_c"] # Updated names
        )

        if has_weather:
            st.info(
                "**Weather-Aware ML Engine Active:** A Random Forest trained on "
                "irradiation (r=0.996), module temperature (r=0.961), ambient "
                "temperature, cyclical time encodings, and a 24-h generation lag."
            )
        else:
            st.warning(
                "⚠️ Weather columns (irradiation, module_temp, ambient_temp) not "
                "found in power_df. Falling back to time-feature-only RF. Merge "
                "the weather sensor CSV in your data loader to unlock the full "
                "accuracy improvement."
            )

        with st.spinner("Processing 15-minute arrays and fitting Random Forest…"):
            result_15min = forecast_next_day_15min(power_df)
            bt_15min     = backtest_15min(power_df, test_days=1)

        fc_vals_15m  = result_15min["forecast"]
        low_vals_15m = result_15min["lower"]
        up_vals_15m  = result_15min["upper"]

        # ── Summary metrics ────────────────────────────────────────────────
        st.subheader("⏱️ Tomorrow's 15-Minute Generation Metrics")
        km1, km2, km3 = st.columns([1,1,1])
        km1.metric("Model Architecture",      "Random Forest")
        km2.metric("Predicted Peak Power",    f"{fc_vals_15m.max():.1f} kW")
        km3.metric("Total Integrated Energy", f"{(fc_vals_15m.sum() * 0.25):.1f} kWh")

        st.divider()

        # ── Forecast curve ────────────────────────────────────────────────
        st.subheader("Predicted Power Output Curve (96 Slots of Tomorrow)")
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=fc_vals_15m.index, y=up_vals_15m, mode="lines",
            line=dict(width=0), showlegend=False, name="Upper CI",
        ))
        fig_curve.add_trace(go.Scatter(
            x=fc_vals_15m.index, y=low_vals_15m, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(82,183,136,0.15)", name="Confidence Envelope",
        ))
        fig_curve.add_trace(go.Scatter(
            x=fc_vals_15m.index, y=fc_vals_15m, mode="lines",
            line=dict(color=COLORS["primary"], width=3),
            name="Power Forecast (kW)",
        ))
        fig_curve.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Time Horizon (Tomorrow)", showgrid=True, gridcolor=COLORS["grid"]),
            yaxis=dict(title="Power Output (kW)",       showgrid=True, gridcolor=COLORS["grid"]),
            height=400, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        st.divider()

        # ── Table + backtest ───────────────────────────────────────────────
        st.subheader("📋 Detailed Time Slots & Hold-Out Validation")
        c_left, c_right = st.columns([1, 2])

        with c_left:
            st.caption("Tomorrow's 15-Min Intervals")
            tbl_15 = build_forecast_table_15min(result_15min)
            st.dataframe(tbl_15, use_container_width=True, hide_index=True, height=350)

        with c_right:
            st.caption("Out-Of-Sample Backtest — Last 24 Hours")
            m_15 = bt_15min["metrics"]
            mb1, mb2, mb3 = st.columns(3)
            mb1.metric("MAE",  f"{m_15['MAE']} kW")
            mb2.metric("RMSE", f"{m_15['RMSE']} kW")
            mb3.metric("MAPE", f"{m_15['MAPE']}%")

            fig_bt15 = go.Figure()
            fig_bt15.add_trace(go.Scatter(
                x=bt_15min["actual"].index,    y=bt_15min["actual"],
                mode="lines", name="Actual Power",
                line=dict(color=COLORS["primary"]),
            ))
            fig_bt15.add_trace(go.Scatter(
                x=bt_15min["predicted"].index, y=bt_15min["predicted"],
                mode="lines", name="Predicted Power",
                line=dict(color=COLORS["warn"], dash="dot"),
            ))
            fig_bt15.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=280, margin=dict(l=40, r=20, t=10, b=40),
            )
            st.plotly_chart(fig_bt15, use_container_width=True)
