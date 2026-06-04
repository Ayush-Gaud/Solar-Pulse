"""
page_overview.py
----------------
Page 1 — Overview Dashboard

Changes from synthetic version:
  - cloud_cover_pct removed (was a synthetic bell-curve proxy, not real data).
  - dc_ac_efficiency_pct and performance_ratio added — both derived from real Kaggle columns.
  - Gauge max_val is data-driven.
  - No synthesised data anywhere.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from viz_utils import COLORS, line_chart, gauge_chart, kpi_values


def render(data: dict):
    power_df = data["power_df"]
    daily_df = data["daily_df"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <span class="logo">🏠</span>
        <div>
            <h1>Overview Dashboard</h1>
            <span class="sub">Real Generation & Weather Sensor Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Date selector ─────────────────────────────────────────────────────────
    all_dates     = sorted(daily_df["date"].unique())
    selected_date = st.date_input(
        "Select date",
        value=all_dates[-1],
        min_value=all_dates[0],
        max_value=all_dates[-1],
    )

    today_daily  = daily_df[daily_df["date"] == selected_date]
    today_hourly = power_df[power_df["date"] == selected_date]

    if today_daily.empty:
        st.warning("No data for selected date.")
        return

    row = today_daily.iloc[0]

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators")
    c1, c2, c3, c4 = st.columns([1.7, 1.6, 1.15, 1.35])
    c1.metric("⚡ Peak AC Power",   f"{row['peak_power_kw']:.1f} kW")
    c2.metric("🔋 Daily AC Energy", f"{row['energy_kwh']:.0f} kWh")
    c3.metric("📊 Capacity Factor", f"{row['capacity_factor_pct']:.1f}%", help="AC energy / (nameplate × 24 h) × 100")
    c4.metric("🌿 CO₂ Saved",       f"{row['co2_saved_kg']:.0f} kg")

    c5, c6, d2, d3 = st.columns([1.7, 1.6, 1.15, 1.35])
    c5.metric("🌡️ Avg Amb. Temp",   f"{row['avg_temp']:.1f} °C")
    c6.metric("☀️ Avg Irradiance",  f"{row['avg_irradiance']:.3f} W/m²")
    # Second row: real derived metrics
    # if "performance_ratio" in row and pd.notna(row.get("performance_ratio")):
    #     d1.metric("📐 Performance Ratio",
    #               f"{row['performance_ratio']:.3f}",
    #               help="Standard IEC metric: AC energy / (irradiance × capacity)")
    if "avg_module_temp" in row and pd.notna(row.get("avg_module_temp")):
        d2.metric("🌡️ Avg Module Temp",
                  f"{row['avg_module_temp']:.1f} °C",
                  help="Real sensor reading from Weather CSV")
    if "dc_ac_efficiency_pct" in row and pd.notna(row.get("dc_ac_efficiency_pct")):
        d3.metric("🔌 DC→AC Efficiency",
                  f"{row['dc_ac_efficiency_pct']:.1f}%",
                  help="AC energy / DC energy × 100 (real inverter measurement)")

    st.divider()

    # ── Power Generation Profile ──────────────────────────────────────────────
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("Power Generation Profile")
        if today_hourly.empty:
            st.info("No 15-min data for this date.")
        else:
            fig = go.Figure()
            # AC power (primary axis)
            fig.add_trace(go.Scatter(
                x=today_hourly["hour"], y=today_hourly["power_kw"],
                mode="lines", fill="tozeroy",
                line=dict(color=COLORS["primary"], width=2.5),
                fillcolor="rgba(45,106,79,0.15)",
                name="AC Power (kW)",
            ))
            # DC power (secondary axis) — real Kaggle column
            if "dc_power_kw" in today_hourly.columns:
                fig.add_trace(go.Scatter(
                    x=today_hourly["hour"], y=today_hourly["dc_power_kw"],
                    mode="lines",
                    line=dict(color=COLORS["secondary"], width=1.5, dash="dot"),
                    name="DC Power (kW)",
                    yaxis="y",
                ))
            # Irradiance (right axis) — real weather sensor
            fig.add_trace(go.Scatter(
                x=today_hourly["hour"], y=today_hourly["irradiance_wm2"],
                mode="lines",
                line=dict(color=COLORS["warn"], width=1.5, dash="dash"),
                name="Irradiance (W/m²)",
                yaxis="y2",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=12),
                margin=dict(l=40, r=60, t=10, b=40),
                xaxis=dict(title="Hour of Day", showgrid=True, gridcolor=COLORS["grid"],
                           tickvals=list(range(0, 25, 2))),
                yaxis=dict(title="Power (kW)", showgrid=True, gridcolor=COLORS["grid"]),
                yaxis2=dict(title="Irradiance (W/m²)", overlaying="y",
                            side="right", showgrid=False),
                legend=dict(x=0.01, y=0.99),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Efficiency Gauge")
        gauge_max = max(30.0, float(daily_df["capacity_factor_pct"].max()) * 1.1)
        st.plotly_chart(
            gauge_chart(row["capacity_factor_pct"], gauge_max, "Capacity Factor", "%"),
            use_container_width=True,
        )

    st.divider()

    # ── Energy Production Trends ──────────────────────────────────────────────
    st.subheader("Energy Production Trends")
    tab_daily, tab_weekly, tab_monthly = st.tabs(["📆 Daily", "📆 Weekly", "📆 Monthly"])

    with tab_daily:
        last30 = daily_df.tail(30).copy()
        last30["date_str"] = last30["date"].astype(str)
        fig = line_chart(last30, "date_str", "energy_kwh",
                         title="Daily AC Energy Output (kWh)", fill=True)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tab_weekly:
        daily_df2       = daily_df.copy()
        daily_df2["week"] = pd.to_datetime(daily_df2["date"]).dt.to_period("W").astype(str)
        weekly = daily_df2.groupby("week")["energy_kwh"].sum().reset_index()
        fig = line_chart(weekly, "week", "energy_kwh",
                         title="Weekly AC Energy Output (kWh)", fill=True,
                         color=COLORS["secondary"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with tab_monthly:
        daily_df3       = daily_df.copy()
        daily_df3["month"] = pd.to_datetime(daily_df3["date"]).dt.to_period("M").astype(str)
        monthly = daily_df3.groupby("month")["energy_kwh"].sum().reset_index()
        fig2 = go.Figure(go.Bar(
            x=monthly["month"], y=monthly["energy_kwh"],
            marker_color=COLORS["primary"], marker_line_width=0,
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(l=40, r=20, t=10, b=60),
            xaxis=dict(title="Month", showgrid=False),
            yaxis=dict(title="Energy (kWh)", showgrid=True, gridcolor=COLORS["grid"]),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Temperature vs Power correlation ─────────────────────────────────────
    st.subheader("🌡️ Temperature vs Daily Energy (Real Sensor Correlation)")
    st.caption(
        "Both axes are real measurements from the Kaggle dataset. "
        "Higher ambient temperature typically reduces solar panel efficiency."
    )
    fig_corr = go.Figure(go.Scatter(
        x=daily_df["avg_temp"],
        y=daily_df["energy_kwh"],
        mode="markers",
        marker=dict(
            color=daily_df["avg_irradiance"],
            colorscale="Greens",
            showscale=True,
            colorbar=dict(title="Irradiance (W/m²)"),
            size=9, opacity=0.8,
        ),
        text=daily_df["date"].astype(str),
        hovertemplate="Date: %{text}<br>Temp: %{x:.1f}°C<br>Energy: %{y:.0f} kWh<extra></extra>",
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(title="Avg Ambient Temperature (°C)", showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(title="Daily AC Energy (kWh)", showgrid=True, gridcolor=COLORS["grid"]),
        margin=dict(l=50, r=20, t=10, b=40),
        height=320,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ── Sustainability Metrics ────────────────────────────────────────────────
    st.subheader("🌿 Sustainability Metrics (Full Dataset)")
    kpis = kpi_values(daily_df)
    s1, s2, s3 = st.columns(3)
    s1.metric("☀️ Total AC Energy Generated", f"{kpis['ytd_energy_mwh']} MWh")
    s2.metric("🌲 Trees Saved Equivalent",
              f"{int(daily_df['trees_equivalent'].sum()):,}",
              help="Each tree absorbs ~21.77 kg CO₂/year")
    s3.metric("💨 CO₂ Offset",
              f"{kpis['co2_ytd_tonnes']} tonnes",
              help="At 0.82 kg CO₂ per kWh saved vs Indian grid")
