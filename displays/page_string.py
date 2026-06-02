"""
page_string.py
--------------
Page 3 — String Performance

The Kaggle Plant 1 dataset does NOT contain string-level measurements.
This page explains that clearly and instead surfaces the most granular
real data available: per-inverter 15-min AC/DC power readings.

No data is synthesised or hallucinated.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from viz_utils import COLORS, LAYOUT_BASE, base_layout


def render(data: dict):
    string_df   = data["string_df"]
    inverter_df = data["inverter_df"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <span class="logo">🔗</span>
        <div>
            <h1>Performance Info</h1>
            <span class="sub">In-Depth Inverter Level Detail</span>
        </div>
    </div>
    """, unsafe_allow_html=True) 

    # # ── Honest data-availability notice ───────────────────────────────────────
    # st.info(
    #     "**ℹ️ No String-Level Data Available**\n\n"
    #     "The Kaggle Solar Power Generation dataset records data at the **inverter level** "
    #     "(SOURCE_KEY), not at the individual string level. String-level measurements "
    #     "(individual strand voltage, current, and energy) are not present in the source CSVs "
    #     "and cannot be derived from it without additional sensors or a monitoring system.\n\n"
    #     "This page shows the most granular real data available: "
    #     "**per-inverter 15-minute AC and DC power readings**."
    # )

    # st.divider()

    # ── Date filter ───────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        all_dates = sorted(inverter_df["date"].unique())
        selected_date = st.date_input(
            "Select Date",
            value=all_dates[-1],
            min_value=all_dates[0],
            max_value=all_dates[-1],
        )
    with col_f2:
        inv_options = sorted(inverter_df["inverter_id"].unique())
        selected_inv = st.selectbox("Focus Inverter", options=["All"] + inv_options)

    sel_date_norm  = pd.Timestamp(selected_date).date()
    inv_dates_norm = pd.to_datetime(inverter_df["date"]).dt.date
    day_df = inverter_df[inv_dates_norm == sel_date_norm].copy()

    if selected_inv != "All":
        day_df = day_df[day_df["inverter_id"] == selected_inv]

    if day_df.empty:
        st.warning("No data for the selected date.")
        return

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total_ac   = day_df["power_kw"].sum() * 0.25          # kWh
    total_dc   = day_df["dc_power_kw"].sum() * 0.25        # kWh
    peak_ac    = day_df["power_kw"].max()
    avg_eff    = day_df["efficiency_pct"].mean()
    fault_cnt  = (day_df["status"] == "FAULT").sum()

    k1, k2, k3, k4, k5 = st.columns([1.8,1.8,1.4,1,1.2])
    k1.metric("⚡ AC Energy",          f"{total_ac:.1f} kWh")
    k2.metric("🔋 DC Energy",          f"{total_dc:.1f} kWh")
    k3.metric("🔝 Peak AC Power",      f"{peak_ac:.1f} kW")
    k4.metric("📊 Avg DC→AC Eff.",     f"{avg_eff:.1f}%")
    k5.metric("⚠️ Zero-Power Intervals", int(fault_cnt))

    st.divider()

    # ── Per-inverter AC vs DC power (15-min, selected day) ───────────────────
    st.subheader("AC vs DC Power by Inverter — 15-Minute Resolution")
    # st.caption("Source: real Kaggle inverter readings. Each point = one 15-min interval.")

    inv_list = sorted(day_df["inverter_id"].unique())
    palette  = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                COLORS["warn"], COLORS["danger"], "#A78BFA"]

    for i in range(0, len(inv_list), 2):
        cols = st.columns(2)
        for j, inv in enumerate(inv_list[i:i+2]):
            sub = day_df[day_df["inverter_id"] == inv].sort_values("hour")
            with cols[j]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sub["hour"], y=sub["dc_power_kw"],
                    mode="lines", name="DC Power",
                    line=dict(color=COLORS["warn"], width=1.5, dash="dot"),
                ))
                fig.add_trace(go.Scatter(
                    x=sub["hour"], y=sub["power_kw"],
                    mode="lines", name="AC Power",
                    line=dict(color=COLORS["primary"], width=2),
                    fill="tozeroy",
                    fillcolor="rgba(45,106,79,0.10)",
                ))
                fig.update_layout(**base_layout(
                    title=dict(text=f"{inv}", font=dict(size=13)),
                    xaxis=dict(title="Hour", tickvals=list(range(0, 25, 3))),
                    yaxis=dict(title="Power (kW)"),
                    legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
                    height=260,
                    margin=dict(l=40, r=10, t=40, b=50),
                ))
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── DC→AC Efficiency heatmap across inverters × hours ─────────────────────
    st.subheader("DC→AC Conversion Efficiency — Inverter × Hour")
    st.caption("Efficiency = AC power / DC power × 100 (with DC > 0).")

    eff_df = day_df[day_df["dc_power_kw"] > 0].copy()
    eff_df["hour_bin"] = eff_df["hour"].apply(lambda h: int(h))

    pivot = (
        eff_df.groupby(["inverter_id", "hour_bin"])["efficiency_pct"]
        .mean()
        .reset_index()
        .pivot(index="inverter_id", columns="hour_bin", values="efficiency_pct")
    )

    if not pivot.empty:
        cs = [
            [0.0,  "#E76F51"],
            [0.35, "#F4A261"],
            [0.65, "#95D5B2"],
            [1.0,  "#2D6A4F"],
        ]
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in pivot.columns],
            y=list(pivot.index),
            colorscale=cs,
            showscale=True,
            hoverongaps=False,
            colorbar=dict(thickness=14, len=0.85, title=dict(text="%", side="right")),
            text=pivot.values.round(1),
            texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
            zmin=80, zmax=100,
        ))
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            title=dict(text="DC→AC Efficiency (%) by Inverter and Hour", font=dict(size=14)),
            margin=dict(l=80, r=20, t=50, b=60),
            height=320,
            xaxis=dict(title="Hour of Day"),
            yaxis=dict(title="Inverter", autorange="reversed"),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No DC power recorded for the selected filters.")

    st.divider()

    # ── Inverter yield comparison (daily_yield from Kaggle) ───────────────────
    st.subheader("Cumulative Daily Yield by Inverter")
    # st.caption(
    #     "DAILY_YIELD is the inverter's own cumulative energy counter "
    #     "for the selected day — a direct Kaggle column, not derived."
    # )

    yield_df = (
        day_df.groupby("inverter_id")["daily_yield"]
        .max()   # take the end-of-day cumulative value
        .reset_index()
        .sort_values("daily_yield", ascending=False)
    )

    bar_colors = [palette[i % len(palette)] for i in range(len(yield_df))]
    fig_yield = go.Figure(go.Bar(
        x=yield_df["inverter_id"],
        y=yield_df["daily_yield"],
        marker_color=bar_colors,
        marker_line_width=0,
        text=yield_df["daily_yield"].round(0).astype(int),
        textposition="outside",
    ))
    fig_yield.update_layout(**base_layout(
        title=dict(text="Lifetime Daily Yield per Inverter (kWh)", font=dict(size=14)),
        xaxis=dict(title="Inverter"),
        yaxis=dict(title="Yield (kWh)"),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    ))
    st.plotly_chart(fig_yield, use_container_width=True)

    st.divider()

    # ── Zero-power events table (real faults) ─────────────────────────────────
    fault_day = day_df[day_df["status"] == "FAULT"].copy()
    if not fault_day.empty:
        st.subheader("⚠️ Zero AC-Power Intervals During Daylight")
        st.caption(
            "Rows where AC power < 0.5 kW between 07:00–18:00. "
            "DC power column reveals whether the inverter tripped (DC present) "
            "or there was simply no irradiance."
        )
        fault_day["likely_cause"] = fault_day["dc_power_kw"].apply(
            lambda dc: "Inverter trip (DC present, AC=0)" if dc > 0.5
                       else "Low irradiance / no generation"
        )
        display = fault_day[["timestamp", "inverter_id", "hour",
                              "dc_power_kw", "likely_cause"]].copy()
        display["timestamp"] = display["timestamp"].astype(str)
        display.columns = ["Timestamp", "Inverter", "Hour",
                           "DC Power (kW)", "Likely Cause"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No zero-power intervals during daylight for the selected filters.")

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("📋 View Raw Inverter Data for the Selected Day"):
        show = day_df[[
            "timestamp", "inverter_id", "hour",
            "power_kw", "dc_power_kw", "efficiency_pct",
            "daily_yield", "total_yield", "status",
        ]].copy()
        show["timestamp"] = show["timestamp"].astype(str)
        show.columns = [
            "Timestamp", "Inverter", "Hour",
            "AC Power (kW)", "DC Power (kW)", "DC→AC Eff. (%)",
            "Daily Yield (kWh)", "Total Yield (kWh)", "Status",
        ]
        st.dataframe(show, use_container_width=True, hide_index=True)
