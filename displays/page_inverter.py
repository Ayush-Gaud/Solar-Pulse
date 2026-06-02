"""
page_inverter.py
----------------
Page 2 — Inverter Analytics

Changes from synthetic version:
  - Removed synthesised voltage_v / current_a columns (not in Kaggle data).
  - Added DC power and daily_yield charts — both real Kaggle columns.
  - energy_kwh derived from ac_power_kw * 0.25 (15-min interval).
  - Date filter uses pd.Timestamp-safe comparison.
  - No synthesised data anywhere.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from viz_utils import COLORS, LAYOUT_BASE, base_layout


def render(data: dict):
    inverter_df = data["inverter_df"]
    fault_df    = data["fault_df"]

    # energy_kwh = AC power × 0.25 h per interval
    if "energy_kwh" not in inverter_df.columns:
        inverter_df = inverter_df.copy()
        inverter_df["energy_kwh"] = inverter_df["power_kw"] * 0.25

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <span class="logo">⚡</span>
        <div>
            <h1>Inverter Analytics</h1>
            <span class="sub">Based on Real Inverter Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([0.5, 3.5])
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
        selected_invs = st.multiselect(
            "Select Inverters",
            options=inv_options,
            default=inv_options,
        )

    sel_date_norm  = pd.Timestamp(selected_date).date()
    inv_dates_norm = pd.to_datetime(inverter_df["date"]).dt.date

    day_df = inverter_df[
        (inv_dates_norm == sel_date_norm) &
        (inverter_df["inverter_id"].isin(selected_invs))
    ].copy()

    # ── Status Summary KPIs ───────────────────────────────────────────────────
    st.subheader("Inverter Status Summary")

    day_status = (
        day_df.groupby("inverter_id")["status"]
        .agg(lambda x: "FAULT" if "FAULT" in x.values else ("OK" if (x != "IDLE").any() else "IDLE"))
        .reset_index()
    )
    active  = (day_status["status"] == "OK").sum()
    faulted = (day_status["status"] == "FAULT").sum()
    idle    = (day_status["status"] == "IDLE").sum()

    day_eff_mask = inv_dates_norm == sel_date_norm
    avg_eff      = inverter_df[day_eff_mask]["efficiency_pct"].mean()
    total_power  = day_df["power_kw"].sum()

    k1, k2, k3, k4, k5 = st.columns([1,1,1,1.2,1.5])
    k1.metric("🟢 Active",         active)
    k2.metric("🔴 Fault Events",   faulted)
    k3.metric("⚪ Idle",            idle)
    k4.metric("📊 Avg DC→AC Eff.", f"{avg_eff:.1f}%")
    k5.metric("⚡ Total AC Power", f"{total_power:.1f} kW")

    st.divider()

    # ── Per-Inverter Performance Cards ────────────────────────────────────────
    st.subheader("Per-Inverter Performance")

    inv_summary = (
        day_df.groupby("inverter_id")
        .agg(
            energy_kwh     =("energy_kwh",     "sum"),
            dc_energy_kwh  =("dc_power_kw",    lambda x: (x * 0.25).sum()),
            peak_kw        =("power_kw",        "max"),
            avg_efficiency =("efficiency_pct",  "mean"),
            fault_count    =("status",  lambda x: (x == "FAULT").sum()),
            daily_yield    =("daily_yield",     "max"),
        )
        .reset_index()
    )

    def health_badge(row):
        if row["fault_count"] > 3:
            return "🔴 Degraded"
        elif row["avg_efficiency"] < 93:
            return "🟡 Warning"
        return "🟢 Healthy"

    inv_summary["health"] = inv_summary.apply(health_badge, axis=1)

    cols = st.columns(3)
    for i, (_, row) in enumerate(inv_summary.iterrows()):
        color  = "#D1FAE5" if "Healthy" in row["health"] else ("#FEF3C7" if "Warning" in row["health"] else "#FEE2E2")
        border = "#52B788" if "Healthy" in row["health"] else ("#F4A261" if "Warning" in row["health"] else "#E76F51")
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:{color}; border-left:4px solid {border};
                        border-radius:10px; padding:14px 16px; margin-bottom:12px;">
                <div style="font-size:1rem; font-weight:700; color:#1B1B2F;">{row['inverter_id']}</div>
                <div style="font-size:0.8rem; color:#374151; margin-top:4px;">{row['health']}</div>
                <hr style="border:none; border-top:1px solid rgba(0,0,0,0.1); margin:8px 0;">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:0.82rem; color:#374151;">
                    <div>⚡ AC Energy: <b>{row['energy_kwh']:.0f} kWh</b></div>
                    <div>🔋 DC Energy: <b>{row['dc_energy_kwh']:.0f} kWh</b></div>
                    <div>🔝 Peak AC: <b>{row['peak_kw']:.1f} kW</b></div>
                    <div>📊 DC→AC Eff: <b>{row['avg_efficiency']:.1f}%</b></div>
                    <div>📈 Daily Yield: <b>{row['daily_yield']:.0f} kWh</b></div>
                    <div>⚠️ Zero-AC: <b>{int(row['fault_count'])} intervals</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Comparative Bar Charts ────────────────────────────────────────────────
    st.subheader("Comparative Analysis")
    col_b1, col_b2 = st.columns(2)

    color_map = {
        "🟢 Healthy":  COLORS["primary"],
        "🟡 Warning":  COLORS["warn"],
        "🔴 Degraded": COLORS["danger"],
    }

    with col_b1:
        bar_colors = inv_summary["health"].map(color_map).fillna(COLORS["secondary"])
        fig = go.Figure(go.Bar(
            x=inv_summary["inverter_id"],
            y=inv_summary["energy_kwh"],
            marker_color=bar_colors,
            marker_line_width=0,
            text=inv_summary["energy_kwh"].round(0).astype(int),
            textposition="outside",
        ))
        fig.update_layout(**base_layout(
            title=dict(text="Daily AC Energy Output (kWh)", font=dict(size=14)),
            height=320, showlegend=False,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_b2:
        eff_colors = [
            COLORS["primary"] if e >= 95 else (COLORS["warn"] if e >= 90 else COLORS["danger"])
            for e in inv_summary["avg_efficiency"]
        ]
        fig2 = go.Figure(go.Bar(
            x=inv_summary["inverter_id"],
            y=inv_summary["avg_efficiency"],
            marker_color=eff_colors,
            marker_line_width=0,
            text=inv_summary["avg_efficiency"].round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig2.add_hline(y=95, line_dash="dot", line_color=COLORS["danger"],
                       annotation_text="Target 95%", annotation_position="bottom right")
        fig2.update_layout(**base_layout(
            title=dict(text="Avg DC→AC Efficiency (%)", font=dict(size=14)),
            yaxis=dict(range=[80, 100]),
            height=320,
            showlegend=False,
        ))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── AC vs DC Power — Hourly ────────────────────────────────────────────────
    st.subheader("Hourly AC & DC Power by Inverter")
    # st.caption("Both columns are direct Kaggle measurements.")

    hourly = (
        day_df.groupby(["hour", "inverter_id"])
        .agg(ac_kw=("power_kw", "mean"), dc_kw=("dc_power_kw", "mean"))
        .reset_index()
    )

    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
               COLORS["warn"], COLORS["danger"], "#A78BFA"]

    tab_ac, tab_dc = st.tabs(["AC Power", "DC Power"])
    for tab, col in zip([tab_ac, tab_dc], ["ac_kw", "dc_kw"]):
        with tab:
            fig3 = go.Figure()
            for i, inv in enumerate(sorted(hourly["inverter_id"].unique())):
                sub = hourly[hourly["inverter_id"] == inv]
                fig3.add_trace(go.Scatter(
                    x=sub["hour"], y=sub[col],
                    mode="lines", name=inv,
                    line=dict(color=palette[i % len(palette)], width=2),
                ))
            fig3.update_layout(**base_layout(
                title=dict(text=f"{'AC' if col == 'ac_kw' else 'DC'} Power (kW) — Hourly", font=dict(size=14)),
                xaxis=dict(title="Hour of Day", tickvals=list(range(0, 25, 2))),
                yaxis=dict(title="Power (kW)"),
                legend=dict(orientation="h", y=-0.4),
                height=340,
            ))
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── 30-Day DC→AC Efficiency Trend ─────────────────────────────────────────
    st.subheader("30-Day DC→AC Efficiency Trend")

    daily_eff = (
        inverter_df[inverter_df["inverter_id"].isin(selected_invs)]
        .groupby(["date", "inverter_id"])["efficiency_pct"]
        .mean()
        .reset_index()
    )

    fig4 = go.Figure()
    for i, inv in enumerate(sorted(daily_eff["inverter_id"].unique())):
        sub = daily_eff[daily_eff["inverter_id"] == inv]
        fig4.add_trace(go.Scatter(
            x=sub["date"].astype(str), y=sub["efficiency_pct"],
            mode="lines", name=inv,
            line=dict(color=palette[i % len(palette)], width=1.8),
        ))
    fig4.add_hline(y=95, line_dash="dot", line_color=COLORS["danger"],
                   annotation_text="Target (95%)", annotation_position="bottom right")
    fig4.update_layout(**base_layout(
        title=dict(text="Daily Avg DC→AC Efficiency per Inverter (%)", font=dict(size=14)),
        xaxis=dict(title="Date", tickangle=45),
        yaxis=dict(title="Efficiency (%)", range=[80, 100]),
        legend=dict(orientation="h", y=-0.5),
        height=360,
    ))
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Cumulative Total Yield ─────────────────────────────────────────────────
    st.subheader("Cumulative Total Yield by Inverter")
    # st.caption("The inverter's lifetime energy counter.")

    total_yield_df = (
        inverter_df[inverter_df["inverter_id"].isin(selected_invs)]
        .groupby("inverter_id")["total_yield"]
        .max()
        .reset_index()
        .sort_values("total_yield", ascending=False)
    )

    fig_ty = go.Figure(go.Bar(
        x=total_yield_df["inverter_id"],
        y=total_yield_df["total_yield"],
        marker_color=[palette[i % len(palette)] for i in range(len(total_yield_df))],
        text=total_yield_df["total_yield"].round(0).astype(int),
        textposition="outside",
    ))
    fig_ty.update_layout(**base_layout(
        title=dict(text="Lifetime Total Yield per Inverter (kWh)", font=dict(size=14)),
        yaxis=dict(title="Total Yield (kWh)"),
        height=300,
        showlegend=False,
    ))
    st.plotly_chart(fig_ty, use_container_width=True)

    # ── Raw data ──────────────────────────────────────────────────────────────
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
