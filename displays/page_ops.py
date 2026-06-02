"""
page_ops.py
-----------
Page 5 — Operational Insights

Changes from synthetic version:
  - fault_df no longer has fault_type / severity / resolved columns
    (those were randomly assigned — not in the Kaggle data).
  - Fault analysis now focuses on what IS real:
      * which inverter lost AC output during daylight
      * DC power at the time (inverter trip vs irradiance issue)
      * duration and frequency of zero-power events
  - Health score derived from real efficiency and fault-event count only.
  - Utilisation target line is data-driven (median peak power).
  - No synthesized data anywhere.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from viz_utils import COLORS, LAYOUT_BASE, base_layout


def render(data: dict):
    inverter_df = data["inverter_df"]
    fault_df    = data["fault_df"]
    daily_df    = data["daily_df"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <span class="logo">🔧</span>
        <div>
            <h1>Operational Insights</h1>
            <span class="sub">Real Fault Detection and Performance Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # st.info(
    #     "**Data transparency:** Fault events are derived from real zero-AC-power intervals "
    #     "during daylight hours (07:00–18:00). No fault types, severities, or resolution "
    #     "statuses are assigned — those fields were not present in the source dataset."
    # )

    # ── Date range filter ─────────────────────────────────────────────────────
    all_dates = sorted(inverter_df["date"].unique())
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        start_date = st.date_input("From", value=all_dates[0],
                                   min_value=all_dates[0], max_value=all_dates[-1])
    with col_f2:
        end_date = st.date_input("To", value=all_dates[-1],
                                 min_value=all_dates[0], max_value=all_dates[-1])

    date_norm = pd.to_datetime(inverter_df["date"]).dt.date
    mask = (date_norm >= start_date) & (date_norm <= end_date)
    filtered_inv = inverter_df[mask].copy()

    if filtered_inv.empty:
        st.warning("No data in selected date range.")
        return

    # ── Plant Health Score ─────────────────────────────────────────────────────
    st.subheader("🏥 Plant Health Assessment")
    st.caption("Derived from real efficiency readings and fault-event counts.")

    total_intervals    = len(filtered_inv)
    fault_intervals    = (filtered_inv["status"] == "FAULT").sum()
    fault_rate_pct     = fault_intervals / max(total_intervals, 1) * 100

    avg_eff            = filtered_inv[filtered_inv["efficiency_pct"].notna()]["efficiency_pct"].mean()
    eff_range          = filtered_inv["efficiency_pct"].quantile([0.1, 0.9])
    eff_lower, eff_upper = float(eff_range.iloc[0]), float(eff_range.iloc[1])

    # Score: 100 - fault_rate_penalty - efficiency_penalty
    eff_penalty  = max(0, (95 - avg_eff) * 3)           # lose points below 95%
    fault_penalty = min(30, fault_rate_pct * 4)      # lose up to 30 pts from faults
    health_score = max(0, min(100, 100 - eff_penalty - fault_penalty))

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric("🏥 Health Score",       f"{health_score:.1f} / 100")
    col_h2.metric("📊 Avg DC→AC Eff.",     f"{avg_eff:.1f}%")
    col_h3.metric("⚡ Fault-Interval Rate", f"{fault_rate_pct:.2f}%")
    col_h4.metric("⚠️ Zero-Power Events",  int(fault_intervals))

    # Health band
    if health_score >= 80:
        st.success(f"✅ Plant is **healthy** (score {health_score:.1f}). Efficiency well within range.")
    elif health_score >= 60:
        st.warning(f"🟡 Plant shows **moderate degradation** (score {health_score:.1f}). "
                   f"Check inverters with repeated zero-power events.")
    else:
        st.error(f"🔴 Plant health is **poor** (score {health_score:.1f}). "
                 f"Significant efficiency loss or high fault frequency detected.")

    st.divider()

    # ── Fault event analysis ───────────────────────────────────────────────────
    st.subheader("⚠️ Zero-Power Event Analysis")

    if fault_df.empty:
        st.success("✅ No zero-power daytime events found in the selected date range.")
    else:
        # Filter fault_df to date range
        fault_date_norm = pd.to_datetime(fault_df["timestamp"]).dt.date
        faults_filtered = fault_df[
            (fault_date_norm >= start_date) & (fault_date_norm <= end_date)
        ].copy()

        if faults_filtered.empty:
            st.success("✅ No zero-power daytime events found in the selected date range.")
        else:
            # Summary by inverter
            fault_summary = (
                faults_filtered.groupby("inverter_id")
                .agg(
                    event_count    =("timestamp",   "count"),
                    avg_dc_power   =("dc_power_kw", "mean"),
                    inverter_trips =("likely_cause", lambda x: (x == "Inverter trip (DC present, AC=0)").sum()),
                    irr_related    =("likely_cause", lambda x: (x == "Low irradiance / no generation").sum()),
                )
                .reset_index()
                .sort_values("event_count", ascending=False)
            )

            st.caption(
                "**Inverter trip** = DC power present but AC output is zero → likely inverter fault.  \n"
                "**Low irradiance** = both DC and AC near zero → likely a cloud/night event, not a hardware fault."
            )

            # Alert cards
            for _, row in fault_summary.iterrows():
                trips = int(row["inverter_trips"])
                if trips > 0:
                    severity_cls = "alert-high"
                    icon = "🔴"
                elif row["event_count"] > 10:
                    severity_cls = "alert-medium"
                    icon = "🟡"
                else:
                    severity_cls = "alert-low"
                    icon = "🟢"

                st.markdown(f"""
                <div class="alert-card {severity_cls}">
                    <strong>{icon} {row['inverter_id']}</strong> —
                    {int(row['event_count'])} zero-power intervals,
                    of which <b>{trips} inverter trips</b>
                    and
                    <b>{int(row['irr_related'])} irradiance-related events</b>.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # Fault timeline
            st.subheader("Zero-Power Event Timeline")
            faults_filtered["date_str"] = pd.to_datetime(
                faults_filtered["timestamp"]
            ).dt.date.astype(str)

            daily_faults = (
                faults_filtered.groupby(["date_str", "inverter_id"])
                .size().reset_index(name="events")
            )

            palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                       COLORS["warn"], COLORS["danger"], "#A78BFA"]
            fig_tl = go.Figure()
            for i, inv in enumerate(sorted(daily_faults["inverter_id"].unique())):
                sub = daily_faults[daily_faults["inverter_id"] == inv]
                fig_tl.add_trace(go.Bar(
                    x=sub["date_str"], y=sub["events"],
                    name=inv,
                    marker_color=palette[i % len(palette)],
                ))
            fig_tl.update_layout(**base_layout(
                barmode="stack",
                title=dict(text="Daily Zero-Power Events by Inverter", font=dict(size=14)),
                xaxis=dict(title="Date", tickangle=30),
                yaxis=dict(title="Event Count (15-min intervals)"),
                legend=dict(orientation="h", y=-0.25),
                height=340,
            ))
            st.plotly_chart(fig_tl, use_container_width=True)

            # DC context: trip vs irradiance scatter
            st.subheader("DC Power During Zero-AC Events")
            st.caption(
                "Points with DC > 0.5 kW are likely hardware faults. "
                "Points near DC=0 are likely caused by low irradiance."
            )
            fig_dc = go.Figure()
            for i, inv in enumerate(sorted(faults_filtered["inverter_id"].unique())):
                sub = faults_filtered[faults_filtered["inverter_id"] == inv]
                fig_dc.add_trace(go.Scatter(
                    x=sub["hour"], y=sub["dc_power_kw"],
                    mode="markers",
                    name=inv,
                    marker=dict(color=palette[i % len(palette)], size=7, opacity=0.7),
                    hovertemplate=(
                        f"<b>{inv}</b><br>"
                        "Hour: %{x:.1f}<br>"
                        "DC Power: %{y:.2f} kW<extra></extra>"
                    ),
                ))
            fig_dc.add_hline(y=0.5, line_dash="dot", line_color=COLORS["danger"],
                             annotation_text="DC threshold (0.5 kW)",
                             annotation_position="top right")
            fig_dc.update_layout(**base_layout(
                title=dict(text="DC Power (kW) at Time of Zero-AC Intervals", font=dict(size=14)),
                xaxis=dict(title="Hour of Day"),
                yaxis=dict(title="DC Power (kW)"),
                height=340,
            ))
            st.plotly_chart(fig_dc, use_container_width=True)

    st.divider()

    # ── Efficiency analysis ────────────────────────────────────────────────────
    st.subheader("📊 DC→AC Conversion Efficiency")
    st.caption("Ratio of AC to DC power per inverter.")

    eff_inv = (
        filtered_inv[filtered_inv["dc_power_kw"] > 0]
        .groupby("inverter_id")["efficiency_pct"]
        .agg(["mean", "min", "max", "std"])
        .round(2)
        .reset_index()
    )
    eff_inv.columns = ["Inverter", "Mean Eff. (%)", "Min Eff. (%)", "Max Eff. (%)", "Std Dev (%)"]

    col_tbl, col_bar = st.columns([1, 2])
    with col_tbl:
        st.dataframe(eff_inv, use_container_width=True, hide_index=True)
    with col_bar:
        colors_eff = [
            COLORS["primary"] if v >= 95 else (COLORS["warn"] if v >= 90 else COLORS["danger"])
            for v in eff_inv["Mean Eff. (%)"]
        ]
        fig_eff = go.Figure(go.Bar(
            x=eff_inv["Inverter"], y=eff_inv["Mean Eff. (%)"],
            marker_color=colors_eff,
            error_y=dict(type="data", array=eff_inv["Std Dev (%)"].tolist(), visible=True,
                         color=COLORS["muted"]),
            text=eff_inv["Mean Eff. (%)"].astype(str) + "%",
            textposition="outside",
        ))
        fig_eff.add_hline(y=95, line_dash="dot", line_color=COLORS["danger"],
                          annotation_text="Target 95%", annotation_position="bottom right")
        fig_eff.update_layout(**base_layout(
            title=dict(text="Mean DC→AC Efficiency by Inverter", font=dict(size=14)),
            yaxis=dict(range=[80, 100], title="Efficiency (%)"),
            height=300,
            showlegend=False,
        ))
        st.plotly_chart(fig_eff, use_container_width=True)

    st.divider()

    # ── Daily energy & peak power trend ───────────────────────────────────────
    st.subheader("📈 Plant-Level Daily Trends")

    daily_date_norm = pd.to_datetime(daily_df["date"]).dt.date
    daily_filtered  = daily_df[
        (daily_date_norm >= start_date) & (daily_date_norm <= end_date)
    ].copy()
    daily_filtered["date_str"] = daily_filtered["date"].astype(str)

    col_e, col_p = st.columns(2)

    with col_e:
        fig_e = go.Figure(go.Bar(
            x=daily_filtered["date_str"],
            y=daily_filtered["energy_kwh"],
            marker_color=COLORS["primary"],
        ))
        fig_e.update_layout(**base_layout(
            title=dict(text="Daily AC Energy Output (kWh)", font=dict(size=14)),
            xaxis=dict(tickangle=45),
            yaxis=dict(title="Energy (kWh)"),
            height=300,
        ))
        st.plotly_chart(fig_e, use_container_width=True)

    with col_p:
        median_peak = daily_filtered["peak_power_kw"].median()
        fig_p = go.Figure(go.Scatter(
            x=daily_filtered["date_str"], y=daily_filtered["peak_power_kw"],
            mode="lines+markers",
            line=dict(color=COLORS["secondary"], width=2),
            marker=dict(size=5),
        ))
        fig_p.add_hline(
            y=median_peak, line_dash="dot", line_color=COLORS["warn"],
            annotation_text=f"Median {median_peak:.0f} kW",
            annotation_position="bottom right",
        )
        fig_p.update_layout(**base_layout(
            title=dict(text="Daily Peak Power (kW)", font=dict(size=14)),
            xaxis=dict(tickangle=45),
            yaxis=dict(title="Peak Power (kW)"),
            height=300,
        ))
        st.plotly_chart(fig_p, use_container_width=True)

    st.divider()

    # ── Performance Ratio trend ────────────────────────────────────────────────
    # if "performance_ratio" in daily_filtered.columns:
    #     st.subheader("📐 Performance Ratio (PR) Trend")
    #     st.caption(
    #         "PR = AC Energy / (Daily Insolation × Nameplate Capacity). "
    #         "A real IEC metric — values ~0.7–0.85 are typical for utility-scale plants."
    #     )
    #     pr_df = daily_filtered[daily_filtered["performance_ratio"].notna()]
    #     fig_pr = go.Figure(go.Scatter(
    #         x=pr_df["date_str"], y=pr_df["performance_ratio"],
    #         mode="lines+markers",
    #         line=dict(color=COLORS["accent"], width=2),
    #         marker=dict(size=5),
    #         fill="tozeroy", fillcolor="rgba(149,213,178,0.15)",
    #     ))
    #     fig_pr.add_hline(y=0.75, line_dash="dot", line_color=COLORS["warn"],
    #                      annotation_text="Typical PR = 0.75")
    #     fig_pr.update_layout(**base_layout(
    #         title=dict(text="Daily Performance Ratio", font=dict(size=14)),
    #         xaxis=dict(tickangle=45),
    #         yaxis=dict(title="PR (dimensionless)", range=[0, 1.1]),
    #         height=300,
    #     ))
    #     st.plotly_chart(fig_pr, use_container_width=True)
    #     st.divider()

    # ── Inverter uptime table ─────────────────────────────────────────────────
    st.subheader("🕒 Inverter Uptime Summary")
    st.caption("Uptime = Intervals with AC power > 0.5 kW during daylight (07:00–18:00).")

    daylight_df = filtered_inv[
        (filtered_inv["hour"] >= 7) & (filtered_inv["hour"] < 18)
    ].copy()

    uptime_tbl = (
        daylight_df.groupby("inverter_id")
        .apply(lambda g: pd.Series({
            "Total Daylight Intervals": len(g),
            "Producing Intervals":      (g["power_kw"] >= 0.5).sum(),
            "Zero-Power Intervals":     (g["power_kw"] < 0.5).sum(),
            "Uptime (%)":              round((g["power_kw"] >= 0.5).mean() * 100, 2),
            "Mean DC→AC Eff. (%)":     round(g[g["dc_power_kw"] > 0]["efficiency_pct"].mean(), 2)
                                       if (g["dc_power_kw"] > 0).any() else float("nan"),
        }))
        .reset_index()
    )
    st.dataframe(uptime_tbl, use_container_width=True, hide_index=True)

    # ── Raw fault table ───────────────────────────────────────────────────────
    if not fault_df.empty:
        with st.expander("📋 View Raw Zero-Power Event Log"):
            fdt = fault_df.copy()
            fdt["timestamp"] = fdt["timestamp"].astype(str)
            fdt.columns = [c.replace("_", " ").title() for c in fdt.columns]
            st.dataframe(fdt, use_container_width=True, hide_index=True)
