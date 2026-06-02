"""
viz_utils.py
------------
Reusable Plotly chart functions for the Sustainable Oasis dashboard.
Import these in any page to keep chart code DRY and consistent.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import re

# ── Design tokens ───────────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#2D6A4F",   # deep green
    "secondary":  "#52B788",   # mid green
    "accent":     "#95D5B2",   # light green
    "warn":       "#F4A261",   # amber
    "danger":     "#E76F51",   # red-orange
    "bg":         "#F8FAF8",   # off-white background
    "card":       "#FFFFFF",
    "text":       "#1B1B2F",
    "muted":      "#6B7280",
    "grid":       "#E5E7EB",
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)

# Default axis styling kept separate from LAYOUT_BASE.
# This prevents key-conflict errors when callers pass xaxis=dict(...) or
# yaxis=dict(...) alongside **LAYOUT_BASE in the same update_layout() call.
AXIS_DEFAULTS = dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def base_layout(**overrides) -> dict:
    """
    Return a layout dict built from LAYOUT_BASE with caller overrides merged in.

    xaxis / yaxis overrides are deep-merged with AXIS_DEFAULTS so callers only
    specify what they want to change (title, range, tickangle…) without repeating
    showgrid / gridcolor every time.

    Usage:
        fig.update_layout(**base_layout(
            title=dict(text="My Chart"),
            xaxis=dict(title="Date", tickangle=45),
            yaxis=dict(title="Power (kW)", range=[0, 500]),
            height=320,
        ))
    """
    layout = {**LAYOUT_BASE}
    for key, value in overrides.items():
        if key in ("xaxis", "yaxis", "xaxis2", "yaxis2") and isinstance(value, dict):
            layout[key] = {**AXIS_DEFAULTS, **value}
        else:
            layout[key] = value
    for axis in ("xaxis", "yaxis"):
        if axis not in layout:
            layout[axis] = {**AXIS_DEFAULTS}
    return layout


# Backward-compat alias — internal viz_utils helpers that call _base_layout still work.
_base_layout = base_layout


# ── 1. Line / Area chart ────────────────────────────────────────────────────────

def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list,
    title: str = "",
    fill: bool = False,
    color: str | None = None,
    labels: dict | None = None,
) -> go.Figure:
    """Generic line (or area) chart."""
    y_cols = [y] if isinstance(y, str) else y
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["warn"]]

    fig = go.Figure()
    for i, col in enumerate(y_cols):
        c = color or palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col],
            name=(labels or {}).get(col, col),
            mode="lines",
            line=dict(color=c, width=2),
            fill="tozeroy" if fill else "none",
            fillcolor=_hex_to_rgba(c, 0.15) if fill else None,
        ))

    fig.update_layout(**_base_layout(title=dict(text=title, font=dict(size=14))))
    return fig


# ── 2. Bar chart ────────────────────────────────────────────────────────────────

def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color_col: str | None = None,
    color_map: dict | None = None,
) -> go.Figure:
    """Vertical bar chart with optional per-bar coloring."""
    if color_col and color_map:
        bar_colors = df[color_col].map(color_map).fillna(COLORS["secondary"])
    else:
        bar_colors = COLORS["secondary"]

    fig = go.Figure(go.Bar(
        x=df[x], y=df[y],
        marker_color=bar_colors,
        marker_line_width=0,
    ))
    fig.update_layout(**_base_layout(title=dict(text=title, font=dict(size=14))))
    return fig


# ── 3. Gauge chart ──────────────────────────────────────────────────────────────

def gauge_chart(value: float, max_val: float, title: str = "", unit: str = "") -> go.Figure:
    """Single-value gauge (like a speedometer)."""
    pct = value / max_val * 100

    if pct >= 85:
        bar_color = COLORS["primary"]
    elif pct >= 60:
        bar_color = COLORS["warn"]
    else:
        bar_color = COLORS["danger"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix=f" {unit}", font=dict(size=22, color=COLORS["text"])),
        title=dict(text=title, font=dict(size=13, color=COLORS["muted"])),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor=COLORS["muted"]),
            bar=dict(color=bar_color, thickness=0.6),
            bgcolor="white",
            borderwidth=1,
            bordercolor=COLORS["grid"],
            steps=[
                dict(range=[0, max_val * 0.5], color=COLORS["bg"]),
                dict(range=[max_val * 0.5, max_val * 0.75], color=COLORS["accent"]),
            ],
            threshold=dict(
                line=dict(color=COLORS["primary"], width=3),
                thickness=0.8,
                value=max_val * 0.9,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=220,
    )
    return fig


# ── 4. Heatmap ─────────────────────────────────────────────────────────────────

def heatmap(
    pivot_df: pd.DataFrame,
    title: str = "",
    colorscale: list | None = None,
) -> go.Figure:
    """Heatmap from a pre-pivoted DataFrame (rows × columns)."""
    cs = colorscale or [
        [0.0, "#E76F51"],
        [0.4, "#F4A261"],
        [0.7, "#95D5B2"],
        [1.0, "#2D6A4F"],
    ]
    fig = go.Figure(go.Heatmap(
        z=pivot_df.values,
        x=list(pivot_df.columns),
        y=list(pivot_df.index),
        colorscale=cs,
        showscale=True,
        hoverongaps=False,
        colorbar=dict(thickness=12, len=0.8),
    ))
    fig.update_layout(
        **_base_layout(title=dict(text=title, font=dict(size=14))),
        height=320,
    )
    return fig


# ── 5. Scatter plot ─────────────────────────────────────────────────────────────

def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str | None = None,
    title: str = "",
    size_col: str | None = None,
) -> go.Figure:
    kwargs = dict(x=x, y=y, title=title, color=color_col, size=size_col)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    fig = px.scatter(df, **kwargs, color_discrete_sequence=[COLORS["primary"], COLORS["warn"], COLORS["danger"]])
    fig.update_layout(**_base_layout())
    return fig


# ── 6. Forecast chart ──────────────────────────────────────────────────────────

def forecast_chart(
    dates_hist: pd.Series,
    actual: pd.Series,
    dates_fore: pd.Series,
    forecast: pd.Series,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    title: str = "Energy Forecast",
) -> go.Figure:
    """Actual line + forecast line + optional confidence band."""
    fig = go.Figure()

    # Confidence interval
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(dates_fore), pd.Series(dates_fore[::-1])]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(82,183,136,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            showlegend=True,
        ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=dates_hist, y=actual,
        mode="lines",
        name="Actual",
        line=dict(color=COLORS["primary"], width=2),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=dates_fore, y=forecast,
        mode="lines+markers",
        name="Forecast",
        line=dict(color=COLORS["warn"], width=2, dash="dot"),
        marker=dict(size=5),
    ))

    fig.update_layout(**_base_layout(title=dict(text=title, font=dict(size=14))))
    return fig


# ── 7. Multi-metric area chart ─────────────────────────────────────────────────

def stacked_area_chart(
    df: pd.DataFrame,
    x: str,
    y_cols: list,
    title: str = "",
    labels: dict | None = None,
) -> go.Figure:
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        c = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col],
            mode="lines",
            name=(labels or {}).get(col, col),
            stackgroup="one",
            line=dict(color=c, width=1),
            fillcolor=_hex_to_rgba(c, 0.6),
        ))
    fig.update_layout(**_base_layout(title=dict(text=title, font=dict(size=14))))
    return fig


# ── 8. KPI card helper (returns dict, rendered via st.metric or custom HTML) ──

def kpi_values(daily_df: pd.DataFrame) -> dict:
    """Compute latest KPI values from daily summary."""
    today = daily_df.iloc[-1]
    ytd_energy = daily_df["energy_kwh"].sum()
    return {
        "power_kw":       round(today["peak_power_kw"], 1),
        "daily_energy":   round(today["energy_kwh"], 1),
        "efficiency":     round(today["capacity_factor_pct"], 1),
        "co2_saved_kg":   round(today["co2_saved_kg"], 1),
        "trees_equiv":    round(today["trees_equivalent"], 1),
        "ytd_energy_mwh": round(ytd_energy / 1000, 2),
        "co2_ytd_tonnes": round(daily_df["co2_saved_kg"].sum() / 1000, 2),
    }
