"""
app.py
------
Entry point for the Sustainable Oasis Solar Analytics Dashboard.
Run with:  streamlit run app.py

REAL-DATA SETUP
---------------
1. Download the Kaggle dataset:
   https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
2. Create a  data/  folder next to this file.
3. Place these two files inside it:
       data/Plant_1_Generation_Data.csv
       data/Plant_1_Weather_Sensor_Data.csv
4. Run:  streamlit run app.py
"""

import streamlit as st
import sys, os

# Make modules importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

# ── ONLY CHANGED LINE vs the original: data_loader instead of data_generator ──
from data_loader import generate_all

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Pulse · Vision Nexus",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Fonts & base ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1B4332 0%, #2D6A4F 100%);
    color: #D8F3DC;
}
[data-testid="stSidebar"] * { color: #D8F3DC !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.93rem; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

/* ---------- KPI metric cards ---------- */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #6B7280 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1B1B2F !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ---------- Page headings ---------- */
h1 { color: #1B4332 !important; font-weight: 700 !important; }
h2 { color: #2D6A4F !important; font-weight: 600 !important; }
h3 { color: #374151 !important; font-weight: 600 !important; }

/* ---------- Plotly chart wrapper ---------- */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* ---------- Info / warning cards ---------- */
.alert-card {
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 0.88rem;
}
.alert-high   { background:#FEE2E2; border-left: 4px solid #E76F51; }
.alert-medium { background:#FEF3C7; border-left: 4px solid #F4A261; }
.alert-low    { background:#D1FAE5; border-left: 4px solid #52B788; }

/* ---------- Dataframe styling ---------- */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ---------- Top header bar ---------- */
.main-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0 18px 0;
    border-bottom: 2px solid #E5E7EB;
    margin-bottom: 24px;
}
.main-header .logo { font-size: 1.8rem; }
.main-header h1 { margin: 0; font-size: 1.5rem; }
.main-header .sub { color: #6B7280; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Load data (cached so it only runs once per session) ────────────────────────
@st.cache_data(show_spinner="Loading real solar plant data …")
def load_data():
    return generate_all(save_csv=False)


DATA = load_data()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Solar Pulse")
    st.markdown("*Solar Analytics Platform*")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "⚡  Inverter Analytics",
            "🔗  Performance Info",
            "📈  Forecasting",
            "🔧  Operational Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Dynamic plant info from real data ─────────────────────────────────────
    inv_count = DATA["inverter_df"]["inverter_id"].nunique()
    date_min  = str(DATA["daily_df"]["date"].min())
    date_max  = str(DATA["daily_df"]["date"].max())

    st.markdown("**Plant:** Kaggle Plant 1")
    st.markdown(f"**Inverters:** {inv_count} units")
    st.markdown(f"**Data range:** {date_min} to {date_max}")
    st.divider()
    st.caption("Powered by Solar Power Generation Data.")


# ── Route to pages ─────────────────────────────────────────────────────────────
if "Overview" in page:
    from displays.page_overview import render
    render(DATA)
elif "Inverter" in page:
    from displays.page_inverter import render
    render(DATA)
elif "Performance" in page:
    from displays.page_string import render
    render(DATA)
elif "Forecasting" in page:
    from displays.page_forecast import render
    render(DATA)
elif "Operational" in page:
    from displays.page_ops import render
    render(DATA)
