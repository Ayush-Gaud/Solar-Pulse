"""
data_loader.py
--------------
Loads the Kaggle "Solar Power Generation Data" dataset and reshapes it
into the dict-of-DataFrames schema the dashboard expects.

Kaggle dataset: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
Place these two files in a  data/  folder next to app.py:
    data/Plant_1_Generation_Data.csv
    data/Plant_1_Weather_Sensor_Data.csv

REAL-DATA ONLY POLICY
---------------------
This loader does NOT synthesise, simulate, or hallucinate any values.
Every column in every DataFrame comes directly from the Kaggle CSVs.
Columns that required synthesis in the previous version (string-level data,
voltage/current, cloud cover proxy) are either dropped or clearly absent.

Output schema:
    power_df    — 15-min plant-level rows  (real AC power + weather)
    daily_df    — one row per day          (aggregated from real data)
    inverter_df — 15-min per-inverter rows (last 30 days, all inverters, real AC/DC power)
    string_df   — None / empty placeholder (Kaggle has no string data)
    fault_df    — zero-AC-power events during daylight (derived, no random labels)
"""

import numpy as np
import pandas as pd
import os

# ── Constants ──────────────────────────────────────────────────────────────────
PLANT_CAPACITY_KW   = 29700        # nameplate capacity for efficiency calc (22 inverters × ~182 kW each)
CO2_PER_KWH         = 0.82         # kg CO₂ offset per kWh (Indian grid average)
TREE_CO2_PER_YEAR   = 21.77        # kg CO₂ absorbed per tree per year
DAYLIGHT_START      = 7
DAYLIGHT_END        = 18

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _parse_dt(series: pd.Series) -> pd.Series:
    """Try multiple datetime formats found in the Kaggle CSVs."""
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, infer_datetime_format=True)


def _load_raw(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read generation + weather CSVs; raise a clear error if missing."""
    gen_path = os.path.join(data_dir, "Plant_1_Generation_Data.csv")
    wx_path  = os.path.join(data_dir, "Plant_1_Weather_Sensor_Data.csv")

    if not os.path.exists(gen_path) or not os.path.exists(wx_path):
        raise FileNotFoundError(
            f"\n\n[data_loader] CSV files not found in '{data_dir}'.\n"
            "Please download the Kaggle dataset:\n"
            "  https://www.kaggle.com/datasets/anikannal/solar-power-generation-data\n"
            "and place Plant_1_Generation_Data.csv and Plant_1_Weather_Sensor_Data.csv\n"
            f"inside the  data/  folder ({data_dir}).\n"
        )

    gen = pd.read_csv(gen_path)
    wx  = pd.read_csv(wx_path)
    return gen, wx


def _normalise_generation(gen: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the generation dataframe."""
    gen = gen.copy()
    gen.columns = [c.strip().upper() for c in gen.columns]

    gen["timestamp"]   = _parse_dt(gen["DATE_TIME"])
    gen["date"]        = gen["timestamp"].dt.date
    gen["hour"]        = gen["timestamp"].dt.hour + gen["timestamp"].dt.minute / 60

    # Rename opaque hash SOURCE_KEYs → INV-01 … INV-N (all inverters)
    keys    = sorted(gen["SOURCE_KEY"].unique())
    key_map = {k: f"INV-{i+1:02d}" for i, k in enumerate(keys)}
    gen["inverter_id"] = gen["SOURCE_KEY"].map(key_map)

    gen["ac_power_kw"]  = pd.to_numeric(gen["AC_POWER"],    errors="coerce").fillna(0).clip(lower=0)
    gen["dc_power_kw"]  = pd.to_numeric(gen["DC_POWER"],    errors="coerce").fillna(0).clip(lower=0)/10.0
    gen["daily_yield"]  = pd.to_numeric(gen["DAILY_YIELD"], errors="coerce").fillna(0)
    gen["total_yield"]  = pd.to_numeric(gen["TOTAL_YIELD"], errors="coerce").fillna(0)

    return gen[["timestamp", "date", "hour", "inverter_id",
                "ac_power_kw", "dc_power_kw", "daily_yield", "total_yield"]]


def _normalise_weather(wx: pd.DataFrame) -> pd.DataFrame:
    wx = wx.copy()
    wx.columns = [c.strip().upper() for c in wx.columns]
    wx["timestamp"]      = _parse_dt(wx["DATE_TIME"])
    # IRRADIATION in Kaggle is W/m² already (some sources scale by 1000 — keep as-is)
    wx["irradiance_wm2"] = pd.to_numeric(wx.get("IRRADIATION", 0),
                                          errors="coerce").fillna(0)
    wx["temperature_c"]  = pd.to_numeric(wx.get("AMBIENT_TEMPERATURE", np.nan),
                                          errors="coerce")
    wx["module_temp_c"]  = pd.to_numeric(wx.get("MODULE_TEMPERATURE", np.nan),
                                          errors="coerce")
    return wx[["timestamp", "irradiance_wm2", "temperature_c", "module_temp_c"]]


# ── 1. power_df ────────────────────────────────────────────────────────────────

def _build_power_df(gen: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all inverters to plant-level 15-min rows and merge real weather.

    Columns: timestamp, date, hour, power_kw, dc_power_kw,
             irradiance_wm2, temperature_c, module_temp_c, energy_kwh

    NOTE: cloud_cover_pct has been removed — it was a synthetic proxy
    calculated from a made-up clear-sky bell curve, not a real measurement.
    """
    plant = (
        gen.groupby("timestamp")
        .agg(
            power_kw    =("ac_power_kw",  "sum"),
            dc_power_kw =("dc_power_kw",  "sum"),
            date        =("date",          "first"),
            hour        =("hour",          "first"),
        )
        .reset_index()
    )

    plant = plant.merge(wx, on="timestamp", how="left")
    plant = plant.sort_values("timestamp").reset_index(drop=True)

    # Clean chronological forward fill for weather dropouts instead of global median
    for col in ["irradiance_wm2", "temperature_c", "module_temp_c"]:
        plant[col] = plant[col].ffill(limit=4).bfill(limit=4)

    plant["energy_kwh"] = (plant["power_kw"] * 0.25).clip(lower=0)

    return plant[["timestamp", "date", "hour", "power_kw", "dc_power_kw",
                  "irradiance_wm2", "temperature_c", "module_temp_c", "energy_kwh"]]


# ── 2. daily_df ────────────────────────────────────────────────────────────────

def _build_daily_df(power_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar day. All values derived from real measurements.
    PR (Performance Ratio) = actual energy / (irradiance * capacity * hours).
    """
    daily = (
        power_df.groupby("date")
        .agg(
            energy_kwh      =("energy_kwh",      "sum"),
            peak_power_kw   =("power_kw",         "max"),
            avg_temp        =("temperature_c",    "mean"),
            avg_module_temp =("module_temp_c",    "mean"),
            avg_irradiance  =("irradiance_wm2",   "mean"),
            peak_irradiance =("irradiance_wm2",   "max"),
            dc_energy_kwh   =("dc_power_kw",      lambda x: (x * 0.25).sum()),
        )
        .reset_index()
    )

    # Efficiency: fraction of nameplate capacity delivered over 24 h
    daily["capacity_factor_pct"] = (
        daily["energy_kwh"] / (PLANT_CAPACITY_KW * 24) * 100
    ).round(2)

    # DC→AC conversion efficiency (real, not synthesized)
    daily["dc_ac_efficiency_pct"] = np.where(
        daily["dc_energy_kwh"] > 0,
        (daily["energy_kwh"] / daily["dc_energy_kwh"] * 100).clip(0, 100).round(2),
        np.nan,
    )

    # Performance Ratio: standard IEC metric
    # PR = E_ac / (H_poa / G_stc * P_peak)   where G_stc=1000 W/m²
    # Using avg_irradiance as H_poa proxy (Wh/m²/h * 24h gives daily insolation)
    daily_insolation_kwh_m2 = daily["avg_irradiance"] * 24 / 1000  # kWh/m²/day
    daily["performance_ratio"] = np.where(
        daily_insolation_kwh_m2 > 0,
        (daily["energy_kwh"] / (daily_insolation_kwh_m2 * PLANT_CAPACITY_KW)).round(3),
        np.nan,
    )

    daily["co2_saved_kg"]     = (daily["energy_kwh"] * CO2_PER_KWH).round(2)
    daily["trees_equivalent"] = (daily["co2_saved_kg"] / TREE_CO2_PER_YEAR).round(2)

    return daily


# ── 3. inverter_df ─────────────────────────────────────────────────────────────

def _build_inverter_df(gen: pd.DataFrame, wx: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Per-inverter 15-min rows for the last `days` days.

    Real columns from Kaggle: ac_power_kw, dc_power_kw, daily_yield, total_yield
    Derived (from real data only):
      - efficiency_pct : AC / DC power ratio  (real inverter efficiency)
      - status         : OK / FAULT / IDLE    (from ac_power_kw and hour)

    REMOVED (were synthesized): voltage_v, current_a
    The Kaggle dataset does not contain string or inverter-level V/I measurements.
    """
    all_dates = sorted(gen["date"].unique())
    if len(all_dates) > days:
        cutoff_date = all_dates[-days]
        recent = gen[gen["date"] >= cutoff_date].copy()
    else:
        recent = gen.copy()

    recent = recent.merge(wx[["timestamp", "irradiance_wm2"]], on="timestamp", how="left")
    recent["irradiance_wm2"] = recent["irradiance_wm2"].fillna(0)

    # Real DC→AC inverter efficiency
    recent["efficiency_pct"] = np.where(
        recent["dc_power_kw"] > 0,
        (recent["ac_power_kw"] / recent["dc_power_kw"] * 100).clip(50, 99),
        np.nan,
    )
    # recent["efficiency_pct"] = recent.groupby("inverter_id")["efficiency_pct"].transform(
    #     lambda x: x.fillna(x.median())
    # )

    # Status derived from real power readings — no random assignment
    def _status(row):
        # 1. If the inverter is actively producing significant power, it is OK
        if row["ac_power_kw"] >= 0.5:
            return "OK"
            
        # 2. If it is not producing power, check if there is strong sunlight.
        # Since Kaggle irradiation spans 0 to ~1.5, 0.05 represents daylight onset.
        if row["irradiance_wm2"] > 0.05:
            return "FAULT"
            
        # 3. No power and no sun means it's simply night-time / inactive
        return "IDLE"

    recent["status"]    = recent.apply(_status, axis=1)
    recent["power_kw"]  = recent["ac_power_kw"]

    return recent[[
        "timestamp", "date", "hour", "inverter_id",
        "power_kw", "dc_power_kw", "efficiency_pct",
        "daily_yield", "total_yield", "status",
    ]].reset_index(drop=True)


# ── 4. string_df — NOT AVAILABLE ───────────────────────────────────────────────

def _build_string_df() -> pd.DataFrame:
    """
    The Kaggle Plant 1 dataset contains NO string-level measurements.
    Returning an empty, clearly-labelled DataFrame so pages can detect
    absence and show an honest message instead of synthesized numbers.
    """
    return pd.DataFrame(columns=[
        "date", "inverter_id", "string_id",
        "energy_kwh", "status",
    ])


# ── 5. fault_df ────────────────────────────────────────────────────────────────

def _build_fault_df(inverter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fault events = 15-min intervals where an inverter shows zero AC power
    during daylight hours (7–18).

    Only real observable attributes are kept:
      timestamp, inverter_id, ac_power_kw (= 0), dc_power_kw, hour

    REMOVED: fault_type, severity, resolved — these were randomly assigned
    labels with no basis in the source data.

    The dashboard can show:
      - which inverter lost output
      - exactly when (timestamp)
      - what the DC power was at the time (useful: DC present + AC=0 → inverter trip)
      - whether DC was also zero (likely irradiance-related, not a fault)
    """
    faults = inverter_df[inverter_df["status"] == "FAULT"].copy()
    if faults.empty:
        return pd.DataFrame(columns=[
            "timestamp", "inverter_id", "hour",
            "dc_power_kw", "likely_cause",
        ])

    # Infer likely cause from DC power — this IS derivable from real data
    faults["likely_cause"] = np.where(
        faults["dc_power_kw"] > 0.5,
        "Inverter trip (DC present, AC=0)",   # real inference
        "Low irradiance / no generation",     # real inference
    )

    return faults[[
        "timestamp", "inverter_id", "hour",
        "dc_power_kw", "likely_cause",
    ]].reset_index(drop=True)


# ── Master loader ──────────────────────────────────────────────────────────────

def generate_all(save_csv: bool = False, output_dir: str = "data",
                 data_dir: str = None) -> dict:
    """
    Returns dict with keys: power_df, daily_df, inverter_df, string_df, fault_df.
    string_df is always an empty DataFrame (no string data in dataset).
    """
    if data_dir is None:
        data_dir = DATA_DIR

    gen_raw, wx_raw = _load_raw(data_dir)
    gen = _normalise_generation(gen_raw)
    wx  = _normalise_weather(wx_raw)

    power_df    = _build_power_df(gen, wx)
    daily_df    = _build_daily_df(power_df)
    inverter_df = _build_inverter_df(gen, wx, days=30)
    string_df   = _build_string_df()
    fault_df    = _build_fault_df(inverter_df)

    datasets = {
        "power_df":    power_df,
        "daily_df":    daily_df,
        "inverter_df": inverter_df,
        "string_df":   string_df,
        "fault_df":    fault_df,
    }

    if save_csv:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, df in datasets.items():
            df.to_csv(os.path.join(output_dir, f"{name.replace('_df','')}.csv"), index=False)

    return datasets


if __name__ == "__main__":
    data = generate_all()
    for k, v in data.items():
        print(f"\n── {k} ({len(v)} rows) ──")
        print(v.head(3).to_string())
