# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Smart Agriculture & Plantation Dashboard (GLOBAL/EN)
# Features:
# - Dark mode + gradient styling
# - Weather map (Folium) + OpenWeather precipitation overlay (if OWM_API_KEY in st.secrets)
# - Fetch weather from Open-Meteo
# - Daily / Weekly / Monthly charts (Plotly): precipitation, temp, humidity
# - Manual & Automatic yield calculation + 3-harvest yearly projection
# - Fertilizer recommendation (N/P/K)
# - Global commodity prices (USD/ton) table (editable) + Excel/PDF export
# - Citizen Reports (image upload + JSON persistence + delete)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import base64
from io import BytesIO
from datetime import datetime as dt, datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from PIL import Image

# Optional PDF export deps (we'll guard them)
try:
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

import pytz

# ───────────────────────── Config & Paths ─────────────────────────
st.set_page_config(page_title="Smart Agriculture Dashboard (Global)", layout="wide")

UPLOAD_DIR = "uploads"
DATA_DIR = "data"
CITIZEN_FILE = "citizen_reports.json"
TODO_FILE = "todo_harian.json"  # kept for compatibility if you want to reuse
COMMODITY_FILE = os.path.join(DATA_DIR, "global_commodity_prices.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ───────────────────────── State ─────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ───────────────────────── Theme / CSS ─────────────────────────
COLOR_GREEN_PADDY = "#d8f3dc"
COLOR_BLUE_DARK = "#1b263b"
COLOR_BLUE_SOFT = "#a8dadc"
COLOR_WHITE = "#ffffff"
COLOR_DARK_SOFT = "#2e3440"
COLOR_INPUT_DARK = "#3b4252"

if st.session_state.dark_mode:
    BACKGROUND = f"linear-gradient(135deg, {COLOR_BLUE_DARK} 75%, {COLOR_BLUE_SOFT} 25%)"
    SIDEBAR_BG = f"linear-gradient(180deg, {COLOR_BLUE_DARK}, {COLOR_BLUE_SOFT})"
    FONT_COLOR = COLOR_WHITE
    INPUT_BG = COLOR_INPUT_DARK
    INPUT_FONT = COLOR_WHITE
    EXPANDER_BG = "#38404f"
    TABLE_BG = "#3b4252"
    HOVER_BG = "#4c566a"
    PLACEHOLDER_COLOR = "#cccccc"
else:
    BACKGROUND = f"linear-gradient(135deg, {COLOR_GREEN_PADDY} 75%, {COLOR_WHITE} 25%)"
    SIDEBAR_BG = f"linear-gradient(180deg, {COLOR_GREEN_PADDY}, {COLOR_WHITE})"
    FONT_COLOR = "#1c1c1e"
    INPUT_BG = COLOR_WHITE
    INPUT_FONT = "#1c1c1e"
    EXPANDER_BG = "#f9f9f9"
    TABLE_BG = "#ffffff"
    HOVER_BG = "#f0f0f0"
    PLACEHOLDER_COLOR = "#888888"

st.markdown(f"""
<style>
html, body, .stApp {{
    background: {BACKGROUND};
    color: {FONT_COLOR};
}}
section[data-testid="stSidebar"] > div:first-child {{
    background: {SIDEBAR_BG};
    padding-top: 20px;
}}
section[data-testid="stSidebar"] * {{
    color: {FONT_COLOR} !important;
}}
input, textarea, select, button, .stTextInput input {{
    background-color: {INPUT_BG} !important;
    color: {INPUT_FONT} !important;
    border: 1px solid #777;
    border-radius: 6px;
}}
input:focus, textarea:focus, select:focus {{
    border: 1px solid #66AFE9 !important;
    outline: none;
}}
div[data-testid="stExpander"] {{
    background: {EXPANDER_BG} !important;
    border-radius: 10px;
    border: 1px solid #555;
    padding: 12px;
    color: {FONT_COLOR} !important;
}}
div[data-testid="stExpander"] summary {{
    color: {FONT_COLOR} !important;
    font-weight: bold;
}}
section div[data-testid="stFileUploaderDropzone"] {{
    background-color: {HOVER_BG} !important;
    border: 2px dashed #aaa !important;
    color: {FONT_COLOR} !important;
}}
section div[data-testid="stFileUploaderDropzone"] * {{
    color: {FONT_COLOR} !important;
}}
.stApp .dataframe, .stApp .stDataFrame, .stApp .stTable {{
    background-color: {TABLE_BG} !important;
    color: {FONT_COLOR} !important;
}}
.stApp .dataframe td, .stApp .dataframe th,
.stApp .stDataFrame td, .stApp .stDataFrame th {{
    color: {FONT_COLOR} !important;
}}
.stApp div, .stApp span, .stApp label, .stApp p,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
    color: {FONT_COLOR} !important;
}}
::placeholder {{
    color: {PLACEHOLDER_COLOR} !important;
}}
.stSelectbox div[data-baseweb="select"] > div,
[data-baseweb="popover"], [role="listbox"] {{
    background-color: {HOVER_BG} !important;
    color: {FONT_COLOR} !important;
    border: 1px solid #777;
}}
[data-baseweb="popover"] * {{
    color: {FONT_COLOR} !important;
}}
div[data-testid="stForm"] {{
    background-color: {EXPANDER_BG} !important;
    border: 1px solid #555 !important;
    border-radius: 10px;
    padding: 12px;
}}
.stApp .stCheckbox label {{
    color: {FONT_COLOR} !important;
}}
input, textarea {{ caret-color: {FONT_COLOR} !important; }}
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.checkbox("Dark Mode", value=st.session_state.dark_mode, key="dark_mode")

# ───────────────────────── Header ─────────────────────────
st.title("Smart Agriculture & Plantation Dashboard — Global")
st.caption("All prices in **USD/ton**. Weather data from Open-Meteo. Optional rain layer from OpenWeather.")

# ───────────────────────── Coordinates ─────────────────────────
LAT = st.sidebar.number_input("Latitude", value=-3.921406, format="%.6f")
LON = st.sidebar.number_input("Longitude", value=119.772731, format="%.6f")

# ───────────────────────── Rain Map ─────────────────────────
with st.expander("Real-time Rainfall Map"):
    m = folium.Map(location=[LAT, LON], zoom_start=10, control_scale=True)
    OWM_API_KEY = st.secrets.get("OWM_API_KEY", "")
    if OWM_API_KEY:
        tile_url = f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={OWM_API_KEY}"
        folium.TileLayer(
            tiles=tile_url,
            attr="© OpenWeatherMap",
            name="Precipitation",
            overlay=True,
            control=True,
            opacity=0.6,
        ).add_to(m)
    folium.Marker([LAT, LON], tooltip="Selected Location").add_to(m)
    st_folium(m, width="100%", height=420)

# ───────────────────────── Weather Fetch ─────────────────────────
weather_url = (
    f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&"
    "daily=temperature_2m_min,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean&"
    "hourly=temperature_2m,precipitation,relative_humidity_2m&timezone=auto"
)
try:
    resp = requests.get(weather_url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
except Exception as e:
    st.error("Failed to retrieve weather data. Please check your connection/coordinates.")
    data = {"daily": {"time": [], "temperature_2m_min": [], "temperature_2m_max": [],
                      "precipitation_sum": [], "relative_humidity_2m_mean": []},
            "hourly": {"time": [], "temperature_2m": [], "precipitation": [], "relative_humidity_2m": []}}

# ───────────────────────── DataFrames ─────────────────────────
# Daily dataframe
df_daily = pd.DataFrame({
    "Date": pd.to_datetime(data.get("daily", {}).get("time", [])),
    "Precipitation (mm)": np.round(data.get("daily", {}).get("precipitation_sum", []), 1),
    "Temp Max (°C)": np.round(data.get("daily", {}).get("temperature_2m_max", []), 1),
    "Temp Min (°C)": np.round(data.get("daily", {}).get("temperature_2m_min", []), 1),
    "Humidity (%)": np.round(data.get("daily", {}).get("relative_humidity_2m_mean", []), 1),
})

# Hourly dataframe
df_hourly = pd.DataFrame({
    "Time": pd.to_datetime(data.get("hourly", {}).get("time", [])),
    "Precipitation (mm)": data.get("hourly", {}).get("precipitation", []),
    "Temperature (°C)": data.get("hourly", {}).get("temperature_2m", []),
    "Humidity (%)": data.get("hourly", {}).get("relative_humidity_2m", []),
})

# Derived resamples (Weekly/Monthly) from hourly
if not df_hourly.empty:
    df_hourly = df_hourly.set_index("Time").sort_index()
    # Daily (from hourly) for consistent grouping (sums/means)
    df_daily_from_hourly = pd.DataFrame({
        "Precipitation (mm)": df_hourly["Precipitation (mm)"].resample("D").sum(),
        "Temperature (°C)": df_hourly["Temperature (°C)"].resample("D").mean(),
        "Humidity (%)": df_hourly["Humidity (%)"].resample("D").mean(),
    }).reset_index().rename(columns={"Time": "Date"})
    # Weekly
    df_weekly = pd.DataFrame({
        "Precipitation (mm)": df_hourly["Precipitation (mm)"].resample("W").sum(),
        "Temperature (°C)": df_hourly["Temperature (°C)"].resample("W").mean(),
        "Humidity (%)": df_hourly["Humidity (%)"].resample("W").mean(),
    }).reset_index().rename(columns={"Time": "Week"})
    # Monthly
    df_monthly = pd.DataFrame({
        "Precipitation (mm)": df_hourly["Precipitation (mm)"].resample("M").sum(),
        "Temperature (°C)": df_hourly["Temperature (°C)"].resample("M").mean(),
        "Humidity (%)": df_hourly["Humidity (%)"].resample("M").mean(),
    }).reset_index().rename(columns={"Time": "Month"})
else:
    df_daily_from_hourly = pd.DataFrame(columns=["Date","Precipitation (mm)","Temperature (°C)","Humidity (%)"])
    df_weekly = pd.DataFrame(columns=["Week","Precipitation (mm)","Temperature (°C)","Humidity (%)"])
    df_monthly = pd.DataFrame(columns=["Month","Precipitation (mm)","Temperature (°C)","Humidity (%)"])

# ───────────────────────── Irrigation Recommendation ─────────────────────────
threshold = st.sidebar.slider("Irrigation Threshold (mm)", 0, 20, 5)
if not df_daily.empty:
    df_daily["Irrigation Advice"] = df_daily["Precipitation (mm)"].apply(
        lambda x: "Irrigation Needed" if x < threshold else "Sufficient"
    )

# ───────────────────────── Tables & Downloads (Daily) ─────────────────────────
with st.expander("Daily Weather Table & Downloads"):
    st.dataframe(df_daily, use_container_width=True)

    # CSV
    csv = df_daily.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="daily_weather.csv", mime="text/csv")

    # Excel (BytesIO) — FIX for Streamlit download_button
    excel_buf = BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_daily.to_excel(writer, index=False, sheet_name="Daily")
        wb = writer.book
        ws = writer.sheets["Daily"]
        date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})
        ws.set_column("A:A", 15, date_fmt)
    excel_buf.seek(0)
    st.download_button(
        "Download Excel",
        data=excel_buf.getvalue(),
        file_name="daily_weather.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Simple HTML export (acts like PDF-like report if printed)
    html = df_daily.to_html(index=False)
    b64 = base64.b64encode(html.encode("utf-8")).decode("utf-8")
    st.markdown(
        f'<a href="data:text/html;base64,{b64}" download="daily_weather_report.html">📥 Download Weather Report (HTML)</a>',
        unsafe_allow_html=True
    )

# ───────────────────────── Charts: Daily / Weekly / Monthly ─────────────────────────
import plotly.express as px

with st.expander("Charts — Daily / Weekly / Monthly"):
    tabs = st.tabs(["Daily", "Weekly", "Monthly"])

    # DAILY (from API daily table)
    with tabs[0]:
        if df_daily.empty:
            st.warning("No daily data.")
        else:
            st.plotly_chart(px.bar(df_daily, x="Date", y="Precipitation (mm)", title="Daily Precipitation (mm)"), use_container_width=True)
            st.plotly_chart(px.line(df_daily, x="Date", y="Temp Max (°C)", title="Daily Max Temperature (°C)"), use_container_width=True)
            st.plotly_chart(px.line(df_daily, x="Date", y="Temp Min (°C)", title="Daily Min Temperature (°C)"), use_container_width=True)
            st.plotly_chart(px.line(df_daily, x="Date", y="Humidity (%)", title="Daily Humidity (%)"), use_container_width=True)

    # WEEKLY (from hourly resample)
    with tabs[1]:
        if df_weekly.empty:
            st.warning("No weekly data.")
        else:
            st.plotly_chart(px.bar(df_weekly, x="Week", y="Precipitation (mm)", title="Weekly Precipitation (mm)"), use_container_width=True)
            st.plotly_chart(px.line(df_weekly, x="Week", y="Temperature (°C)", title="Weekly Avg Temperature (°C)"), use_container_width=True)
            st.plotly_chart(px.line(df_weekly, x="Week", y="Humidity (%)", title="Weekly Avg Humidity (%)"), use_container_width=True)

    # MONTHLY (from hourly resample)
    with tabs[2]:
        if df_monthly.empty:
            st.warning("No monthly data.")
        else:
            st.plotly_chart(px.bar(df_monthly, x="Month", y="Precipitation (mm)", title="Monthly Precipitation (mm)"), use_container_width=True)
            st.plotly_chart(px.line(df_monthly, x="Month", y="Temperature (°C)", title="Monthly Avg Temperature (°C)"), use_container_width=True)
            st.plotly_chart(px.line(df_monthly, x="Month", y="Humidity (%)", title="Monthly Avg Humidity (%)"), use_container_width=True)

# ───────────────────────── Yield Model (Linear Regression demo) ─────────────────────────
model_df = pd.DataFrame({
    "Precipitation (mm)": [3.2, 1.0, 5.5, 0.0, 6.0],
    "Temp (°C)": [30, 32, 29, 31, 33],
    "Humidity (%)": [75, 80, 78, 82, 79],
    "Yield (kg/ha)": [5100, 4800, 5300, 4500, 5500],
})
yield_model = LinearRegression().fit(
    model_df.drop("Yield (kg/ha)", axis=1), model_df["Yield (kg/ha)"]
)

with st.expander("Yield Prediction"):
    # Manual (with weather inputs)
    st.subheader("Manual — Rice (With Weather Inputs)")
    ch_manual = st.number_input("Precipitation (mm)", value=5.0, key="manual_rain")
    t_manual = st.number_input("Max Temperature (°C)", value=32.0, key="manual_temp")
    h_manual = st.number_input("Humidity (%)", value=78.0, key="manual_hum")
    area_manual = st.number_input("Field Area (ha)", value=1.0, key="manual_area")
    price_manual = st.number_input("Rice Price (USD/kg)", value=0.45, key="manual_price")
    cost_manual = st.number_input("Production Cost per ha (USD)", value=350.0, key="manual_cost")

    pred_manual = float(yield_model.predict([[ch_manual, t_manual, h_manual]])[0])
    total_manual = pred_manual * area_manual
    revenue_manual = total_manual * price_manual
    net_profit_manual = revenue_manual - (cost_manual * area_manual)

    st.markdown(f"""
    - **Predicted Yield (Manual):** {pred_manual:,.0f} kg/ha  
    - **Total Yield:** {total_manual:,.0f} kg  
    - **Gross Revenue:** ${revenue_manual:,.2f}  
    - **Net Profit:** ${net_profit_manual:,.2f}
    """)

    # Generic (no weather) for multiple commodities
    st.subheader("Automatic (Quick) — Multi-Commodity (No Weather Inputs)")
    commodities_list = ["Rice", "Corn", "Coffee", "Cocoa", "Coconut", "Porang"]
    commodity_sel = st.selectbox("Select Commodity", commodities_list, key="auto_quick_commodity")
    default_yield = {
        "Rice": 5000,
        "Corn": 6000,
        "Coffee": 1200,
        "Cocoa": 1500,
        "Coconut": 2000,
        "Porang": 10000,
    }
    yield_per_ha = default_yield.get(commodity_sel, 5000)
    area2 = st.number_input("Field Area (ha)", value=1.0, key="auto_quick_area")
    price2 = st.number_input(f"{commodity_sel} Price (USD/kg)", value=0.45, key="auto_quick_price")
    cost2 = st.number_input("Production Cost per ha (USD)", value=350.0, key="auto_quick_cost")

    total2 = yield_per_ha * area2
    revenue2 = total2 * price2
    profit2 = revenue2 - (cost2 * area2)

    st.markdown(f"""
    - **Predicted Yield ({commodity_sel}):** {yield_per_ha:,.0f} kg/ha  
    - **Total Yield:** {total2:,.0f} kg  
    - **Gross Revenue:** ${revenue2:,.2f}  
    - **Net Profit:** ${profit2:,.2f}
    """)

    # Automatic (Rice) — uses current daily weather averages
    st.subheader("Automatic — Rice (Uses Current Daily Weather Averages)")
    area_auto = st.number_input("Field Area (ha) — Auto", value=1.0, key="auto_area")
    price_auto = st.number_input("Rice Price (USD/kg) — Auto", value=0.45, key="auto_price")
    cost_auto = st.number_input("Production Cost per ha (USD) — Auto", value=350.0, key="auto_cost")

    if not df_daily.empty:
        avg_inputs = df_daily[["Precipitation (mm)", "Temp Max (°C)", "Humidity (%)"]].mean().values.reshape(1, -1)
        pred_auto = float(yield_model.predict(avg_inputs)[0])
    else:
        pred_auto = 0.0

    total_auto = pred_auto * area_auto
    revenue_auto = total_auto * price_auto
    net_profit_auto = revenue_auto - (cost_auto * area_auto)

    st.markdown(f"""
    - **Predicted Yield (Auto):** {pred_auto:,.0f} kg/ha  
    - **Total Yield:** {total_auto:,.0f} kg  
    - **Gross Revenue:** ${revenue_auto:,.2f}  
    - **Net Profit:** ${net_profit_auto:,.2f}
    """)

    # 3-harvest projection (Rice)
    st.markdown("**Yearly Projection — Rice (3 Harvests)**")
    if not df_daily.empty:
        df_h1 = df_daily.head(7)
        df_h2 = df_daily[60:67] if len(df_daily) >= 67 else df_daily.tail(7)
        df_h3 = df_daily[120:127] if len(df_daily) >= 127 else df_daily.tail(7)

        p1 = float(yield_model.predict(df_h1[["Precipitation (mm)", "Temp Max (°C)", "Humidity (%)"]].mean().values.reshape(1, -1))[0])
        p2 = float(yield_model.predict(df_h2[["Precipitation (mm)", "Temp Max (°C)", "Humidity (%)"]].mean().values.reshape(1, -1))[0])
        p3 = float(yield_model.predict(df_h3[["Precipitation (mm)", "Temp Max (°C)", "Humidity (%)"]].mean().values.reshape(1, -1))[0])
    else:
        p1 = p2 = p3 = 0.0

    area_year = st.number_input("Field Area (ha) — Yearly", value=1.0, key="year_area")
    price_year = st.number_input("Rice Price (USD/kg) — Yearly", value=0.45, key="year_price")
    cost_year = st.number_input("Production Cost per ha (USD) — Yearly", value=350.0, key="year_cost")

    t1 = p1 * area_year
    t2 = p2 * area_year
    t3 = p3 * area_year
    total_year_yield = t1 + t2 + t3
    total_year_revenue = total_year_yield * price_year
    total_year_cost = cost_year * area_year * 3
    total_year_profit = total_year_revenue - total_year_cost

    st.write("**Harvest 1**")
    st.write(f"- Predicted: {p1:,.0f} kg/ha | Total: {t1:,.0f} kg | ${t1 * price_year:,.2f}")
    st.write("**Harvest 2**")
    st.write(f"- Predicted: {p2:,.0f} kg/ha | Total: {t2:,.0f} kg | ${t2 * price_year:,.2f}")
    st.write("**Harvest 3**")
    st.write(f"- Predicted: {p3:,.0f} kg/ha | Total: {t3:,.0f} kg | ${t3 * price_year:,.2f}")

    st.success(f"🟩 Total Yearly Yield: {total_year_yield:,.0f} kg | ${total_year_revenue:,.2f}")
    st.success(f"🟨 Net Yearly Profit: ${total_year_profit:,.2f}")

# ───────────────────────── Fertilizer Calculator ─────────────────────────
with st.expander("Fertilizer Calculator"):
    crop = st.selectbox("Select Commodity", ["Rice", "Corn", "Soybean", "Coffee", "Cocoa", "Coconut", "Porang"], key="fert_crop")
    area_f = st.number_input("Field Area (ha)", value=1.0, min_value=0.01, step=0.1, key="fert_area")

    # simple NPK recommendation (kg/ha), generic global-ish placeholders
    recommendations = {
        "Rice": {
            "Urea (N)": {"dose": 250, "desc": "Promotes vegetative growth (leaf & stem)"},
            "SP-36 (P)": {"dose": 100, "desc": "Improves root & tiller, enhances panicle yield"},
            "KCl (K)": {"dose": 100, "desc": "Improves pest/disease tolerance & grain quality"},
        },
        "Corn": {
            "Urea (N)": {"dose": 300, "desc": "Supports vegetative growth"},
            "SP-36 (P)": {"dose": 150, "desc": "Enhances root & ear formation"},
            "KCl (K)": {"dose": 100, "desc": "Improves kernel filling & stress tolerance"},
        },
        "Soybean": {
            "Urea (N)": {"dose": 100, "desc": "Low dose; legumes can fix nitrogen"},
            "SP-36 (P)": {"dose": 100, "desc": "Supports flowering & pod formation"},
            "KCl (K)": {"dose": 75, "desc": "Improves quality & shelf life"},
        },
        "Coffee": {
            "NPK": {"dose": 500, "desc": "Improves growth & berry production"},
        },
        "Cocoa": {
            "Urea (N)": {"dose": 150, "desc": "Leaf & pod growth"},
            "TSP (P)": {"dose": 100, "desc": "Flower & pod formation"},
            "KCl (K)": {"dose": 150, "desc": "Improves flavor & bean quality"},
        },
        "Coconut": {
            "NPK": {"dose": 300, "desc": "Improves growth & productivity"},
        },
        "Porang": {
            "Urea (N)": {"dose": 200, "desc": "Boosts leaf & corm growth"},
            "KCl (K)": {"dose": 100, "desc": "Improves corm size & weight"},
        },
    }

    rows = []
    for fert, meta in recommendations.get(crop, {}).items():
        total_dose = meta["dose"] * area_f
        rows.append({"Fertilizer": fert, "Total (kg)": round(total_dose, 2), "Function": meta["desc"]})

    df_fert = pd.DataFrame(rows)
    if not df_fert.empty:
        st.markdown("### Recommended Doses")
        st.dataframe(df_fert, use_container_width=True)
    else:
        st.info("No data available for this crop.")

# ───────────────────────── Global Commodity Prices (USD/ton) ─────────────────────────
def load_commodities():
    if os.path.exists(COMMODITY_FILE):
        try:
            with open(COMMODITY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    # default sample global-ish placeholders (USD/ton)
    return [
        {"Commodity": "Rice", "Price (USD/ton)": 520},
        {"Commodity": "Corn", "Price (USD/ton)": 260},
        {"Commodity": "Soybean", "Price (USD/ton)": 480},
        {"Commodity": "Wheat", "Price (USD/ton)": 310},
        {"Commodity": "Coffee", "Price (USD/ton)": 2100},
        {"Commodity": "Cocoa", "Price (USD/ton)": 3400},
        {"Commodity": "Palm Oil", "Price (USD/ton)": 950},
        {"Commodity": "Tea", "Price (USD/ton)": 3200},
        {"Commodity": "Sugar", "Price (USD/ton)": 450},
        {"Commodity": "Cotton", "Price (USD/ton)": 1900},
    ]

def save_commodities(data):
    with open(COMMODITY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if "commodities" not in st.session_state:
    st.session_state.commodities = load_commodities()

with st.expander("Global Commodity Prices (USD/ton)"):
    st.markdown("Edit directly in the table below and click **Save**:")
    df_edit = pd.DataFrame(st.session_state.commodities)

    edited_df = st.data_editor(
        df_edit,
        column_config={
            "Commodity": st.column_config.TextColumn("Commodity"),
            "Price (USD/ton)": st.column_config.NumberColumn("Price (USD/ton)", format="$%d"),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        key="editor_prices"
    )

    if st.button("Save Prices"):
        st.session_state.commodities = edited_df.to_dict(orient="records")
        save_commodities(st.session_state.commodities)
        st.success("✅ Global commodity prices updated.")

    # Show table
    st.dataframe(pd.DataFrame(st.session_state.commodities), use_container_width=True)

    # ── Export: Excel & PDF
    col1, col2 = st.columns(2)

    # Excel (BytesIO) — FIX
    excel_prices_buf = BytesIO()
    with pd.ExcelWriter(excel_prices_buf, engine="xlsxwriter") as writer:
        pd.DataFrame(st.session_state.commodities).to_excel(writer, index=False, sheet_name="Commodities")
    excel_prices_buf.seek(0)
    with col1:
        st.download_button(
            "⬇️ Download Prices (Excel)",
            data=excel_prices_buf.getvalue(),
            file_name="global_commodities.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # PDF (if reportlab available)
    def export_prices_pdf(df):
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("Global Commodity Prices (USD/ton)", styles["Heading1"]),
            Spacer(1, 12),
        ]
        data_tbl = [df.columns.tolist()] + df.values.tolist()
        table = Table(data_tbl)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        doc.build(elements)
        buf.seek(0)
        return buf

    with col2:
        if REPORTLAB_OK:
            pdf_buf = export_prices_pdf(pd.DataFrame(st.session_state.commodities))
            st.download_button(
                "⬇️ Download Prices (PDF)",
                data=pdf_buf,
                file_name="global_commodities.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF export requires `reportlab`. Install it to enable PDF downloads.")

# ───────────────────────── Citizen Reports ─────────────────────────
def load_reports():
    if os.path.exists(CITIZEN_FILE):
        try:
            with open(CITIZEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_reports(data):
    with open(CITIZEN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if "reports" not in st.session_state:
    st.session_state.reports = load_reports()

with st.expander("Citizen Reports"):
    with st.form("form_reports"):
        name = st.text_input("Name")
        contact = st.text_input("Contact")
        issue_type = st.selectbox("Type", ["Irrigation Issue", "Pest/Disease", "Weather Condition", "Other"])
        location = st.text_input("Location")
        description = st.text_area("Description")
        image = st.file_uploader("Upload Image (optional)", type=["png", "jpg", "jpeg"])
        send = st.form_submit_button("Submit")

        if send:
            if name.strip() and contact.strip() and description.strip():
                img_path = None
                if image is not None:
                    ext = os.path.splitext(image.name)[1]
                    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                    fpath = os.path.join(UPLOAD_DIR, fname)
                    with open(fpath, "wb") as f:
                        f.write(image.getbuffer())
                    img_path = fpath

                new_report = {
                    "Name": name.strip(),
                    "Contact": contact.strip(),
                    "Type": issue_type,
                    "Location": location.strip(),
                    "Description": description.strip(),
                    "Datetime": datetime.now(pytz.timezone("Asia/Makassar")).strftime("%d %B %Y %H:%M"),
                    "Image": img_path,
                }
                st.session_state.reports.append(new_report)
                save_reports(st.session_state.reports)
                st.success("Report submitted successfully.")
            else:
                st.warning("Please fill all required fields (Name, Contact, Description).")

    # List reports
    if not st.session_state.reports:
        st.info("No reports yet.")
    for i, rep in enumerate(st.session_state.reports):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(
                f"**{rep['Datetime']}**  \n"
                f"**{rep['Type']}** by *{rep['Name']}*  \n"
                f"**Location:** {rep['Location']}  \n"
                f"{rep['Description']}"
            )
            if rep.get("Image"):
                try:
                    img = Image.open(rep["Image"])
                    st.image(img, width=320)
                except Exception:
                    st.warning("Image cannot be displayed.")
        with col2:
            if st.button("🗑 Delete", key=f"del_rep_{i}"):
                if rep.get("Image") and os.path.exists(rep["Image"]):
                    try:
                        os.remove(rep["Image"])
                    except Exception:
                        pass
                st.session_state.reports.pop(i)
                save_reports(st.session_state.reports)
                st.experimental_rerun()

# ───────────────────────── Footer ─────────────────────────
st.markdown("---")
st.caption("© 2025 — Global Smart Agriculture Dashboard")
