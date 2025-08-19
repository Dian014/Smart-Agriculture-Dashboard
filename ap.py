import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64
from datetime import datetime as dt
from datetime import datetime
UPLOAD_DIR = "uploads"
LAPORAN_FILE = "laporan_warga.json"
import pytz
import subprocess
import json
import os
from PIL import Image
from rapidfuzz import process, fuzz

------------------ Constants ------------------

UPLOAD_DIR = "uploads" LAPORAN_FILE = "laporan_warga.json" PRICE_FILE = "data/commodity_prices.json" TODO_FILE = "daily_tasks.json"

------------------ Page config ------------------

st.set_page_config( page_title="AgriSphere — Global Smart Farming Intelligence", page_icon="🌾", layout="wide", )

------------------ Theme state ------------------

if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

------------------ Base colors ------------------

COLOR_GREEN = "#d8f3dc" COLOR_NAVY = "#1b263b" COLOR_SKY = "#a8dadc" COLOR_WHITE = "#ffffff" COLOR_SOFT_DARK = "#2e3440" COLOR_INPUT_DARK = "#3b4252"

------------------ Derived by mode --------------

if st.session_state.dark_mode: BACKGROUND = f"linear-gradient(135deg, {COLOR_NAVY} 75%, {COLOR_SKY} 25%)" SIDEBAR_BG = f"linear-gradient(180deg, {COLOR_NAVY}, {COLOR_SKY})" FONT_COLOR = COLOR_WHITE INPUT_BG = COLOR_INPUT_DARK INPUT_FONT = COLOR_WHITE EXPANDER_BG = "#38404f" TABLE_BG = "#3b4252" HOVER_BG = "#4c566a" PLACEHOLDER_COLOR = "#cccccc" else: BACKGROUND = f"linear-gradient(135deg, {COLOR_GREEN} 75%, {COLOR_WHITE} 25%)" SIDEBAR_BG = f"linear-gradient(180deg, {COLOR_GREEN}, {COLOR_WHITE})" FONT_COLOR = "#1c1c1e" INPUT_BG = COLOR_WHITE INPUT_FONT = "#1c1c1e" EXPANDER_BG = "#f9f9f9" TABLE_BG = "#ffffff" HOVER_BG = "#f0f0f0" PLACEHOLDER_COLOR = "#888888"

------------------ Global CSS -------------------

st.markdown( f"""

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

/* General input styling */
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

/* Expander styling */
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

/* Upload area */
section div[data-testid="stFileUploaderDropzone"] {{
    background-color: {HOVER_BG} !important;
    border: 2px dashed #aaa !important;
    color: {FONT_COLOR} !important;
}}
section div[data-testid="stFileUploaderDropzone"] * {{
    color: {FONT_COLOR} !important;
}}

/* Table styling */
.stApp .dataframe, .stApp .stDataFrame, .stApp .stTable {{
    background-color: {TABLE_BG} !important;
    color: {FONT_COLOR} !important;
}}
.stApp .dataframe td, .stApp .dataframe th,
.stApp .stDataFrame td, .stApp .stDataFrame th {{
    color: {FONT_COLOR} !important;
}}

/* All other texts */
.stApp div, .stApp span, .stApp label, .stApp p,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
    color: {FONT_COLOR} !important;
}}

/* Placeholder text */
::placeholder {{
    color: {PLACEHOLDER_COLOR} !important;
}}

/* Selectbox & dropdown */
.stSelectbox div[data-baseweb="select"] > div,
[data-baseweb="popover"], [role="listbox"] {{
    background-color: {HOVER_BG} !important;
    color: {FONT_COLOR} !important;
    border: 1px solid #777;
}}
[data-baseweb="popover"] * {{
    color: {FONT_COLOR} !important;
}}

/* Form container */
div[data-testid="stForm"] {{
    background-color: {EXPANDER_BG} !important;
    border: 1px solid #555 !important;
    border-radius: 10px;
    padding: 12px;
}}

/* Checkbox text */
.stApp .stCheckbox label {{
    color: {FONT_COLOR} !important;
}}

/* Cursor color */
input, textarea {{
    caret-color: {FONT_COLOR} !important;
}}
</style>""", unsafe_allow_html=True, )

------------------ Sidebar ------------------

with st.sidebar: st.checkbox("Dark Mode", value=st.session_state.dark_mode, key="dark_mode") st.markdown("Location (coordinates)") st.caption("Coordinates (global default at 0,0)") LAT = st.number_input("Latitude", value=0.0, format="%.6f") LON = st.number_input("Longitude", value=0.0, format="%.6f") threshold = st.slider("Irrigation Threshold (mm/day)", 0, 20, 5)

------------------ Header ------------------

st.title("AgriSphere • Global Smart Farming Intelligence") st.markdown( """ This dashboard provides global, location-based weather insights, irrigation guidance, yield prediction, fertilizer planning, and commodity price tracking for both crops and plantation commodities. """ )

------------------ Precipitation map ------------------

with st.expander("Real-time Precipitation Map"): m = folium.Map(location=[LAT, LON], zoom_start=6, control_scale=True) OWM_API_KEY = st.secrets.get("OWM_API_KEY", "") if OWM_API_KEY: tile_url = ( f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={OWM_API_KEY}" ) folium.TileLayer( tiles=tile_url, attr="© OpenWeatherMap", name="Precipitation", overlay=True, control=True, opacity=0.6, ).add_to(m) folium.Marker([LAT, LON], tooltip="Selected Location").add_to(m) st_folium(m, width="100%", height=400)

------------------ Fetch weather ------------------

weather_url = ( f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&" "daily=temperature_2m_min,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean&" "hourly=temperature_2m,precipitation,relative_humidity_2m&timezone=auto" ) try: resp = requests.get(weather_url, timeout=20) resp.raise_for_status() data = resp.json() except Exception as e: st.error(f"Failed to fetch weather data: {e}") st.stop()

------------------ Daily dataframe ------------------

df_daily = pd.DataFrame( { "Date": pd.to_datetime(data["daily"]["time"]), "Precipitation (mm)": np.round(data["daily"]["precipitation_sum"], 1), "Max Temp (°C)": np.round(data["daily"]["temperature_2m_max"], 1), "Min Temp (°C)": np.round(data["daily"]["temperature_2m_min"], 1), "Humidity (%)": np.round(data["daily"]["relative_humidity_2m_mean"], 1), } )

df_daily["Irrigation Recommendation"] = np.where( df_daily["Precipitation (mm)"] < threshold, "Irrigation Needed", "Sufficient" )

------------------ Show daily table ------------------

with st.expander("Daily Weather Table"): st.dataframe(df_daily, use_container_width=True)

csv = df_daily.to_csv(index=False).encode("utf-8") st.download_button("Download CSV", csv, "daily_weather.csv", "text/csv")

Excel export with xlsxwriter fallback to openpyxl

excel_io = BytesIO() try: with pd.ExcelWriter(excel_io, engine="xlsxwriter") as writer: df_daily.to_excel(writer, index=False, sheet_name="Daily Weather") workbook = writer.book worksheet = writer.sheets["Daily Weather"] date_format = workbook.add_format({"num_format": "yyyy-mm-dd"}) worksheet.set_column("A:A", 15, date_format) excel_bytes = excel_io.getvalue() except Exception: excel_io = BytesIO() with pd.ExcelWriter(excel_io, engine="openpyxl") as writer: df_daily.to_excel(writer, index=False, sheet_name="Daily Weather") excel_bytes = excel_io.getvalue()

st.download_button("Download Excel", data=excel_bytes, file_name="daily_weather.xlsx")

html_table = df_daily.to_html(index=False) b64 = base64.b64encode(html_table.encode("utf-8")).decode("utf-8") href = ( f'<a href="data:text/html;base64,{b64}" download="daily_weather.html">📥 Download Daily Weather (HTML)</a>' ) st.markdown(href, unsafe_allow_html=True)

------------------ Hourly dataframe ------------------

df_hourly = pd.DataFrame( { "Time": pd.to_datetime(data["hourly"]["time"]), "Precipitation (mm)": data["hourly"]["precipitation"], "Temperature (°C)": data["hourly"]["temperature_2m"], "Humidity (%)": data["hourly"]["relative_humidity_2m"], } )

------------------ Charts ------------------

with st.expander("Daily Charts"): st.plotly_chart( px.bar(df_daily, x="Date", y="Precipitation (mm)", title="Daily Precipitation"), use_container_width=True, ) st.plotly_chart( px.line(df_daily, x="Date", y="Max Temp (°C)", title="Daily Max Temperature"), use_container_width=True, ) st.plotly_chart( px.line(df_daily, x="Date", y="Min Temp (°C)", title="Daily Min Temperature"), use_container_width=True, ) st.plotly_chart( px.line(df_daily, x="Date", y="Humidity (%)", title="Daily Humidity"), use_container_width=True, )

------------------ Next 48 hours charts ------------------

now_ts = pd.Timestamp.now() df_next48 = df_hourly[df_hourly["Time"] > now_ts].head(48) with st.expander("Hourly Forecast (Next 48 Hours)"): if df_next48.empty: st.warning("No forward-looking hourly forecast available right now.") else: st.plotly_chart( px.line( df_next48, x="Time", y="Precipitation (mm)", title="Hourly Precipitation (Next 48 Hours)" ), use_container_width=True, ) st.plotly_chart( px.line( df_next48, x="Time", y="Temperature (°C)", title="Hourly Temperature (Next 48 Hours)" ), use_container_width=True, ) st.plotly_chart( px.line(df_next48, x="Time", y="Humidity (%)", title="Hourly Humidity (Next 48 Hours)"), use_container_width=True, )

------------------ Simple yield model ------------------

model_df = pd.DataFrame( { "Precipitation (mm)": [3.2, 1.0, 5.5, 0.0, 6.0], "Temperature (°C)": [30, 32, 29, 31, 33], "Humidity (%)": [75, 80, 78, 82, 79], "Yield (kg/ha)": [5100, 4800, 5300, 4500, 5500], } ) model = LinearRegression().fit( model_df.drop("Yield (kg/ha)", axis=1), model_df["Yield (kg/ha)"] )

------------------ Yield prediction ------------------

with st.expander("Yield Prediction"): # ---- Manual prediction with weather inputs (Rice) ---- st.subheader("Rice Yield (Manual Inputs)") ch_rice = st.number_input("Precipitation (mm)", value=5.0, key="rice_ch") t_rice = st.number_input("Max Temperature (°C)", value=32.0, key="rice_t") h_rice = st.number_input("Humidity (%)", value=78.0, key="rice_h") area_rice = st.number_input("Field Size (ha)", value=1.0, key="rice_area") price_rice = st.number_input("Price per kg (any currency)", value=1.0, key="rice_price") cost_per_ha_rice = st.number_input( "Production Cost per ha (same currency)", value=1000.0, key="rice_cost" )

pred_rice = float(model.predict([[ch_rice, t_rice, h_rice]])[0])
total_rice = pred_rice * area_rice
revenue_rice = total_rice * price_rice
profit_rice = revenue_rice - (cost_per_ha_rice * area_rice)

st.markdown(
    f"""

Predicted Rice Yield (Manual): {pred_rice:,.0f} kg/ha

Total Yield: {total_rice:,.0f} kg

Gross Revenue: {revenue_rice:,.2f}

Net Profit: {profit_rice:,.2f} """ )

---- Manual generic (Any commodity, default yields) ----

st.subheader("Automatic Yield (Generic Commodity, default yields)") commodities = [ "Rice", "Maize (Corn)", "Wheat", "Soybean", "Coffee", "Cocoa", "Oil Palm", "Rubber", "Tea", "Cotton", "Cassava", "Potato", ] commodity = st.selectbox("Select Commodity", commodities, key="generic_cmd") default_yield = { "Rice": 5000, "Maize (Corn)": 6000, "Wheat": 4500, "Soybean": 3000, "Coffee": 1200, "Cocoa": 1500, "Oil Palm": 2000, "Rubber": 1800, "Tea": 2500, "Cotton": 1600, "Cassava": 10000, "Potato": 25000, } yield_per_ha = float(default_yield.get(commodity, 5000)) area = st.number_input("Field Size (ha)", value=1.0, key="gen_area") price = st.number_input(f"Price per kg for {commodity}", value=1.0, key="gen_price") cost_per_ha = st.number_input("Production Cost per ha", value=1000.0, key="gen_cost")

total_yield = yield_per_ha * area revenue = total_yield * price profit = revenue - (cost_per_ha * area)

st.markdown( f"""

Predicted Yield {commodity}: {yield_per_ha:,.0f} kg/ha

Total Yield: {total_yield:,.0f} kg

Gross Revenue: {revenue:,.2f}

Net Profit: {profit:,.2f} """ )

---- Auto based on current daily weather (Rice) ----

st.subheader("Automatic Rice Yield (from Daily Weather Averages)") area_auto = st.number_input("Field Size (ha) (auto)", value=1.0, key="auto_area") price_auto = st.number_input("Price per kg (auto)", value=1.0, key="auto_price") cost_auto = st.number_input("Production Cost per ha (auto)", value=1000.0, key="auto_cost")

if not df_daily.empty: X_auto = ( df_daily[["Precipitation (mm)", "Max Temp (°C)", "Humidity (%)"]] .mean() .values.reshape(1, -1) ) pred_auto = float(model.predict(X_auto)[0]) else: pred_auto = 0.0

total_auto = pred_auto * area_auto revenue_auto = total_auto * price_auto profit_auto = revenue_auto - (cost_auto * area_auto)

st.markdown( f"""

Predicted Rice Yield (Auto): {pred_auto:,.0f} kg/ha

Total Yield: {total_auto:,.0f} kg

Gross Revenue: {revenue_auto:,.2f}

Net Profit: {profit_auto:,.2f} """ )

---- Annual projection (3 rice harvests) ----

st.markdown("Annual Rice Projection (3 harvests)") df_h1 = df_daily.head(7) x1 = ( df_h1[["Precipitation (mm)", "Max Temp (°C)", "Humidity (%)"]] .mean() .values.reshape(1, -1) ) p1 = float(model.predict(x1)[0])

df_h2 = df_daily[60:67] if len(df_daily) >= 67 else df_daily.tail(7) x2 = ( df_h2[["Precipitation (mm)", "Max Temp (°C)", "Humidity (%)"]] .mean() .values.reshape(1, -1) ) p2 = float(model.predict(x2)[0])

df_h3 = df_daily[120:127] if len(df_daily) >= 127 else df_daily.tail(7) x3 = ( df_h3[["Precipitation (mm)", "Max Temp (°C)", "Humidity (%)"]] .mean() .values.reshape(1, -1) ) p3 = float(model.predict(x3)[0])

area_year = st.number_input("Field Size (ha) (annual)", value=1.0, key="annual_area") price_year = st.number_input("Price per kg (annual)", value=1.0, key="annual_price") cost_year = st.number_input("Production Cost per ha (annual)", value=1000.0, key="annual_cost")

t1 = p1 * area_year t2 = p2 * area_year t3 = p3 * area_year total_y = t1 + t2 + t3 revenue_y = total_y * price_year total_cost_y = cost_year * area_year * 3 profit_y = revenue_y - total_cost_y

st.write("#### Harvest 1") st.write(f"- Predicted: {p1:,.0f} kg/ha | Total: {t1:,.0f} kg | {t1 * price_year:,.2f}") st.write("#### Harvest 2") st.write(f"- Predicted: {p2:,.0f} kg/ha | Total: {t2:,.0f} kg | {t2 * price_year:,.2f}") st.write("#### Harvest 3") st.write(f"- Predicted: {p3:,.0f} kg/ha | Total: {t3:,.0f} kg | {t3 * price_year:,.2f}")

st.success(f"🟩 Annual Total Yield: {total_y:,.0f} kg | Revenue: {revenue_y:,.2f}") st.success(f"🟨 Annual Net Profit: {profit_y:,.2f}")


------------------ Fertilizer calculator ------------------

with st.expander("Fertilizer Calculator"): crop = st.selectbox( "Select Commodity", [ "Rice", "Maize (Corn)", "Soybean", "Coffee", "Cocoa", "Oil Palm", "Rubber", "Tea", "Cotton", "Cassava", "Potato", ], key="fert_cmd", ) area_f = st.number_input("Field Size (ha)", value=1.0, min_value=0.01, step=0.1, key="fert_area")

# VERY simple, generic recommendations (illustrative only)
fert_reco = {
    "Rice": {
        "Urea": {"dose": 250, "function": "Boost vegetative growth"},
        "SP-36/TSP": {"dose": 100, "function": "Rooting and tillering"},
        "KCl": {"dose": 100, "function": "Disease tolerance & grain quality"},
    },
    "Maize (Corn)": {
        "Urea": {"dose": 300, "function": "Vegetative growth"},
        "SP-36/TSP": {"dose": 150, "function": "Ear formation"},
        "KCl": {"dose": 100, "function": "Kernel filling & tolerance"},
    },
    "Soybean": {
        "Urea": {"dose": 100, "function": "Low dose; N fixation"},
        "SP-36/TSP": {"dose": 100, "function": "Flowering & pod formation"},
        "KCl": {"dose": 75, "function": "Quality & shelf life"},
    },
    "Coffee": {"NPK": {"dose": 500, "function": "Overall growth & fruiting"}},
    "Cocoa": {
        "Urea": {"dose": 150, "function": "Leaf & pod growth"},
        "TSP": {"dose": 100, "function": "Flower & pod formation"},
        "KCl": {"dose": 150, "function": "Bean flavor & quality"},
    },
    "Oil Palm": {"NPK": {"dose": 300, "function": "Growth & productivity"}},
    "Rubber": {"NPK": {"dose": 250, "function": "Vegetative growth"}},
    "Tea": {"NPK": {"dose": 200, "function": "Leaf growth & yield"}},
    "Cotton": {"NPK": {"dose": 250, "function": "Vegetative & boll set"}},
    "Cassava": {"NPK": {"dose": 200, "function": "Root development"}},
    "Potato": {"NPK": {"dose": 250, "function": "Tuber development"}},
}

fert_rows = []
for fert, d in fert_reco.get(crop, {}).items():
    total_dose = d["dose"] * area_f
    fert_rows.append({
        "Fertilizer": fert,
        "Total (kg)": round(total_dose, 2),
        "Function": d["function"],
    })

df_fert = pd.DataFrame(fert_rows)
if not df_fert.empty:
    st.markdown("### Recommended Fertilization Plan")
    st.markdown(df_fert.to_html(classes='styled-table', index=False), unsafe_allow_html=True)
else:
    st.info("No fertilizer data available for this crop.")

------------------ Commodity Prices ------------------

os.makedirs("data", exist_ok=True)

Load & save helpers

def load_prices(): if os.path.exists(PRICE_FILE): with open(PRICE_FILE, "r", encoding="utf-8") as f: try: return json.load(f) except json.JSONDecodeError: pass # Global sample defaults (edit freely) return [ {"Commodity": "Rice", "Price per kg": 1.0}, {"Commodity": "Maize (Corn)", "Price per kg": 0.8}, {"Commodity": "Wheat", "Price per kg": 0.7}, {"Commodity": "Soybean", "Price per kg": 0.9}, {"Commodity": "Coffee", "Price per kg": 3.0}, {"Commodity": "Cocoa", "Price per kg": 2.5}, {"Commodity": "Oil Palm", "Price per kg": 0.5}, {"Commodity": "Rubber", "Price per kg": 1.2}, {"Commodity": "Tea", "Price per kg": 2.0}, ]

def save_prices(data): with open(PRICE_FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

if "commodity_prices" not in st.session_state: st.session_state.commodity_prices = load_prices()

with st.expander("Commodity Prices (editable)"): st.markdown("Edit prices directly in the table (free units/currency).") df_edit = pd.DataFrame(st.session_state.commodity_prices)

edited_df = st.data_editor(
    df_edit,
    column_config={
        "Commodity": st.column_config.TextColumn("Commodity"),
        "Price per kg": st.column_config.NumberColumn("Price per kg"),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="dynamic",
    key="editor_prices",
)

if st.button("Save Prices"):
    st.session_state.commodity_prices = edited_df.to_dict(orient="records")
    save_prices(st.session_state.commodity_prices)
    st.success("✅ Commodity prices updated.")

------------------ Field Reports ------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)

Load/save helpers for reports

def load_reports(filename): if os.path.exists(filename): with open(filename, "r", encoding="utf-8") as f: try: return json.load(f) except json.JSONDecodeError: return [] return []

def save_reports(filename, data): with open(filename, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)

if "reports" not in st.session_state: st.session_state.reports = load_reports(LAPORAN_FILE)

if "reports_updated" not in st.session_state: st.session_state.reports_updated = False

with st.expander("Field Reports"): with st.form("report_form"): name = st.text_input("Name") contact = st.text_input("Contact") issue_type = st.selectbox("Type", ["Irrigation Issue", "Pest/Disease", "Weather Condition", "Other"]) location = st.text_input("Location (free text)") desc = st.text_area("Description") image = st.file_uploader("Upload Image (optional)", type=["png", "jpg", "jpeg"]) submit = st.form_submit_button("Submit Report")

if submit:
    if name.strip() and contact.strip() and desc.strip():
        img_path = None
        if image is not None:
            ext = os.path.splitext(image.name)[1]
            filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{ext}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(image.getbuffer())
            img_path = filepath

        new_report = {
            "Name": name.strip(),
            "Contact": contact.strip(),
            "Type": issue_type,
            "Location": location.strip(),
            "Description": desc.strip(),
            "Date": datetime.utcnow().strftime("%d %B %Y %H:%M UTC"),
            "Image": img_path,
        }
        st.session_state.reports.append(new_report)
        save_reports(LAPORAN_FILE, st.session_state.reports)
        st.session_state.reports_updated = True
        st.success("Report submitted.")
    else:
        st.warning("Please complete all required fields before submitting.")

# Show reports
for i, r in enumerate(st.session_state.reports):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(
            f"{r['Date']}**  \n"
            f"{r['Type']} by {r['Name']}  \n"
            f"{r['Location']}  \n"
            f"{r['Description']}"
        )
        if r.get("Image"):
            try:
                img = Image.open(r["Image"])
                st.image(img, width=300)
            except Exception:
                st.warning("Image preview unavailable.")
    with col2:
        if st.button("🗑 Delete", key=f"del_report_{i}"):
            if r.get("Image") and os.path.exists(r["Image"]):
                try:
                    os.remove(r["Image"])
                except Exception:
                    pass
            st.session_state.reports.pop(i)
            save_reports(LAPORAN_FILE, st.session_state.reports)
            st.session_state.reports_updated = True
            st.rerun()

------------------ Daily Reminders ------------------

def load_todo(): if os.path.exists(TODO_FILE): with open(TODO_FILE, "r", encoding="utf-8") as f: try: return json.load(f) except json.JSONDecodeError: return [] return []

def save_todo(data): with open(TODO_FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)

if "todo" not in st.session_state: st.session_state.todo = load_todo()

with st.expander("Daily Tasks"): new_task = st.text_input("Add New Task:") if st.button("✅ Save Task"): if new_task.strip(): st.session_state.todo.append(new_task.strip()) save_todo(st.session_state.todo) st.success("Task saved.") else: st.warning("Task cannot be empty.")

for i, task in enumerate(st.session_state.todo):
    c1, c2 = st.columns([0.9, 0.1])
    c1.markdown(f"- {task}")
    if c2.button("🗑", key=f"del_task_{i}"):
        st.session_state.todo.pop(i)
        save_todo(st.session_state.todo)
        st.rerun()

------------------ Footer ------------------

st.markdown("---") st.caption("© 2025 – AgriSphere")
