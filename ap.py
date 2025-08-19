import streamlit as st
import os
import json
import numpy as np
import folium
import pandas as pd
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from datetime import datetime as dt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Smart Agriculture & Plantation Dashboard",
    layout="wide"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ------------------ Constants ------------------
UPLOAD_DIR = "uploads"
REPORT_FILE = "global_reports.json"
PRICE_FILE = "global_commodity_prices.json"
TODO_FILE = "daily_tasks.json"

# ------------------ Functions ------------------

# Precipitation Map
def plot_precipitation_map(lat, lon, rainfall):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        location=[lat, lon],
        popup=f"Rainfall: {rainfall:.2f} mm",
        icon=folium.Icon(color="blue", icon="tint"),
    ).add_to(m)
    return m

# Yield Prediction
def predict_yield(rainfall, temperature, soil_moisture):
    X = np.array([
        [100, 25, 30],
        [200, 27, 40],
        [150, 26, 35],
        [300, 28, 45]
    ])
    y = np.array([2.5, 3.0, 2.8, 3.5])

    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(np.array([[rainfall, temperature, soil_moisture]]))
    return float(pred[0])

# Fertilizer Recommendation
def recommend_fertilizer(crop_type, soil_n, soil_p, soil_k):
    recs = {
        "rice": {"N": 100, "P": 50, "K": 50},
        "corn": {"N": 120, "P": 60, "K": 40},
        "soybean": {"N": 80, "P": 40, "K": 40},
        "wheat": {"N": 110, "P": 55, "K": 50},
    }
    target = recs.get(crop_type.lower(), {"N": 90, "P": 45, "K": 45})
    return {
        "N": max(target["N"] - soil_n, 0),
        "P": max(target["P"] - soil_p, 0),
        "K": max(target["K"] - soil_k, 0),
    }

# Commodity Prices
def load_prices():
    if not os.path.exists(PRICE_FILE):
        return []
    with open(PRICE_FILE, "r") as f:
        return json.load(f)

def save_prices(data):
    with open(PRICE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_price(commodity, price):
    data = load_prices()
    now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    data.append({"commodity": commodity, "price": price, "timestamp": now})
    save_prices(data)

# Reports
def load_reports():
    if not os.path.exists(REPORT_FILE):
        return []
    with open(REPORT_FILE, "r") as f:
        return json.load(f)

def save_reports(data):
    with open(REPORT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_report(report_type, description):
    reports = load_reports()
    now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    reports.append({"type": report_type, "description": description, "timestamp": now})
    save_reports(reports)

# Task Manager
def load_tasks():
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE, "r") as f:
        return json.load(f)

def save_tasks(data):
    with open(TODO_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_task(task):
    tasks = load_tasks()
    tasks.append({"task": task, "done": False})
    save_tasks(tasks)

def toggle_task(idx):
    tasks = load_tasks()
    if 0 <= idx < len(tasks):
        tasks[idx]["done"] = not tasks[idx]["done"]
    save_tasks(tasks)

# PDF Export
def generate_pdf(prices_df, reports_df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("🌍 Smart Agriculture & Plantation Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Prices
    if not prices_df.empty:
        elements.append(Paragraph("💹 Commodity Prices", styles["Heading2"]))
        table_data = [list(prices_df.columns)] + prices_df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E8F5E9"))
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    # Reports
    if not reports_df.empty:
        elements.append(Paragraph("📝 Reports", styles["Heading2"]))
        table_data = [list(reports_df.columns)] + reports_df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2196F3")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E3F2FD"))
        ]))
        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ------------------ Layout ------------------

st.title("🌱 Smart Agriculture & Plantation Dashboard")

# Weather Insights
st.header("☁️ Weather Insights")
lat = st.number_input("Latitude", value=-3.8)
lon = st.number_input("Longitude", value=120.0)
rainfall = st.slider("Rainfall (mm)", 0, 500, 150)
temperature = st.slider("Temperature (°C)", 10, 40, 26)
soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 35)

map_obj = plot_precipitation_map(lat, lon, rainfall)
st_folium(map_obj, width=700, height=450)

# Yield Prediction
st.header("📈 Yield Prediction")
pred_yield = predict_yield(rainfall, temperature, soil_moisture)
st.success(f"Estimated yield: {pred_yield:.2f} tons/ha")

# Fertilizer
st.header("🌾 Fertilizer Recommendation")
crop = st.selectbox("Select Crop", ["Rice", "Corn", "Soybean", "Wheat"])
soil_n = st.number_input("Soil Nitrogen (N)", min_value=0)
soil_p = st.number_input("Soil Phosphorus (P)", min_value=0)
soil_k = st.number_input("Soil Potassium (K)", min_value=0)

if st.button("Get Recommendation"):
    rec = recommend_fertilizer(crop, soil_n, soil_p, soil_k)
    st.success(f"Recommended additional nutrients for {crop}: N={rec['N']}, P={rec['P']}, K={rec['K']}")

# Commodity Prices
st.header("💹 Global Commodity Prices")
commodity_options = [
    "Rice", "Corn", "Soybean", "Wheat",
    "Coffee", "Cocoa", "Palm Oil", "Rubber",
    "Tea", "Sugarcane", "Cotton", "Tobacco",
    "Clove", "Pepper", "Nutmeg", "Cinnamon"
]
selected_commodity = st.selectbox("Select Commodity", commodity_options)
price = st.number_input("Enter price (USD/ton)", min_value=0.0, step=1.0)

if st.button("Add Price"):
    update_price(selected_commodity, price)
    st.success(f"✅ Added price for {selected_commodity}: ${price:.2f} per ton")

st.subheader("📊 Latest Prices")
prices = load_prices()
if prices:
    df_prices = pd.DataFrame(prices)
    st.dataframe(df_prices)
else:
    st.info("No price data available.")
    df_prices = pd.DataFrame()

# Reports
st.header("📝 Reports")
report_type = st.selectbox("Report Type", ["Pest & Disease", "Weather Anomaly", "Logistics", "Market"])
description = st.text_area("Description")

if st.button("Submit Report"):
    add_report(report_type, description)
    st.success("Report submitted successfully!")

st.subheader("📂 All Reports")
reports = load_reports()
if reports:
    df_reports = pd.DataFrame(reports)
    st.dataframe(df_reports)
else:
    st.info("No reports yet.")
    df_reports = pd.DataFrame()

# Export Section
st.header("⬇️ Export Data")
col1, col2, col3 = st.columns(3)
with col1:
    if not df_prices.empty:
        st.download_button(
            "Download Prices (CSV)",
            df_prices.to_csv(index=False).encode("utf-8"),
            "commodity_prices.csv",
            "text/csv"
        )
with col2:
    if not df_reports.empty:
        st.download_button(
            "Download Reports (CSV)",
            df_reports.to_csv(index=False).encode("utf-8"),
            "reports.csv",
            "text/csv"
        )
with col3:
    if not (df_prices.empty and df_reports.empty):
        pdf_file = generate_pdf(df_prices, df_reports)
        st.download_button(
            "📄 Download PDF Report",
            pdf_file,
            "global_agriculture_report.pdf",
            "application/pdf"
        )

# Task Manager
st.header("✅ Daily Tasks")
task = st.text_input("Add a new task")
if st.button("Add Task"):
    add_task(task)
    st.success("Task added!")

tasks = load_tasks()
for idx, t in enumerate(tasks):
    if st.checkbox(t["task"], value=t["done"], key=idx):
        toggle_task(idx)

# ------------------ Footer ------------------
st.markdown("---")
st.caption("🌍 Smart Agriculture & Plantation Dashboard – Global Edition")
