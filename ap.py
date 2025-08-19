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

# ------------------ Custom CSS ------------------
def set_custom_style():
    dark = st.session_state.dark_mode

    if dark:
        sidebar_bg = "#4CAF50"  # green paddy
        body_bg = "linear-gradient(135deg, #0D47A1, #000000)"  # dark blue to black
        table_header = "#388E3C"
        font_color = "white"
        list_bg = "black"
        list_font = "white"
    else:
        sidebar_bg = "#0D47A1"  # deep blue
        body_bg = "linear-gradient(135deg, #A5D6A7, #ffffff)"  # green paddy to white
        table_header = "#1E88E5"
        font_color = "black"
        list_bg = "#81D4FA"  # light water blue
        list_font = "black"

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background: {sidebar_bg};
        }}
        .stApp {{
            background: {body_bg};
            color: {font_color};
        }}
        table.dataframe th {{
            background: {table_header};
            color: white !important;
        }}
        table.dataframe td {{
            background: linear-gradient(90deg, #0D47A1, #4CAF50, #81D4FA);
            color: {font_color} !important;
        }}
        .feature-list {{
            background: {list_bg};
            color: {list_font};
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 8px;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_style()

# ------------------ Functions ------------------
def plot_precipitation_map(lat, lon, rainfall):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker(
        location=[lat, lon],
        popup=f"Rainfall: {rainfall:.2f} mm",
        icon=folium.Icon(color="blue", icon="tint"),
    ).add_to(m)
    return m

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
    return float(model.predict(np.array([[rainfall, temperature, soil_moisture]]))[0])

def recommend_fertilizer(crop_type, soil_n, soil_p, soil_k):
    recs = {
        "rice": {"N": 100, "P": 50, "K": 50},
        "corn": {"N": 120, "P": 60, "K": 40},
        "soybean": {"N": 80, "P": 40, "K": 40},
        "wheat": {"N": 110, "P": 55, "K": 50},
        "cocoa": {"N": 90, "P": 50, "K": 60},
        "palm oil": {"N": 150, "P": 70, "K": 80}
    }
    target = recs.get(crop_type.lower(), {"N": 90, "P": 45, "K": 45})
    return {
        "N": max(target["N"] - soil_n, 0),
        "P": max(target["P"] - soil_p, 0),
        "K": max(target["K"] - soil_k, 0),
    }

def export_pdf(dataframe, filename="report.pdf"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Smart Agriculture & Plantation Report", styles['Title']), Spacer(1, 12)]
    table_data = [list(dataframe.columns)] + dataframe.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#A5D6A7")),
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------ Layout ------------------
st.title("🌱 Smart Agriculture & Plantation Dashboard")

# Weather Insights
st.markdown('<div class="feature-list">☁️ Weather Insights</div>', unsafe_allow_html=True)
lat = st.number_input("Latitude", value=-3.8)
lon = st.number_input("Longitude", value=120.0)
rainfall = st.slider("Rainfall (mm)", 0, 500, 150)
temperature = st.slider("Temperature (°C)", 10, 40, 26)
soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 35)
st_folium(plot_precipitation_map(lat, lon, rainfall), width=700, height=450)

# Yield Prediction
st.markdown('<div class="feature-list">📈 Yield Prediction</div>', unsafe_allow_html=True)
pred = predict_yield(rainfall, temperature, soil_moisture)
st.success(f"Estimated yield: {pred:.2f} tons/ha")

# Fertilizer Recommendation
st.markdown('<div class="feature-list">🌾 Fertilizer Recommendation</div>', unsafe_allow_html=True)
crop = st.selectbox("Select Crop", ["Rice", "Corn", "Soybean", "Wheat", "Cocoa", "Palm Oil"])
soil_n = st.number_input("Soil Nitrogen (N)", min_value=0)
soil_p = st.number_input("Soil Phosphorus (P)", min_value=0)
soil_k = st.number_input("Soil Potassium (K)", min_value=0)
if st.button("Get Recommendation"):
    rec = recommend_fertilizer(crop, soil_n, soil_p, soil_k)
    st.success(f"N={rec['N']}, P={rec['P']}, K={rec['K']}")

# Commodity Prices
st.markdown('<div class="feature-list">💹 Global Commodity Prices</div>', unsafe_allow_html=True)
commodities = {
    "Rice": 450,
    "Corn": 320,
    "Soybean": 580,
    "Wheat": 400,
    "Cocoa": 3100,
    "Palm Oil": 900
}
df_prices = pd.DataFrame(list(commodities.items()), columns=["Commodity", "Price (USD/ton)"])
st.dataframe(df_prices, use_container_width=True)

# Export Section
st.markdown('<div class="feature-list">📑 Export Report</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.download_button(
        "⬇️ Download Prices as Excel",
        df_prices.to_excel("commodities.xlsx", index=False),
        file_name="commodities.xlsx"
    ):
        st.success("Excel file exported successfully!")
with col2:
    pdf_buffer = export_pdf(df_prices)
    st.download_button(
        "⬇️ Download Prices as PDF",
        data=pdf_buffer,
        file_name="commodities.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.caption("🌍 Global Smart Agriculture – Responsive UI with Gradient Theme")
