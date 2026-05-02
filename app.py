from pathlib import Path
import streamlit as st
import pandas as pd
import time
import requests

from src.preprocess import load_data, select_features, fit_scalers, transform_features, create_sequences
from src.agent import load_lstm_model, IrrigationAgent
from src.config import SEQ_LENGTH, TARGET_COL, MIN_THRESHOLD, SAFETY_THRESHOLD, RAIN_LIMIT


WATER_MIN_SECURITY = 15
JETSON_API_URL = "http://192.168.137.86:5000/status"


@st.cache_resource
def load_system():
    project_root = Path(__file__).resolve().parent

    csv_path = project_root / "data" / "extended_soil_moisture.csv"
    model_path = project_root / "models" / "best_model_lstm.pt"

    df = load_data(str(csv_path))
    feature_cols = select_features(df)

    train_end = int(len(df) * 0.70)
    train_df = df.iloc[:train_end].copy()

    x_scaler, y_scaler = fit_scalers(train_df, feature_cols)
    X_scaled, y_scaled = transform_features(df, feature_cols, x_scaler, y_scaler)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

    model, checkpoint = load_lstm_model(str(model_path), device="cpu")
    agent = IrrigationAgent(model=model, y_scaler=y_scaler, device="cpu")

    return df, X_seq, agent


st.set_page_config(page_title="Smart Irrigation Agent", layout="wide")
st.title("🌱 Smart Irrigation Agent - LSTM + Jetson + ESP32 Demo")

df, X_seq, agent = load_system()
last_sequence = X_seq[-1]
real_current_moisture = float(df[TARGET_COL].iloc[-1])

st.sidebar.header("Mode de démonstration")

demo_mode = st.sidebar.selectbox(
    "Choisir le mode",
    ["Simulation software", "Live Jetson API"]
)

rain_forecast = st.sidebar.slider("Rain forecast (mm)", 0.0, 10.0, 0.0, 0.1)

current_moisture = real_current_moisture
water_level = 100.0
predicted_moisture = None
final_action = "PUMP_OFF"
final_reason = "No data"
pump_status = "OFF"
auto_refresh = False


if demo_mode == "Simulation software":
    current_moisture = st.sidebar.slider(
        "Current soil moisture",
        0.0,
        100.0,
        real_current_moisture,
        0.1
    )

    water_level = st.sidebar.slider(
        "Water level",
        0.0,
        100.0,
        80.0,
        0.1
    )

    if st.sidebar.button("Test DRY scenario"):
        current_moisture = 20.0
        rain_forecast = 0.0

    if st.sidebar.button("Test RAIN scenario"):
        current_moisture = 20.0
        rain_forecast = 5.0

    if st.sidebar.button("Test LOW WATER"):
        water_level = 5.0

    result = agent.run(
        sequence=last_sequence,
        current_moisture=current_moisture,
        rain_forecast=rain_forecast
    )

    predicted_moisture = result["predicted_moisture"]
    final_action = result["action"]
    final_reason = result["reason"]

    if water_level < WATER_MIN_SECURITY:
        final_action = "PUMP_OFF"
        final_reason = "Water tank level too low - pump protected"

    pump_status = "ON" if final_action == "PUMP_ON" else "OFF"


else:
    st.sidebar.subheader("Jetson API")
    jetson_url = st.sidebar.text_input("Jetson API URL", JETSON_API_URL)
    auto_refresh = st.sidebar.checkbox("Auto refresh live", value=True)

    try:
        response = requests.get(jetson_url, timeout=3)
        data = response.json()

        current_moisture = float(data.get("soil_moisture", 0.0))
        water_level = float(data.get("water_level", 0.0))
        predicted_moisture = float(data.get("predicted_moisture", 0.0))
        final_action = data.get("action", "PUMP_OFF")
        final_reason = data.get("reason", "No reason")
        pump_status = data.get("pump_status", "OFF")

        st.sidebar.success("Connected to Jetson API")

    except Exception as e:
        st.sidebar.error("Cannot connect to Jetson API")
        st.error(f"Connection error: {e}")
        predicted_moisture = 0.0


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Current moisture", f"{current_moisture:.2f}%")
col2.metric("Predicted moisture", f"{predicted_moisture:.2f}%")
col3.metric("Water level", f"{water_level:.2f}%")

if final_action == "PUMP_ON":
    col4.metric("Decision", "PUMP_ON", delta="Irrigation ON 💧")
else:
    col4.metric("Decision", "PUMP_OFF", delta="No irrigation ✅")

col5.metric("Pump status", pump_status)

st.subheader("Decision explanation")
st.write(f"**Reason:** {final_reason}")

if water_level < WATER_MIN_SECURITY:
    st.error("🚨 Water level is too low. Pump is blocked for safety.")

if final_action == "PUMP_ON":
    st.success("💧 Pump status: ON")
    st.markdown("### 🚿 Irrigation is activated")
else:
    st.info("✅ Pump status: OFF")
    st.markdown("### 🌿 No irrigation needed")

st.subheader("Agent thresholds")
st.write(f"- Minimum threshold: {MIN_THRESHOLD}")
st.write(f"- Safety threshold: {SAFETY_THRESHOLD}")
st.write(f"- Rain limit: {RAIN_LIMIT} mm")
st.write(f"- Water safety threshold: {WATER_MIN_SECURITY}%")

st.subheader("System command / API data")
st.json({
    "source": demo_mode,
    "action": final_action,
    "current_moisture": round(current_moisture, 2),
    "water_level": round(water_level, 2),
    "predicted_moisture": round(predicted_moisture, 2),
    "rain_forecast": rain_forecast,
    "pump_status": pump_status,
    "reason": final_reason
})

st.subheader("Last soil moisture values")
chart_df = df[[TARGET_COL]].tail(50).reset_index(drop=True)
st.line_chart(chart_df)

st.subheader("Latest dataset rows")
st.dataframe(df.tail(10), use_container_width=True)

if demo_mode == "Live Jetson API" and auto_refresh:
    time.sleep(1)
    st.rerun()