from pathlib import Path
import streamlit as st
import pandas as pd
import time

from src.preprocess import load_data, select_features, fit_scalers, transform_features, create_sequences
from src.agent import load_lstm_model, IrrigationAgent
from src.config import SEQ_LENGTH, TARGET_COL, MIN_THRESHOLD, SAFETY_THRESHOLD, RAIN_LIMIT
from src.hardware_live import ESP32Connection


WATER_MIN_SECURITY = 15


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
st.title("🌱 Smart Irrigation Agent - LSTM + ESP32 Demo")

df, X_seq, agent = load_system()
last_sequence = X_seq[-1]
real_current_moisture = float(df[TARGET_COL].iloc[-1])

if "esp32" not in st.session_state:
    st.session_state.esp32 = None

if "control_mode" not in st.session_state:
    st.session_state.control_mode = "AUTO"

if "manual_pump_state" not in st.session_state:
    st.session_state.manual_pump_state = "OFF"

st.sidebar.header("Mode de démonstration")

demo_mode = st.sidebar.selectbox(
    "Choisir le mode",
    ["Simulation software", "Live ESP32"]
)

rain_forecast = st.sidebar.slider("Rain forecast (mm)", 0.0, 10.0, 0.0, 0.1)

st.sidebar.header("Contrôle irrigation")
st.session_state.control_mode = st.sidebar.radio(
    "Mode de contrôle",
    ["AUTO", "MANUAL"],
    index=0 if st.session_state.control_mode == "AUTO" else 1
)

current_moisture = real_current_moisture
water_level = 100.0
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

else:
    port = st.sidebar.text_input("ESP32 COM Port", "COM5")
    auto_refresh = st.sidebar.checkbox("Auto refresh live", value=True)

    if st.sidebar.button("Connect ESP32"):
        try:
            st.session_state.esp32 = ESP32Connection(port=port, baud=115200)
            st.session_state.esp32.connect()
            st.sidebar.success("ESP32 connected")
        except Exception as e:
            st.sidebar.error(f"Connection error: {e}")

    if st.session_state.esp32 is not None:
        m, w = st.session_state.esp32.read_sensors()

        if m is not None:
            current_moisture = m
            water_level = w
        else:
            st.warning("Waiting for ESP32 data...")
    else:
        st.warning("ESP32 not connected.")


result = agent.run(
    sequence=last_sequence,
    current_moisture=current_moisture,
    rain_forecast=rain_forecast
)

final_action = result["action"]
final_reason = result["reason"]

if water_level < WATER_MIN_SECURITY:
    final_action = "PUMP_OFF"
    final_reason = "Water tank level too low - pump protected"

if st.session_state.control_mode == "MANUAL":
    st.subheader("Manual pump control")

    col_on, col_off = st.columns(2)

    with col_on:
        if st.button("💧 Pump ON"):
            st.session_state.manual_pump_state = "ON"

    with col_off:
        if st.button("🛑 Pump OFF"):
            st.session_state.manual_pump_state = "OFF"

    if st.session_state.manual_pump_state == "ON":
        if water_level < WATER_MIN_SECURITY:
            final_action = "PUMP_OFF"
            final_reason = "Manual ON blocked: water level too low"
        else:
            final_action = "PUMP_ON"
            final_reason = "Manual mode: user activated pump"
    else:
        final_action = "PUMP_OFF"
        final_reason = "Manual mode: user stopped pump"


if demo_mode == "Live ESP32" and st.session_state.get("esp32") is not None:
    st.session_state.esp32.send_decision(final_action)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Current moisture", f"{current_moisture:.2f}%")
col2.metric("Predicted moisture", f"{result['predicted_moisture']:.2f}%")
col3.metric("Water level", f"{water_level:.2f}%")

if final_action == "PUMP_ON":
    col4.metric("Decision", "PUMP_ON", delta="Irrigation ON 💧")
else:
    col4.metric("Decision", "PUMP_OFF", delta="No irrigation ✅")

col5.metric("Mode", st.session_state.control_mode)

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

st.subheader("ESP32 command")
st.json({
    "mode": st.session_state.control_mode,
    "action": final_action,
    "duration_sec": result["duration_sec"] if final_action == "PUMP_ON" else 0,
    "current_moisture": round(current_moisture, 2),
    "water_level": round(water_level, 2),
    "predicted_moisture": round(result["predicted_moisture"], 2),
    "rain_forecast": rain_forecast,
    "reason": final_reason
})

st.subheader("Last soil moisture values")
chart_df = df[[TARGET_COL]].tail(50).reset_index(drop=True)
st.line_chart(chart_df)

st.subheader("Latest dataset rows")
st.dataframe(df.tail(10), use_container_width=True)

if demo_mode == "Live ESP32" and auto_refresh:
    time.sleep(1)
    st.rerun()