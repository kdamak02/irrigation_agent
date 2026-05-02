import streamlit as st
import requests
import time

JETSON_URL = "http://192.168.137.86:5000/status"
JETSON_CONTROL_URL = "http://192.168.137.86:5000/control"

st.set_page_config(page_title="Smart Irrigation", layout="wide")
st.title("🌱 Smart Irrigation System (Jetson + ESP32)")

# ================================
# Sidebar (Control Panel)
# ================================
st.sidebar.header("Control Panel")

mode = st.sidebar.radio("Mode", ["AUTO", "MANUAL"])

auto_refresh = st.sidebar.checkbox("Auto refresh", True)

# ================================
# Manual Control Buttons
# ================================
if mode == "MANUAL":
    st.sidebar.subheader("Manual Pump Control")

    if st.sidebar.button("💧 Pump ON"):
        try:
            requests.post(JETSON_CONTROL_URL, json={
                "mode": "MANUAL",
                "action": "PUMP_ON"
            })
        except:
            st.error("Failed to send ON command")

    if st.sidebar.button("🛑 Pump OFF"):
        try:
            requests.post(JETSON_CONTROL_URL, json={
                "mode": "MANUAL",
                "action": "PUMP_OFF"
            })
        except:
            st.error("Failed to send OFF command")

else:
    # AUTO mode
    try:
        requests.post(JETSON_CONTROL_URL, json={
            "mode": "AUTO"
        })
    except:
        st.error("Failed to set AUTO mode")

# ================================
# Get data from Jetson
# ================================
try:
    response = requests.get(JETSON_URL, timeout=3)
    data = response.json()

    soil = data["soil_moisture"]
    water = data["water_level"]
    predicted = data["predicted_moisture"]
    action = data["action"]
    pump = data["pump_status"]
    reason = data["reason"]
    control_mode = data.get("control_mode", "AUTO")

except:
    st.error("❌ Cannot connect to Jetson")
    st.stop()

# ================================
# Display Metrics
# ================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Soil Moisture", f"{soil:.2f}%")
col2.metric("Water Level", f"{water:.2f}%")
col3.metric("Predicted Moisture", f"{predicted:.2f}%")
col4.metric("Pump", pump)

# ================================
# Decision Section
# ================================
st.subheader("🤖 Decision")

if action == "PUMP_ON":
    st.success("💧 Pump is ON")
else:
    st.info("🌿 Pump is OFF")

st.write(f"**Mode:** {control_mode}")
st.write(f"**Reason:** {reason}")

# ================================
# JSON Debug
# ================================
st.subheader("📡 Raw Data")
st.json(data)

# ================================
# Auto Refresh
# ================================
if auto_refresh:
    time.sleep(1)
    st.rerun()