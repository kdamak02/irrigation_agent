from flask import Flask, jsonify, request
import serial
import time
import threading
from pathlib import Path

from src.preprocess import load_data, select_features, fit_scalers, transform_features, create_sequences
from src.agent import load_lstm_model, IrrigationAgent
from src.config import SEQ_LENGTH

PORT = "/dev/ttyUSB0"
BAUD = 115200
WATER_MIN_SECURITY = 15

app = Flask(__name__)

control_mode = "AUTO"        # AUTO or MANUAL
manual_action = "PUMP_OFF"   # PUMP_ON or PUMP_OFF

latest_data = {
    "soil_moisture": None,
    "water_level": None,
    "predicted_moisture": None,
    "action": "PUMP_OFF",
    "reason": "Waiting for data",
    "pump_status": "OFF",
    "control_mode": "AUTO"
}


def parse_esp32_line(line):
    try:
        if not line.startswith("MOISTURE:"):
            return None, None

        parts = line.split(";")
        moisture = float(parts[0].split(":")[1])
        water = float(parts[1].split(":")[1])

        return moisture, water
    except Exception:
        return None, None


def init_agent():
    project_root = Path(__file__).resolve().parent.parent

    csv_path = project_root / "data" / "extended_soil_moisture.csv"
    model_path = project_root / "models" / "best_model_lstm.pt"

    df = load_data(str(csv_path))
    feature_cols = select_features(df)

    train_end = int(len(df) * 0.7)
    train_df = df.iloc[:train_end]

    x_scaler, y_scaler = fit_scalers(train_df, feature_cols)
    X_scaled, y_scaled = transform_features(df, feature_cols, x_scaler, y_scaler)
    X_seq, _ = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

    model, _ = load_lstm_model(str(model_path))
    agent = IrrigationAgent(model, y_scaler)

    return agent, X_seq[-1]


def hardware_loop():
    global latest_data, control_mode, manual_action

    print("Loading IA agent...")
    agent, sequence = init_agent()

    print("Connecting ESP32...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    print("Jetson API + IA ready")

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        moisture, water = parse_esp32_line(line)

        if moisture is None:
            continue

        result = agent.run(
            sequence=sequence,
            current_moisture=moisture,
            rain_forecast=0.0
        )

        ai_action = result["action"]
        reason = result["reason"]

        if control_mode == "AUTO":
            action = ai_action
        else:
            action = manual_action
            reason = "Manual mode command"

        if water < WATER_MIN_SECURITY:
            action = "PUMP_OFF"
            reason = "Water level too low"

        if action == "PUMP_ON":
            ser.write(b"ON\n")
            pump_status = "ON"
        else:
            ser.write(b"OFF\n")
            pump_status = "OFF"

        latest_data = {
            "soil_moisture": moisture,
            "water_level": water,
            "predicted_moisture": round(result["predicted_moisture"], 2),
            "action": action,
            "reason": reason,
            "pump_status": pump_status,
            "control_mode": control_mode,
            "ai_action": ai_action,
            "manual_action": manual_action
        }

        print(latest_data)


@app.route("/status")
def status():
    return jsonify(latest_data)


@app.route("/control", methods=["POST"])
def control():
    global control_mode, manual_action

    data = request.get_json()

    mode = data.get("mode")
    action = data.get("action")

    if mode in ["AUTO", "MANUAL"]:
        control_mode = mode

    if action in ["PUMP_ON", "PUMP_OFF"]:
        manual_action = action

    return jsonify({
        "status": "ok",
        "control_mode": control_mode,
        "manual_action": manual_action
    })


@app.route("/")
def home():
    return jsonify({
        "message": "Smart Irrigation Jetson API is running",
        "status_url": "/status",
        "control_url": "/control"
    })


if __name__ == "__main__":
    thread = threading.Thread(target=hardware_loop)
    thread.daemon = True
    thread.start()

    app.run(host="0.0.0.0", port=5000)