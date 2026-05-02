import serial
import time
from pathlib import Path

from src.preprocess import load_data, select_features, fit_scalers, transform_features, create_sequences
from src.agent import load_lstm_model, IrrigationAgent
from src.config import SEQ_LENGTH

PORT = "COM5"
BAUD = 115200
WATER_MIN_SECURITY = 15


def parse_esp32_line(line):
    """
    Format attendu:
    MOISTURE:45;WATER:80
    """
    if not line.startswith("MOISTURE:"):
        return None, None

    try:
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


def main():
    print("Connecting to ESP32...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    agent, sequence = init_agent()

    print("Agent IA prêt 🚀")

    while True:
        line = ser.readline().decode(errors="ignore").strip()

        moisture, water = parse_esp32_line(line)

        if moisture is None:
            continue

        print(f"Humidité sol: {moisture}% | Niveau eau: {water}%")

        if water < WATER_MIN_SECURITY:
            print("Niveau eau faible -> pompe OFF sécurité")
            ser.write(b"OFF\n")
            continue

        result = agent.run(
            sequence=sequence,
            current_moisture=moisture,
            rain_forecast=0.0
        )

        print("Decision:", result["action"])

        if result["action"] == "PUMP_ON":
            ser.write(b"ON\n")
        else:
            ser.write(b"OFF\n")


if __name__ == "__main__":
    main()