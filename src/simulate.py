from pathlib import Path

from .preprocess import load_data, select_features, fit_scalers, transform_features, create_sequences
from .agent import load_lstm_model, IrrigationAgent
from .config import SEQ_LENGTH, TARGET_COL


def print_result(title, result):
    print(f"\n=== {title} ===")
    print(f"Current moisture   : {result['current_moisture']:.2f}")
    print(f"Predicted moisture : {result['predicted_moisture']:.2f}")
    print(f"Rain forecast      : {result['rain_forecast']:.2f} mm")
    print(f"Decision           : {result['action']}")
    print(f"Duration           : {result['duration_sec']} sec")
    print(f"Reason             : {result['reason']}")

    command = {
        "action": result["action"],
        "duration_sec": result["duration_sec"],
        "predicted_moisture": round(result["predicted_moisture"], 2),
        "reason": result["reason"]
    }

    print("\nJSON command:")
    print(command)


def main():
    project_root = Path(__file__).resolve().parent.parent

    csv_path = project_root / "data" / "extended_soil_moisture.csv"
    model_path = project_root / "models" / "best_model_lstm.pt"

    print("Loading dataset...")
    df = load_data(str(csv_path))

    print("Selecting features...")
    feature_cols = select_features(df)
    print(f"Number of selected features: {len(feature_cols)}")

    n = len(df)
    train_end = int(n * 0.70)

    train_df = df.iloc[:train_end].copy()
    full_df = df.copy()

    print("Fitting scalers...")
    x_scaler, y_scaler = fit_scalers(train_df, feature_cols)

    print("Transforming features...")
    X_scaled, y_scaled = transform_features(full_df, feature_cols, x_scaler, y_scaler)

    print("Creating sequences...")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
    print(f"Sequence shape: {X_seq.shape}")

    print("Loading trained LSTM model...")
    model, checkpoint = load_lstm_model(str(model_path), device="cpu")

    expected_input_size = checkpoint["input_size"]
    actual_input_size = X_seq.shape[2]

    print(f"Model expects input_size = {expected_input_size}")
    print(f"Current data has input_size = {actual_input_size}")

    if expected_input_size != actual_input_size:
        raise ValueError(
            f"Input size mismatch: model expects {expected_input_size}, but current data has {actual_input_size}"
        )

    agent = IrrigationAgent(model=model, y_scaler=y_scaler, device="cpu")

    last_sequence = X_seq[-1]
    current_moisture = df[TARGET_COL].iloc[-1]

    result_real = agent.run(
        sequence=last_sequence,
        current_moisture=current_moisture,
        rain_forecast=0.0
    )

    result_demo_dry = agent.run(
        sequence=last_sequence,
        current_moisture=22.0,
        rain_forecast=0.0
    )

    result_demo_rain = agent.run(
        sequence=last_sequence,
        current_moisture=22.0,
        rain_forecast=5.0
    )

    print_result("REAL SAMPLE RESULT", result_real)
    print_result("DEMO DRY SCENARIO", result_demo_dry)
    print_result("DEMO RAIN SCENARIO", result_demo_rain)


if __name__ == "__main__":
    main()