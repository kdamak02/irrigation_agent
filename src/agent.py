import torch
import torch.nn as nn

from .config import MIN_THRESHOLD, SAFETY_THRESHOLD, USE_RAIN, RAIN_LIMIT

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_lstm_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)

    model = LSTMRegressor(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


class IrrigationAgent:
    def __init__(self, model, y_scaler, device="cpu"):
        self.model = model
        self.y_scaler = y_scaler
        self.device = device
        self.model.eval()

    def predict_moisture(self, sequence):
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_scaled = self.model(x).cpu().numpy()
            pred_real = self.y_scaler.inverse_transform(pred_scaled)[0][0]
        return float(pred_real)

    def decide(self, current_moisture, predicted_moisture, rain_forecast=0.0):
        if current_moisture > SAFETY_THRESHOLD:
            return {
                "action": "PUMP_OFF",
                "duration_sec": 0,
                "reason": "Soil already sufficiently wet"
            }

        if USE_RAIN and rain_forecast >= RAIN_LIMIT:
            return {
                "action": "PUMP_OFF",
                "duration_sec": 0,
                "reason": "Rain expected"
            }

        # Cas 1 : le sol est déjà sec maintenant
        if current_moisture < MIN_THRESHOLD:
            moisture_gap = MIN_THRESHOLD - current_moisture

            if moisture_gap < 2:
                duration_sec = 60
            elif moisture_gap < 5:
                duration_sec = 120
            else:
                duration_sec = 180

            return {
                "action": "PUMP_ON",
                "duration_sec": duration_sec,
                "reason": "Current soil moisture below threshold"
            }

        # Cas 2 : le modèle prévoit que le sol va devenir sec
        if predicted_moisture < MIN_THRESHOLD:
            moisture_gap = MIN_THRESHOLD - predicted_moisture

            if moisture_gap < 2:
                duration_sec = 60
            elif moisture_gap < 5:
                duration_sec = 120
            else:
                duration_sec = 180

            return {
                "action": "PUMP_ON",
                "duration_sec": duration_sec,
                "reason": "Predicted soil moisture below threshold"
            }

        return {
            "action": "PUMP_OFF",
            "duration_sec": 0,
            "reason": "No irrigation needed"
        }

    def run(self, sequence, current_moisture, rain_forecast=0.0):
        predicted_moisture = self.predict_moisture(sequence)
        decision = self.decide(current_moisture, predicted_moisture, rain_forecast)

        return {
            "current_moisture": float(current_moisture),
            "predicted_moisture": float(predicted_moisture),
            "rain_forecast": float(rain_forecast),
            "action": decision["action"],
            "duration_sec": decision["duration_sec"],
            "reason": decision["reason"]
        }