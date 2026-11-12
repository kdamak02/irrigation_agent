# Smart Irrigation Agent (Jetson Nano + ESP32)

Edge component of the [AgriSmart](https://github.com/kdamak02/agrismart-full-system) platform.
An **LSTM model** forecasts soil moisture from soil temperature and spectral bands, and an irrigation agent
decides when to switch the pump on, with safety rules. It runs on a **Jetson Nano** connected to an **ESP32**
that reads the sensors and drives the pump relay. A **Streamlit** dashboard shows live data and allows manual control.

```
Soil moisture + water level sensors ─► ESP32 ──serial──► Jetson Nano (Flask API + LSTM agent) ─► pump relay
                                                              ▲
                                               Streamlit dashboard (AUTO / MANUAL)
```

## How it works

1. The **ESP32** ([`firmware/esp32_irrigation.ino`](firmware/esp32_irrigation.ino)) reads soil moisture and water level,
   shows them on an I2C LCD and sends them over serial (`MOISTURE:..;WATER:..`).
2. The **Jetson** runs a Flask API ([`src/jetson_api.py`](src/jetson_api.py)) that parses the readings and builds
   a 24-step sequence of features.
3. The **LSTM regressor** ([`src/agent.py`](src/agent.py), 2 layers, 64 hidden units, model in `models/best_model_lstm.pt`)
   predicts the next soil-moisture value.
4. The **irrigation agent** decides `PUMP_ON` / `PUMP_OFF` with a reason, using thresholds from [`src/config.py`](src/config.py):
   minimum moisture (25%), safety cut-off (40%), rain forecast limit, and a minimum water level in the tank.
5. The **Streamlit dashboard** ([`app.py`](app.py)) polls `/status`, displays moisture, prediction and pump state,
   and sends manual commands to `/control`.

## Repository

| Path | Content |
| --- | --- |
| `app.py` | Streamlit dashboard (AUTO / MANUAL mode, live status) |
| `src/agent.py` | LSTM model and decision logic |
| `src/jetson_api.py` | Flask API on the Jetson: serial reading, prediction, pump control |
| `src/preprocess.py` | Feature selection, scaling, sequence creation |
| `src/simulate.py`, `src/hardware_bridge.py`, `src/hardware_live.py` | Simulation and hardware bridges |
| `data/extended_soil_moisture.csv` | Training data: soil moisture, soil temperature and spectral bands (454–950 nm) |
| `models/best_model_lstm.pt` | Trained LSTM checkpoint |
| `firmware/esp32_irrigation.ino` | ESP32 sensor and relay firmware |

## Run

```bash
pip install -r requirements.txt

# on the Jetson (ESP32 plugged on /dev/ttyUSB0)
python -m src.jetson_api

# dashboard (set the Jetson address in app.py)
streamlit run app.py
```

## Tech

Python · PyTorch (LSTM) · scikit-learn · Flask · Streamlit · PySerial · ESP32 (Arduino) · Jetson Nano
