from flask import Flask, render_template, request, jsonify
from datetime import datetime
from pathlib import Path
import threading
import time
import joblib
import numpy as np
import serial
import serial.tools.list_ports

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)

# Load the same artifacts produced by the uploaded training program.
model = joblib.load(MODEL_DIR / "best_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

latest_data = {
    "ph": None,
    "turbidity": None,
    "temperature": None,
    "prediction": None,
    "confidence": None,
    "timestamp": None,
    "source": "Waiting for ESP32 on COM3..."
}

history = []
MAX_HISTORY = 60
lock = threading.Lock()


def predict_water_quality(ph, turbidity, temperature):
    """Predict using the saved scaler, best model and label encoder."""
    values = np.array([[float(ph), float(turbidity), float(temperature)]], dtype=float)
    scaled = scaler.transform(values)

    encoded_prediction = model.predict(scaled)[0]
    class_name = str(label_encoder.inverse_transform([encoded_prediction])[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled)[0]
        confidence = float(np.max(probabilities) * 100.0)

    return class_name, confidence


def save_reading(ph, turbidity, temperature, source="ESP32 WiFi"):
    prediction, confidence = predict_water_quality(
        ph, turbidity, temperature
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reading = {
        "ph": round(float(ph), 2),
        "turbidity": round(float(turbidity), 2),
        "temperature": round(float(temperature), 2),
        "prediction": prediction,
        "confidence": round(confidence, 2) if confidence is not None else None,
        "timestamp": now,
        "source": source
    }

    with lock:
        latest_data.update(reading)
        history.append(reading.copy())
        if len(history) > MAX_HISTORY:
            del history[:-MAX_HISTORY]

    return reading



# ---------- USB Serial configuration ----------
SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200
SERIAL_RETRY_SECONDS = 3

def serial_reader():
    """
    Reads ESP32 lines in this format:
    DATA,<ph>,<turbidity_adc>,<temperature>
    Example: DATA,6.82,2180,28.50
    """
    while True:
        try:
            print(f"[SERIAL] Opening {SERIAL_PORT} at {SERIAL_BAUD} baud...")
            with serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2) as ser:
                time.sleep(2)  # allow ESP32 serial connection to settle
                print(f"[SERIAL] Connected to {SERIAL_PORT}")
                with lock:
                    latest_data["source"] = f"USB Serial ({SERIAL_PORT})"

                while True:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw:
                        continue
                    print(f"[SERIAL] {raw}")

                    parts = [part.strip() for part in raw.split(",")]
                    if len(parts) != 4 or parts[0].upper() != "DATA":
                        continue

                    try:
                        ph = float(parts[1])
                        turbidity = float(parts[2])
                        temperature = float(parts[3])
                        if not all(np.isfinite(v) for v in (ph, turbidity, temperature)):
                            continue
                        reading = save_reading(
                            ph, turbidity, temperature,
                            source=f"USB Serial ({SERIAL_PORT})"
                        )
                        print(
                            "[SERIAL] Saved reading: "
                            f"pH={reading['ph']}, turbidity={reading['turbidity']}, "
                            f"temperature={reading['temperature']}, "
                            f"prediction={reading['prediction']}"
                        )
                    except (TypeError, ValueError) as exc:
                        print(f"[SERIAL] Invalid DATA line ({exc}): {raw}")

        except serial.SerialException as exc:
            with lock:
                latest_data["source"] = f"Serial reconnecting on {SERIAL_PORT}"
            print(f"[SERIAL] Cannot open/read {SERIAL_PORT}: {exc}")
            print(f"[SERIAL] Check the port and close Arduino Serial Monitor. Retrying in {SERIAL_RETRY_SECONDS}s...")
            time.sleep(SERIAL_RETRY_SECONDS)
        except Exception as exc:
            print(f"[SERIAL] Unexpected error: {exc}")
            time.sleep(SERIAL_RETRY_SECONDS)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/data", methods=["GET"])
def get_data():
    with lock:
        return jsonify({
            "latest": latest_data.copy(),
            "history": history.copy(),
            "model": {
                "name": "best_model.pkl",
                "classes": label_encoder.classes_.tolist()
            }
        })


@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor_data():
    """
    USB serial reader accepts DATA CSV lines (JSON endpoint remains available for testing):
    {
      "ph": 6.5,
      "turbidity": 3200,
      "temperature": 27.4
    }
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"success": False, "error": "JSON body is required"}), 400

    try:
        ph = float(data.get("ph"))
        turbidity = float(
            data.get("turbidity", data.get("turb", data.get("Trub")))
        )
        temperature = float(data.get("temperature", data.get("temp", data.get("Temperature"))))

        if not all(np.isfinite(v) for v in [ph, turbidity, temperature]):
            raise ValueError("Sensor values must be finite numbers")

        reading = save_reading(ph, turbidity, temperature)
        return jsonify({"success": True, "reading": reading}), 200

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "Invalid sensor values. Expected numeric ph, turbidity and temperature."
        }), 400


@app.route("/api/predict", methods=["POST"])
def predict_only():
    """Manual prediction endpoint for testing."""
    data = request.get_json(silent=True) or {}

    try:
        ph = float(data["ph"])
        turbidity = float(data["turbidity"])
        temperature = float(data["temperature"])
        prediction, confidence = predict_water_quality(
            ph, turbidity, temperature
        )

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2) if confidence is not None else None
        })
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "Send ph, turbidity and temperature as numeric values."
        }), 400


@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "time": datetime.now().isoformat(timespec="seconds")
    })


if __name__ == "__main__":
    # Start one background thread that reads sensor data over USB Serial.
    reader_thread = threading.Thread(target=serial_reader, daemon=True)
    reader_thread.start()

    # Local laptop dashboard; no Wi-Fi required for sensor collection.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
