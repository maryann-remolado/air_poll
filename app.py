{"id":"42910","variant":"standard","title":"Fixed app.py"}
# app.py - CLEAN & FIXED VERSION
"""
Flask API for Metro Manila Air Pollution Risk Assessment
Safe, stable, and dependency-light version.
"""

from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

# ------------------------------------------------------------
# CORS (manual)
# ------------------------------------------------------------
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


# ------------------------------------------------------------
# Load Model (if available)
# ------------------------------------------------------------
model = None
scaler = None
label_encoder = None
feature_names = [
    "pm25", "pm10", "no2", "so2", "co",
    "o3", "temperature", "humidity"
]
model_accuracy = 0.85
MODEL_PATH = "air_pollution_model.pkl"

try:
    print("🔍 Checking for trained model...")

    if os.path.exists(MODEL_PATH):
        import joblib
        import numpy as np

        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]
        scaler = model_data["scaler"]
        label_encoder = model_data["label_encoder"]
        feature_names = model_data["features"]
        model_accuracy = model_data["accuracy"]

        print(f"✅ Model loaded. Accuracy: {model_accuracy:.2%}")
        print(f"📌 Features: {feature_names}")
    else:
        print(f"⚠️ Model file not found: {MODEL_PATH}")
        print("⚠️ Using fallback rule-based predictions.")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Reverting to rule-based fallback.")


# ------------------------------------------------------------
# Dashboard Mock Data
# ------------------------------------------------------------
dashboard_data = {
    "risk_distribution": {"Low": 42, "Moderate": 48, "High": 10},
    "summary": {
        "total_samples": 1000,
        "avg_pm25": 25.5,
        "model_accuracy": model_accuracy * 100,
    },
    "monthly_trends": [
        {"period": "2025-01", "pm25": 28},
        {"period": "2025-02", "pm25": 32},
        {"period": "2025-03", "pm25": 35},
        {"period": "2025-04", "pm25": 30},
        {"period": "2025-05", "pm25": 25},
        {"period": "2025-06", "pm25": 22},
        {"period": "2025-07", "pm25": 28},
        {"period": "2025-08", "pm25": 33},
        {"period": "2025-09", "pm25": 38},
        {"period": "2025-10", "pm25": 35},
        {"period": "2025-11", "pm25": 32},
    ],
}


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def calculate_aqi(pm25: float):
    """Simplified AQI logic based on PM2.5."""
    if pm25 <= 12:
        return pm25 * (50/12)
    elif pm25 <= 35.4:
        return 51 + (pm25 - 12.1) * (49/23.3)
    elif pm25 <= 55.4:
        return 101 + (pm25 - 35.5) * (49/19.9)
    elif pm25 <= 150.4:
        return 151 + (pm25 - 55.5) * (49/94.9)
    return 201 + (pm25 - 150.5) * (99/49.5)


def get_aqi_category(aqi: float):
    """Return AQI category string."""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    return "Very Unhealthy"


def get_recommendations(risk_level: str, aqi: float):
    """Recommendations based on model risk and AQI."""
    rec = {"general": [], "sensitive_groups": [], "actions": []}

    if risk_level == "Low" or aqi <= 50:
        rec["general"] = [
            "Air quality is satisfactory.",
            "Normal outdoor activities are safe.",
        ]
        rec["actions"] = ["Continue regular outdoor activities."]

    elif risk_level == "Moderate" or aqi <= 100:
        rec["general"] = [
            "Air quality is acceptable.",
            "Sensitive individuals should reduce prolonged outdoor exertion.",
        ]
        rec["sensitive_groups"] = [
            "Children, elderly, and people with respiratory conditions."
        ]
        rec["actions"] = ["Limit outdoor burning.", "Reduce vehicle idling."]

    else:  # High risk
        rec["general"] = [
            "Air quality is unhealthy.",
            "Everyone may experience health effects.",
        ]
        rec["sensitive_groups"] = ["Avoid outdoor activities."]
        rec["actions"] = [
            "Issue public health advisories.",
            "Implement traffic reduction policies.",
        ]

    return rec


# ------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json or {}
        params = {
            key: float(data.get(key, default))
            for key, default in {
                "pm25": 25, "pm10": 50, "no2": 30, "so2": 10,
                "co": 1.5, "o3": 40, "temperature": 28, "humidity": 65
            }.items()
        }

        location = data.get("location", "Metro Manila")

        # ----- ML MODEL PREDICTION -----
        if model and scaler and label_encoder:
            import numpy as np

            arr = np.array([[params[f] for f in feature_names]])
            scaled = scaler.transform(arr)

            encoded_pred = model.predict(scaled)[0]
            prediction = label_encoder.inverse_transform([encoded_pred])[0]

            prob = model.predict_proba(scaled)[0]
            prob_dict = {
                str(label_encoder.classes_[i]).lower(): float(prob[i] * 100)
                for i in range(len(prob))
            }

            confidence = float(max(prob) * 100)

        else:
            # ----- FALLBACK RULE-BASED -----
            pm25 = params["pm25"]

            if pm25 <= 12:
                prediction = "Low"
                prob_dict = {"low": 90, "moderate": 8, "high": 2}
                confidence = 95
            elif pm25 <= 35.4:
                prediction = "Moderate"
                prob_dict = {"low": 10, "moderate": 85, "high": 5}
                confidence = 90
            else:
                prediction = "High"
                prob_dict = {"low": 2, "moderate": 8, "high": 90}
                confidence = 85

        # AQI
        aqi = calculate_aqi(params["pm25"])

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": prob_dict,
            "aqi": round(aqi, 1),
            "aqi_category": get_aqi_category(aqi),
            "parameters": params,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "type": "Decision Tree" if model else "Rule-Based",
                "accuracy": round(model_accuracy * 100, 2),
                "features_used": feature_names,
            },
            "recommendations": get_recommendations(prediction, aqi),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ------------------------------------------------------------
# Dashboard, Model Info, Health Check
# ------------------------------------------------------------
@app.route("/api/dashboard", methods=["GET", "OPTIONS"])
def get_dashboard():
    if request.method == "OPTIONS":
        return "", 200
    return jsonify({"success": True, "dashboard": dashboard_data})


@app.route("/api/model", methods=["GET", "OPTIONS"])
def get_model_info():
    if request.method == "OPTIONS":
        return "", 200

    classes = (
        label_encoder.classes_.tolist()
        if label_encoder and hasattr(label_encoder, "classes_")
        else ["Low", "Moderate", "High"]
    )

    return jsonify({
        "success": True,
        "model": {
            "name": "Decision Tree Classifier" if model else "Rule-Based System",
            "accuracy": round(model_accuracy * 100, 2),
            "features": feature_names,
            "classes": classes,
            "description": "Decision Tree model for Metro Manila pollution risk assessment."
        }
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "accuracy": round(model_accuracy * 100, 2),
    })


@app.route("/api/test")
def test():
    return jsonify({"success": True, "message": "API running!"})


# ------------------------------------------------------------
# Root Documentation Page
# ------------------------------------------------------------
@app.route("/")
def index():
    status = "Loaded" if model else "Fallback Mode"
    color = "#d4edda" if model else "#f8d7da"

    return f"""
    <html><body style='font-family:Arial; padding:30px'>
        <h1>Metro Manila Air Pollution API</h1>

        <div style='padding:10px; background:{color}; border-radius:5px;'>
            <b>Model Status:</b> {status}
        </div>

        <h3>Available Endpoints:</h3>
        <ul>
            <li>/api/health</li>
            <li>/api/test</li>
            <li>/api/predict</li>
            <li>/api/dashboard</li>
            <li>/api/model</li>
        </ul>
    </body></html>
    """


# ------------------------------------------------------------
# Run Server
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Air Pollution API Server")
    print("📡 Listening on http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
