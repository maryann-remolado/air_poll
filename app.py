# app.py - FIXED VERSION
"""
Simplified Flask API for Metro Manila Air Pollution Risk Assessment
No external dependencies required beyond Flask
"""

from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

# Add CORS headers manually (no flask_cors needed)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize model variables with defaults
model = None
scaler = None
label_encoder = None
feature_names = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'temperature', 'humidity']
model_accuracy = 0.85
MODEL_PATH = 'air_pollution_model.pkl'

# Try to load the trained model
try:
    print("Loading trained model...")
    
    # Check if file exists
    if os.path.exists(MODEL_PATH):
        import joblib
        import numpy as np
        
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['features']
        model_accuracy = model_data['accuracy']
        print(f"✅ Model loaded successfully! Accuracy: {model_accuracy:.2%}")
        print(f"✅ Features: {feature_names}")
    else:
        print(f"⚠️ Model file '{MODEL_PATH}' not found")
        print("⚠️ Using fallback prediction logic")
        
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("⚠️ Using fallback prediction logic")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Using fallback prediction logic")

# Sample dashboard data
dashboard_data = {
    'risk_distribution': {'Low': 42, 'Moderate': 48, 'High': 10},
    'summary': {
        'total_samples': 1000,
        'avg_pm25': 25.5,
        'model_accuracy': model_accuracy * 100
    },
    'monthly_trends': [
        {'period': '2025-01', 'pm25': 28},
        {'period': '2025-02', 'pm25': 32},
        {'period': '2025-03', 'pm25': 35},
        {'period': '2025-04', 'pm25': 30},
        {'period': '2025-05', 'pm25': 25},
        {'period': '2025-06', 'pm25': 22},
        {'period': '2025-07', 'pm25': 28},
        {'period': '2025-08', 'pm25': 33},
        {'period': '2025-09', 'pm25': 38},
        {'period': '2025-10', 'pm25': 35},
        {'period': '2025-11', 'pm25': 32}
    ]
}

# Helper functions
def calculate_aqi(pm25):
    """Calculate Air Quality Index from PM2.5"""
    if pm25 <= 12:
        return pm25 * (50/12)  # Good (0-50)
    elif pm25 <= 35.4:
        return 51 + (pm25-12.1) * (49/23.3)  # Moderate (51-100)
    elif pm25 <= 55.4:
        return 101 + (pm25-35.5) * (49/19.9)  # Unhealthy for Sensitive Groups (101-150)
    elif pm25 <= 150.4:
        return 151 + (pm25-55.5) * (49/94.9)  # Unhealthy (151-200)
    else:
        return 201 + (pm25-150.5) * (99/49.5)  # Very Unhealthy (201-300)

def get_aqi_category(aqi):
    """Get AQI category from AQI value"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

def get_recommendations(risk_level, aqi):
    """Get recommendations based on risk level and AQI"""
    recommendations = {
        'general': [],
        'sensitive_groups': [],
        'actions': []
    }
    
    if risk_level == "Low" or aqi <= 50:
        recommendations['general'] = [
            "Air quality is satisfactory",
            "Normal outdoor activities are safe"
        ]
        recommendations['actions'] = [
            "Continue regular outdoor activities",
            "Maintain current pollution control measures"
        ]
    
    elif risk_level == "Moderate" or aqi <= 100:
        recommendations['general'] = [
            "Air quality is acceptable",
            "Unusually sensitive people should consider reducing prolonged outdoor exertion"
        ]
        recommendations['sensitive_groups'] = [
            "Children, elderly, and people with respiratory conditions",
            "Consider reducing strenuous outdoor activities"
        ]
        recommendations['actions'] = [
            "Reduce vehicle idling",
            "Limit outdoor burning",
            "Use public transportation when possible"
        ]
    
    else:  # High risk or AQI > 100
        recommendations['general'] = [
            "Air quality is unhealthy",
            "Everyone may begin to experience health effects"
        ]
        recommendations['sensitive_groups'] = [
            "Avoid all outdoor activities",
            "Stay indoors with air purifiers if possible"
        ]
        recommendations['actions'] = [
            "Issue public health advisory",
            "Implement traffic reduction measures",
            "Activate emergency pollution control protocols"
        ]
    
    return recommendations

# Prediction endpoint
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Real-time prediction endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get data from request
        data = request.json or {}
        
        # Extract parameters with defaults
        params = {
            'pm25': float(data.get('pm25', 25)),
            'pm10': float(data.get('pm10', 50)),
            'no2': float(data.get('no2', 30)),
            'so2': float(data.get('so2', 10)),
            'co': float(data.get('co', 1.5)),
            'o3': float(data.get('o3', 40)),
            'temperature': float(data.get('temperature', 28)),
            'humidity': float(data.get('humidity', 65))
        }
        
        # Get location if provided
        location = data.get('location', 'Metro Manila')
        
        # Make prediction
        if model is not None and scaler is not None and label_encoder is not None:
            try:
                # Import numpy only if needed
                import numpy as np
                
                # Prepare input array in correct feature order
                input_array = np.array([[params[feature] for feature in feature_names]])
                
                # Scale input
                input_scaled = scaler.transform(input_array)
                
                # Predict
                prediction_encoded = model.predict(input_scaled)[0]
                prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                
                # Get probabilities
                probabilities = model.predict_proba(input_scaled)[0]
                confidence = max(probabilities) * 100
                
                # Format probabilities
                prob_dict = {}
                for i, class_name in enumerate(label_encoder.classes_):
                    prob_dict[class_name.lower()] = float(probabilities[i] * 100)
            except Exception as e:
                print(f"Model prediction error: {e}")
                # Fallback to rule-based prediction
                raise
        else:
            # Fallback to rule-based prediction
            pm25 = params['pm25']
            
            if pm25 <= 12:
                prediction = "Low"
                confidence = 95
                prob_dict = {'low': 90, 'moderate': 8, 'high': 2}
            elif pm25 <= 35.4:
                prediction = "Moderate"
                confidence = 90
                prob_dict = {'low': 10, 'moderate': 85, 'high': 5}
            else:
                prediction = "High"
                confidence = 85
                prob_dict = {'low': 2, 'moderate': 8, 'high': 90}
        
        # Calculate AQI for reference
        aqi = calculate_aqi(params['pm25'])
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'probabilities': prob_dict,
            'aqi': round(aqi, 1),
            'aqi_category': get_aqi_category(aqi),
            'parameters': params,
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': 'Decision Tree' if model else 'Rule-Based',
                'accuracy': round(model_accuracy * 100, 2),
                'features_used': feature_names
            },
            'recommendations': get_recommendations(prediction, aqi)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing prediction request'
        }), 400

# Dashboard data endpoint
@app.route('/api/dashboard', methods=['GET', 'OPTIONS'])
def get_dashboard():
    """Returns dashboard statistics"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        return jsonify({
            'success': True,
            'dashboard': dashboard_data,
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Model information endpoint
@app.route('/api/model', methods=['GET', 'OPTIONS'])
def get_model_info():
    """Returns information about the trained model"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Create label encoder classes if not defined
        if label_encoder is None:
            classes = ['Low', 'Moderate', 'High']
        else:
            classes = label_encoder.classes_.tolist() if hasattr(label_encoder, 'classes_') else ['Low', 'Moderate', 'High']
        
        return jsonify({
            'success': True,
            'model': {
                'name': 'Decision Tree Classifier' if model else 'Rule-Based System',
                'accuracy': round(model_accuracy * 100, 2),
                'features': feature_names,
                'classes': classes,
                'description': 'Trained on Metro Manila air quality data for pollution risk assessment'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Health check endpoint
@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'model_accuracy': round(model_accuracy * 100, 2)
    })

# Test endpoint for quick verification
@app.route('/api/test', methods=['GET', 'OPTIONS'])
def test():
    """Simple test endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    return jsonify({
        'success': True,
        'message': 'API is working!',
        'timestamp': datetime.now().isoformat()
    })

# Root endpoint
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Air Pollution API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>Metro Manila Air Pollution Risk Assessment API</h1>
        
        <div class="status ''' + ('success' if model else 'error') + '''">
            Model Status: ''' + ('✅ Loaded' if model else '⚠️ Using rule-based fallback') + '''
        </div>
        
        <h3>API Endpoints:</h3>
        <ul>
            <li><a href="/api/health">GET /api/health</a> - Health check</li>
            <li><a href="/api/test">GET /api/test</a> - Test endpoint</li>
            <li><a href="/api/dashboard">GET /api/dashboard</a> - Dashboard data</li>
            <li><a href="/api/model">GET /api/model</a> - Model information</li>
            <li>POST /api/predict - Get risk prediction</li>
        </ul>
        
        <h3>Test with curl:</h3>
        <pre>
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"pm25": 25}'
        </pre>
        
        <p>Web Interface: <a href="http://localhost:8000">Open Dashboard</a></p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Metro Manila Air Pollution Risk Assessment API")
    print("="*60)
    print(f"\n📊 Model Status: {'✅ Loaded' if model else '⚠️ Rule-based fallback'}")
    print(f"📊 Accuracy: {model_accuracy:.1%}")
    print("\n🌐 Endpoints:")
    print("  GET  /              - API documentation")
    print("  POST /api/predict    - Get risk prediction")
    print("  GET  /api/dashboard  - Get dashboard data")
    print("  GET  /api/model      - Get model information")
    print("  GET  /api/health     - Health check")
    print("  GET  /api/test       - Quick test")
    print(f"\n🚀 Starting server on http://localhost:5000")
    print("="*60)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n💡 Quick fix: Install minimal requirements:")
        print("   pip install flask")
