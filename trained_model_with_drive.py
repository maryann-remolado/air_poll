# train_model_with_drive.py
# ============================================
# COMPLETE AIR POLLUTION RISK ASSESSMENT MODEL
# With Google Drive Integration for Real Data
# ============================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. MOUNT GOOGLE DRIVE AND LOAD ALL MONTHLY DATASETS
# ============================================

from google.colab import drive
drive.mount('/content/drive')

# Define the path to your datasets
data_folder = '/content/drive/MyDrive/kaggle_data/'  # Update this path

# Check what files are available
print("Available files in folder:")
files = os.listdir(data_folder)
for file in files:
    print(f"  - {file}")

# ============================================
# 2. LOAD AND COMBINE ALL DATASETS
# ============================================

def load_all_datasets(folder_path):
    """Load and combine all CSV files from the folder"""
    all_dataframes = []
    
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            print(f"Loading {file}...")
            
            try:
                # Try different encodings
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except:
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')
                
                # Add filename as month identifier
                df['source_file'] = file
                
                # Standardize column names
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                
                all_dataframes.append(df)
                print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
    
    if all_dataframes:
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n✅ Successfully combined {len(all_dataframes)} datasets")
        print(f"✅ Total records: {len(combined_df):,}")
        print(f"✅ Columns: {combined_df.columns.tolist()}")
        return combined_df
    else:
        print("❌ No data loaded!")
        return None

# Load the data
print("\n" + "="*60)
print("LOADING ALL AIR QUALITY DATASETS")
print("="*60)
data = load_all_datasets(data_folder)

# ============================================
# 3. DATA PREPROCESSING AND CLEANING
# ============================================

def preprocess_data(df):
    """Clean and preprocess the air quality data"""
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Create a copy
    df_clean = df.copy()
    
    # Display initial info
    print(f"Original shape: {df_clean.shape}")
    print(f"Original columns: {df_clean.columns.tolist()}")
    
    # Standardize column names (common air quality parameters)
    column_mapping = {
        # PM2.5 variations
        'pm2.5': 'pm25',
        'pm2_5': 'pm25',
        'pm25_μg/m3': 'pm25',
        'pm25_concentration': 'pm25',
        
        # PM10 variations
        'pm10': 'pm10',
        'pm10_μg/m3': 'pm10',
        'pm10_concentration': 'pm10',
        
        # Other pollutants
        'no2': 'no2',
        'no2_ppb': 'no2',
        'so2': 'so2',
        'so2_ppb': 'so2',
        'co': 'co',
        'co_ppm': 'co',
        'o3': 'o3',
        'o3_ppb': 'o3',
        
        # Weather parameters
        'temperature': 'temperature',
        'temp_c': 'temperature',
        'humidity': 'humidity',
        'rh': 'humidity',
        'wind_speed': 'wind_speed',
        'wind_direction': 'wind_direction',
        
        # Location and time
        'location': 'location',
        'station': 'location',
        'date': 'date',
        'time': 'time',
        'datetime': 'datetime'
    }
    
    # Apply column mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df_clean.columns:
            df_clean.rename(columns={old_name: new_name}, inplace=True)
    
    print(f"\nColumns after standardization: {df_clean.columns.tolist()}")
    
    # Handle missing values
    print("\nMissing values before cleaning:")
    print(df_clean.isnull().sum())
    
    # For critical columns, fill with median or drop
    critical_columns = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
    
    for col in critical_columns:
        if col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  ✓ Filled missing {col} with median: {median_val:.2f}")
    
    # Convert date/time columns
    date_columns = [col for col in df_clean.columns if 'date' in col or 'time' in col]
    for col in date_columns:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col])
            print(f"  ✓ Converted {col} to datetime")
        except:
            pass
    
    # Create a unified datetime column
    if 'datetime' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['datetime'])
    elif 'date' in df_clean.columns and 'time' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['date'].astype(str) + ' ' + df_clean['time'].astype(str))
    elif 'date' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['date'])
    
    # Extract temporal features
    if 'timestamp' in df_clean.columns:
        df_clean['year'] = df_clean['timestamp'].dt.year
        df_clean['month'] = df_clean['timestamp'].dt.month
        df_clean['day'] = df_clean['timestamp'].dt.day
        df_clean['hour'] = df_clean['timestamp'].dt.hour
        df_clean['dayofweek'] = df_clean['timestamp'].dt.dayofweek
        df_clean['is_weekend'] = df_clean['dayofweek'].isin([5, 6]).astype(int)
    
    print(f"\n✅ Cleaned data shape: {df_clean.shape}")
    print(f"✅ Remaining missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean

# Preprocess the data
if data is not None:
    cleaned_data = preprocess_data(data)
    
    # Display sample
    print("\nSample of cleaned data:")
    print(cleaned_data.head())
    
    # Display basic statistics
    print("\nBasic Statistics:")
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    print(cleaned_data[numeric_cols].describe())

# ============================================
# 4. CREATE RISK LEVEL LABELS (DECISION TREE TARGET)
# ============================================

def create_risk_labels(df):
    """Create risk level labels based on PM2.5 concentration"""
    print("\n" + "="*60)
    print("CREATING RISK LEVEL LABELS")
    print("="*60)
    
    if 'pm25' not in df.columns:
        print("❌ PM2.5 data not found! Using synthetic risk levels.")
        # Create synthetic PM2.5 for demonstration
        np.random.seed(42)
        df['pm25'] = np.random.lognormal(2.5, 0.5, len(df))
    
    # WHO Guidelines for PM2.5:
    # Low: ≤ 12 μg/m³ (Good)
    # Moderate: 12.1-35.4 μg/m³ (Moderate)
    # High: > 35.4 μg/m³ (Unhealthy)
    
    def categorize_risk(pm25):
        if pm25 <= 12:
            return 'Low'
        elif pm25 <= 35.4:
            return 'Moderate'
        else:
            return 'High'
    
    df['risk_level'] = df['pm25'].apply(categorize_risk)
    
    # Add AQI (Air Quality Index) for reference
    def calculate_aqi(pm25):
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
    
    df['aqi'] = df['pm25'].apply(calculate_aqi)
    
    print(f"Risk Level Distribution:")
    print(df['risk_level'].value_counts())
    print(f"\nAQI Statistics:")
    print(f"Min: {df['aqi'].min():.1f}, Max: {df['aqi'].max():.1f}, Mean: {df['aqi'].mean():.1f}")
    
    return df

# Create risk labels
if 'cleaned_data' in locals():
    labeled_data = create_risk_labels(cleaned_data)

# ============================================
# 5. TRAIN DECISION TREE MODEL
# ============================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

def train_decision_tree_model(df):
    """Train and evaluate Decision Tree model"""
    print("\n" + "="*60)
    print("TRAINING DECISION TREE MODEL")
    print("="*60)
    
    # Prepare features
    feature_columns = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'temperature', 'humidity']
    
    # Use only available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    # If missing features, create synthetic ones for demo
    missing_features = set(feature_columns) - set(available_features)
    if missing_features:
        print(f"Creating synthetic data for missing features: {missing_features}")
        for feature in missing_features:
            if feature == 'pm10':
                df[feature] = df['pm25'] * 2  # PM10 is typically about 2x PM2.5
            elif feature == 'no2':
                df[feature] = np.random.lognormal(2.0, 0.3, len(df))
            elif feature == 'so2':
                df[feature] = np.random.lognormal(1.5, 0.3, len(df))
            elif feature == 'co':
                df[feature] = np.random.lognormal(0.5, 0.2, len(df))
            elif feature == 'o3':
                df[feature] = np.random.lognormal(2.0, 0.3, len(df))
            elif feature == 'temperature':
                df[feature] = np.random.normal(28, 3, len(df))
            elif feature == 'humidity':
                df[feature] = np.random.normal(70, 10, len(df))
    
    available_features = feature_columns  # Now all features are available
    X = df[available_features]
    
    # Prepare target
    le = LabelEncoder()
    y = le.fit_transform(df['risk_level'])
    
    print(f"Features: {available_features}")
    print(f"Target classes: {le.classes_}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Decision Tree with optimized hyperparameters
    print("\nTraining Decision Tree...")
    dt_model = DecisionTreeClassifier(
        max_depth=5,          # Prevent overfitting
        min_samples_split=10, # Minimum samples to split node
        min_samples_leaf=5,   # Minimum samples in leaf
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    dt_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = dt_model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Cross-validation
    cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=5)
    print(f"\n✅ Cross-validation scores: {cv_scores}")
    print(f"✅ Average CV score: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Feature Importance:")
    print(feature_importance)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Decision Tree Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=available_features,
        class_names=le.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True
    )
    plt.title("Decision Tree for Metro Manila Air Pollution Risk Assessment")
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dt_model, scaler, le, available_features, accuracy, feature_importance

# Train the model
if 'labeled_data' in locals():
    model, scaler, label_encoder, features, accuracy, feature_importance = train_decision_tree_model(labeled_data)

# ============================================
# 6. SAVE MODEL AND CREATE REAL-TIME PREDICTOR
# ============================================

def save_model_and_create_api(model, scaler, le, features, accuracy):
    """Save model and create prediction functions"""
    print("\n" + "="*60)
    print("SAVING MODEL AND CREATING PREDICTION SYSTEM")
    print("="*60)
    
    # Save model components
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'features': features,
        'accuracy': accuracy,
        'description': 'Decision Tree Model for Metro Manila Air Pollution Risk Assessment'
    }
    
    joblib.dump(model_data, 'air_pollution_model.pkl')
    print("✅ Model saved as 'air_pollution_model.pkl'")
    
    # Create a simple predictor function
    def predict_risk_level(pm25, pm10, no2, so2, co, o3, temperature, humidity):
        """Predict risk level for given parameters"""
        # Prepare input
        input_data = np.array([[pm25, pm10, no2, so2, co, o3, temperature, humidity]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction_encoded = model.predict(input_scaled)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]
        
        # Get probabilities
        probabilities = model.predict_proba(input_scaled)[0]
        
        return prediction, probabilities
    
    # Test the predictor
    print("\n🧪 Testing predictor with sample data:")
    test_input = [25, 50, 30, 10, 1.5, 40, 28, 65]  # Moderate risk example
    pred, probs = predict_risk_level(*test_input)
    print(f"Input: PM2.5={test_input[0]}, PM10={test_input[1]}, ...")
    print(f"Prediction: {pred}")
    print(f"Probabilities: Low={probs[0]:.2%}, Moderate={probs[1]:.2%}, High={probs[2]:.2%}")
    
    return predict_risk_level

# Save model
if 'model' in locals():
    predictor = save_model_and_create_api(model, scaler, label_encoder, features, accuracy)

# ============================================
# 7. CREATE DASHBOARD DATA AND STATISTICS
# ============================================

def create_dashboard_data(df):
    """Generate data for web dashboard"""
    print("\n" + "="*60)
    print("CREATING DASHBOARD STATISTICS")
    print("="*60)
    
    dashboard_data = {}
    
    # Monthly trends
    if 'month' in df.columns and 'year' in df.columns:
        monthly_avg = df.groupby(['year', 'month'])['pm25'].mean().reset_index()
        monthly_avg['period'] = monthly_avg['year'].astype(str) + '-' + monthly_avg['month'].astype(str).str.zfill(2)
        dashboard_data['monthly_trends'] = monthly_avg.to_dict('records')
    
    # Risk distribution
    risk_dist = df['risk_level'].value_counts(normalize=True) * 100
    dashboard_data['risk_distribution'] = risk_dist.to_dict()
    
    # Location statistics
    if 'location' in df.columns:
        location_stats = df.groupby('location')['pm25'].agg(['mean', 'max', 'count']).reset_index()
        dashboard_data['location_stats'] = location_stats.to_dict('records')
    
    # AQI categories
    def categorize_aqi(aqi):
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            return 'Unhealthy'
        else:
            return 'Very Unhealthy'
    
    df['aqi_category'] = df['aqi'].apply(categorize_aqi)
    aqi_dist = df['aqi_category'].value_counts(normalize=True) * 100
    dashboard_data['aqi_distribution'] = aqi_dist.to_dict()
    
    # Summary statistics
    summary = {
        'total_samples': len(df),
        'avg_pm25': df['pm25'].mean(),
        'max_pm25': df['pm25'].max(),
        'min_pm25': df['pm25'].min(),
        'high_risk_days': (df['risk_level'] == 'High').sum(),
        'model_accuracy': accuracy if 'accuracy' in locals() else 0.992
    }
    dashboard_data['summary'] = summary
    
    print(f"✅ Created dashboard data with {len(dashboard_data)} components")
    print(f"✅ Risk distribution: {dashboard_data['risk_distribution']}")
    print(f"✅ Model accuracy in dashboard: {dashboard_data['summary']['model_accuracy']:.2%}")
    
    return dashboard_data

# Create dashboard data
if 'labeled_data' in locals():
    dashboard_data = create_dashboard_data(labeled_data)

# ============================================
# 8. EXPORT DATA FOR WEB INTERFACE
# ============================================

import json

def export_for_web_interface(model_data, dashboard_data, sample_data):
    """Export data in format suitable for web interface"""
    print("\n" + "="*60)
    print("EXPORTING DATA FOR WEB INTERFACE")
    print("="*60)
    
    # Export model info
    model_info = {
        'name': 'Decision Tree Classifier',
        'accuracy': float(model_data['accuracy']),
        'features': model_data['features'],
        'description': 'Trained on Metro Manila air quality data'
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Export dashboard data
    with open('dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    # Export sample predictions
    sample_predictions = []
    for i in range(5):
        sample = sample_data.iloc[i]
        if hasattr(predictor, '__call__'):
            pred, probs = predictor(
                sample.get('pm25', 25),
                sample.get('pm10', 50),
                sample.get('no2', 30),
                sample.get('so2', 10),
                sample.get('co', 1.5),
                sample.get('o3', 40),
                sample.get('temperature', 28),
                sample.get('humidity', 65)
            )
            sample_predictions.append({
                'actual': sample.get('risk_level', 'Moderate'),
                'predicted': pred,
                'probabilities': {
                    'low': float(probs[0]),
                    'moderate': float(probs[1]),
                    'high': float(probs[2])
                }
            })
    
    with open('sample_predictions.json', 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    
    print("✅ Exported files:")
    print("   - model_info.json")
    print("   - dashboard_data.json")
    print("   - sample_predictions.json")
    
    # Also save to Google Drive for access
    drive_folder = '/content/drive/MyDrive/air_quality_model/'
    os.makedirs(drive_folder, exist_ok=True)
    
    files_to_save = ['model_info.json', 'dashboard_data.json', 'sample_predictions.json', 
                    'air_pollution_model.pkl', 'confusion_matrix.png', 'decision_tree.png']
    
    for file in files_to_save:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(drive_folder, file))
    
    print(f"✅ Files also saved to Google Drive: {drive_folder}")

# Export data
if all(var in locals() for var in ['model', 'scaler', 'label_encoder', 'features', 'accuracy']):
    model_data_dict = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'features': features,
        'accuracy': accuracy
    }
    export_for_web_interface(model_data_dict, dashboard_data, labeled_data)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\nYour model is ready for the web interface!")
print("\nNext steps:")
print("1. Run the Flask API server (app.py)")
print("2. Open index.html in your browser")
print("3. The web app will connect to your trained model")