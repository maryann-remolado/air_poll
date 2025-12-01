# train_model_with_drive.py
# ============================================
# COMPLETE AIR POLLUTION RISK ASSESSMENT MODEL
# With optional Google Drive Integration for Real Data
# ============================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Optional: gdown (may not be installed in offline setups)
try:
    import gdown
except Exception:
    gdown = None

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# --------------------------------------------
# Utilities
# --------------------------------------------
def normalize_col(col: str) -> str:
    """Normalize column names to a predictable form."""
    if not isinstance(col, str):
        return col
    s = col.lower().strip()
    s = s.replace('µ', 'u').replace('μ', 'u')
    s = s.replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '')
    s = s.replace('.', '_')
    # common replacements
    s = s.replace('pm2_5', 'pm25').replace('pm2_5', 'pm25').replace('pm25_μg_m3', 'pm25')
    s = s.replace('pm25_μg/m3', 'pm25')
    s = s.replace('pm_25', 'pm25')
    s = s.replace('pm10_μg_m3', 'pm10')
    return s

# ============================================
# 1. DATA LOADER (gdown optional, local CSV fallback, else sample)
# ============================================
DRIVE_FILE_IDS = {
    # 'air_quality_2023': '1gnbhvPkIEnvBJWwWbWHMjzg1c_fDI4Ia',  # example
}

def load_datasets_from_drive_or_local():
    """Try Drive (if gdown available) then local CSVs then sample data."""
    datasets = []

    # Attempt Google Drive download if gdown available and DRIVE_FILE_IDS provided
    if gdown and DRIVE_FILE_IDS:
        print("📥 Attempting to download datasets via gdown...")
        for name, file_id in DRIVE_FILE_IDS.items():
            try:
                url = f'https://drive.google.com/uc?id={file_id}'
                output = f'{name}.csv'
                gdown.download(url, output, quiet=False)
                df = pd.read_csv(output)
                datasets.append(df)
                print(f"   ✓ Downloaded {output} ({len(df)} rows)")
            except Exception as e:
                print(f"   ✗ Failed to download {name}: {e}")

    # If no datasets from drive, check local CSVs in working directory
    if not datasets:
        local_csvs = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
        if local_csvs:
            print("📁 Found local CSV files:")
            for f in local_csvs:
                try:
                    df = pd.read_csv(f)
                    datasets.append(df)
                    print(f"   ✓ Loaded {f} ({len(df)} rows)")
                except Exception as e:
                    print(f"   ✗ Failed to load {f}: {e}")

    # Combine or fallback to sample
    if datasets:
        combined = pd.concat(datasets, ignore_index=True)
        print(f"\n✅ Loaded {len(datasets)} dataset(s) — total rows: {len(combined)}")
        return combined

    print("⚠️ No datasets loaded (Drive/local). Generating sample data...")
    return create_sample_data()

def create_sample_data(n_samples=1000):
    """Create sample Metro Manila air quality data with standardized column names."""
    np.random.seed(42)
    data = {
        'pm25': np.random.lognormal(2.5, 0.5, n_samples),
        'pm10': np.random.lognormal(3.0, 0.4, n_samples),
        'no2': np.random.lognormal(2.0, 0.3, n_samples),
        'so2': np.random.lognormal(1.5, 0.3, n_samples),
        'co': np.random.lognormal(0.5, 0.2, n_samples),
        'o3': np.random.lognormal(2.0, 0.3, n_samples),
        'temperature': np.random.normal(28, 3, n_samples),
        'humidity': np.random.normal(70, 10, n_samples),
    }
    df = pd.DataFrame(data)
    df['risk_level'] = df['pm25'].apply(lambda x: 'Low' if x <= 12 else 'Moderate' if x <= 35.4 else 'High')
    return df

# Load
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
data = load_datasets_from_drive_or_local()

# ============================================
# 2. DATA PREPROCESSING
# ============================================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the air quality data."""
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)

    if df is None or df.empty:
        raise ValueError("No dataframe provided to preprocess.")

    df_clean = df.copy()

    # Normalize column names
    new_cols = {col: normalize_col(col) for col in df_clean.columns}
    df_clean.rename(columns=new_cols, inplace=True)

    print(f"Original shape: {df_clean.shape}")
    print(f"Columns after normalization: {df_clean.columns.tolist()}")

    # Critical pollutant columns expected
    critical_columns = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
    for col in critical_columns:
        if col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  ✓ Filled missing {col} with median: {median_val:.3f}")

    # Temperature & humidity
    for col in ['temperature','humidity']:
        if col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Date parsing: convert any date/time-like columns to datetime
    date_cols = [c for c in df_clean.columns if 'date' in c or 'time' in c or 'timestamp' in c or 'datetime' in c]
    for col in date_cols:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col])
            print(f"  ✓ Converted {col} to datetime")
        except Exception:
            pass

    # Create unified timestamp if possible
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
        df_clean['is_weekend'] = df_clean['dayofweek'].isin([5,6]).astype(int)

    print(f"\n✅ Cleaned data shape: {df_clean.shape}")
    print(f"✅ Remaining missing values (total): {df_clean.isnull().sum().sum()}")
    return df_clean

cleaned_data = preprocess_data(data)

print("\nSample of cleaned data:")
print(cleaned_data.head())

# ============================================
# 3. CREATE RISK LEVEL LABELS & AQI
# ============================================
def create_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create risk level labels and compute AQI from PM2.5."""
    print("\n" + "="*60)
    print("CREATING RISK LEVEL LABELS")
    print("="*60)

    df = df.copy()

    if 'pm25' not in df.columns:
        print("❌ PM2.5 not found — creating synthetic pm25 column.")
        df['pm25'] = np.random.lognormal(2.5, 0.5, len(df))

    def categorize_risk(pm25):
        if pm25 <= 12:
            return 'Low'
        elif pm25 <= 35.4:
            return 'Moderate'
        else:
            return 'High'

    df['risk_level'] = df['pm25'].apply(categorize_risk)

    def calculate_aqi(pm25):
        if pm25 <= 12:
            return pm25 * (50/12)
        elif pm25 <= 35.4:
            return 51 + (pm25-12.1) * (49/23.3)
        elif pm25 <= 55.4:
            return 101 + (pm25-35.5) * (49/19.9)
        elif pm25 <= 150.4:
            return 151 + (pm25-55.5) * (49/94.9)
        else:
            return 201 + (pm25-150.5) * (99/49.5)

    df['aqi'] = df['pm25'].apply(calculate_aqi)

    print("Risk Level Distribution:")
    print(df['risk_level'].value_counts())
    print(f"AQI min/max/mean: {df['aqi'].min():.1f} / {df['aqi'].max():.1f} / {df['aqi'].mean():.1f}")
    return df

labeled_data = create_risk_labels(cleaned_data)

# ============================================
# 4. TRAIN DECISION TREE MODEL
# ============================================
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_decision_tree_model(df: pd.DataFrame):
    print("\n" + "="*60)
    print("TRAINING DECISION TREE MODEL")
    print("="*60)

    feature_columns = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'temperature', 'humidity']

    # Ensure features exist (create synthetic if missing)
    for feat in feature_columns:
        if feat not in df.columns:
            if feat == 'pm10' and 'pm25' in df.columns:
                df[feat] = df['pm25'] * 2
            elif feat in ['no2','so2','co','o3']:
                df[feat] = np.random.lognormal(1.5, 0.4, len(df))
            elif feat == 'temperature':
                df[feat] = np.random.normal(28,3,len(df))
            elif feat == 'humidity':
                df[feat] = np.random.normal(70,10,len(df))

    X = df[feature_columns].astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df['risk_level'])

    print(f"Features: {feature_columns}")
    print(f"Target classes: {le.classes_}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5,
                                      random_state=42, class_weight='balanced')
    dt_model.fit(X_train_scaled, y_train)

    y_pred = dt_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation
    try:
        cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=5)
        print(f"Cross-val scores: {cv_scores}, mean = {cv_scores.mean():.4f}")
    except Exception as e:
        print("Cross-val failed:", e)
        cv_scores = None

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # Confusion matrix plot (save)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Decision Tree')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Plot tree: ensure class_names is list (not ndarray)
    plt.figure(figsize=(18,8))
    plot_tree(dt_model,
              feature_names=feature_columns,
              class_names=le.classes_.tolist(),
              filled=True, rounded=True, fontsize=8, proportion=True)
    plt.title("Decision Tree - Air Pollution Risk")
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=200, bbox_inches='tight')
    plt.close()

    return dt_model, scaler, le, feature_columns, accuracy, feature_importance

model = scaler = label_encoder = features = accuracy = feature_importance = None
if 'labeled_data' in locals():
    model, scaler, label_encoder, features, accuracy, feature_importance = train_decision_tree_model(labeled_data)

# ============================================
# 5. SAVE MODEL AND CREATE PREDICTOR
# ============================================
def save_model_and_create_api(model, scaler, le, features, accuracy, feature_importance=None):
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)

    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,          # NOTE: key name matches app.py expectation
        'features': features,
        'accuracy': accuracy,
        'feature_importance': feature_importance
    }

    joblib.dump(model_data, 'air_pollution_model.pkl')
    print("✅ Model saved as 'air_pollution_model.pkl'")

    def predictor(pm25, pm10, no2, so2, co, o3, temperature, humidity):
        arr = np.array([[pm25, pm10, no2, so2, co, o3, temperature, humidity]])
        arr_s = scaler.transform(arr)
        pred_enc = model.predict(arr_s)[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        probs = model.predict_proba(arr_s)[0]
        return pred_label, probs

    # Test predictor
    print("\n🧪 Predictor test:")
    test_input = [25, 50, 30, 10, 1.5, 40, 28, 65]
    try:
        p, ps = predictor(*test_input)
        print(f"  Example Input: {test_input} -> Prediction: {p}, probs: {ps}")
    except Exception as e:
        print("  Predictor test failed:", e)

    return predictor

predictor = None
if model is not None:
    predictor = save_model_and_create_api(model, scaler, label_encoder, features, accuracy, feature_importance)

# ============================================
# 6. CREATE DASHBOARD DATA
# ============================================
def create_dashboard_data(df):
    dashboard = {}
    if 'year' in df.columns and 'month' in df.columns:
        monthly_avg = df.groupby(['year','month'])['pm25'].mean().reset_index()
        monthly_avg['period'] = monthly_avg['year'].astype(str) + '-' + monthly_avg['month'].astype(str).str.zfill(2)
        dashboard['monthly_trends'] = monthly_avg.to_dict('records')
    dashboard['risk_distribution'] = (df['risk_level'].value_counts(normalize=True) * 100).to_dict()
    if 'location' in df.columns:
        dashboard['location_stats'] = df.groupby('location')['pm25'].agg(['mean','max','count']).reset_index().to_dict('records')
    df['aqi_category'] = df['aqi'].apply(lambda a: 'Good' if a<=50 else 'Moderate' if a<=100 else 'Unhealthy for Sensitive Groups' if a<=150 else 'Unhealthy' if a<=200 else 'Very Unhealthy')
    dashboard['aqi_distribution'] = (df['aqi_category'].value_counts(normalize=True) * 100).to_dict()
    dashboard['summary'] = {
        'total_samples': len(df),
        'avg_pm25': float(df['pm25'].mean()),
        'max_pm25': float(df['pm25'].max()),
        'min_pm25': float(df['pm25'].min()),
        'high_risk_days': int((df['risk_level']=='High').sum()),
        'model_accuracy': float(accuracy) if accuracy is not None else 0.0
    }
    return dashboard

dashboard_data = create_dashboard_data(labeled_data)

# ============================================
# 7. EXPORT FOR WEB INTERFACE (model_info, dashboard, sample_preds)
# ============================================
import json

def export_for_web_interface(model_data, dashboard_data, sample_data, predictor):
    print("\n" + "="*60)
    print("EXPORTING FOR WEB INTERFACE")
    print("="*60)

    model_info = {
        'name': 'Decision Tree Classifier',
        'accuracy': float(model_data['accuracy']),
        'features': model_data['features'],
        'description': 'Trained on Metro Manila air quality data'
    }
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    with open('dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    sample_predictions = []
    # Guard: ensure there are at least 5 rows
    n_samples = min(5, len(sample_data))
    for i in range(n_samples):
        sample = sample_data.iloc[i]
        if callable(predictor):
            pred, probs = predictor(
                float(sample.get('pm25', 25)),
                float(sample.get('pm10', 50)),
                float(sample.get('no2', 30)),
                float(sample.get('so2', 10)),
                float(sample.get('co', 1.5)),
                float(sample.get('o3', 40)),
                float(sample.get('temperature', 28)),
                float(sample.get('humidity', 65))
            )
            sample_predictions.append({
                'input': {
                    'pm25': float(sample.get('pm25', np.nan)),
                    'pm10': float(sample.get('pm10', np.nan)),
                    'no2': float(sample.get('no2', np.nan)),
                    'so2': float(sample.get('so2', np.nan)),
                    'co': float(sample.get('co', np.nan)),
                    'o3': float(sample.get('o3', np.nan)),
                    'temperature': float(sample.get('temperature', np.nan)),
                    'humidity': float(sample.get('humidity', np.nan)),
                },
                'actual': sample.get('risk_level', None),
                'predicted': pred,
                'probabilities': { 'class_'+str(i): float(p) for i,p in enumerate(probs) }
            })

    with open('sample_predictions.json', 'w') as f:
        json.dump(sample_predictions, f, indent=2)

    # Optionally copy to Drive folder if available in the environment (Colab path)
    drive_folder = '/content/drive/MyDrive/air_quality_model/'
    try:
        os.makedirs(drive_folder, exist_ok=True)
        files_to_save = ['model_info.json', 'dashboard_data.json', 'sample_predictions.json',
                         'air_pollution_model.pkl', 'confusion_matrix.png', 'decision_tree.png']
        for file in files_to_save:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(drive_folder, file))
        print(f"✅ Exported files copied to Google Drive folder (if accessible): {drive_folder}")
    except Exception:
        # not fatal if drive not available
        pass

    print("✅ Export complete: model_info.json, dashboard_data.json, sample_predictions.json")

if model is not None:
    model_data_dict = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'features': features,
        'accuracy': accuracy
    }
    export_for_web_interface(model_data_dict, dashboard_data, labeled_data, predictor)

print("\n" + "="*60)
print("✅ TRAINING & EXPORT COMPLETE!")
print("="*60)
print("Model file: ./air_pollution_model.pkl")
