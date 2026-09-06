# ============================================================================
# CALORIE BURN PREDICTION - FULL PIPELINE
# ============================================================================
# This script trains an XGBoost model, generates predictions, and saves files
# ============================================================================

import kagglehub
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

def load_data():
    """Load and preprocess the dataset"""
    print("Loading data...")
    path = kagglehub.dataset_download("ruchikakumbhar/calories-burnt-prediction")
    df = pd.read_csv(os.path.join(path, 'calories.csv'))
    df = df.drop('User_ID', axis=1)
    df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def train_model(df):
    """Train XGBoost model"""
    print("Training XGBoost model...")
    X = df.drop('Calories', axis=1)
    y = df['Calories']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        enable_categorical=False
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: MAE={mae:.2f} kcal, R²={r2:.4f}")
    
    return model, scaler, feature_names

def save_files(model, scaler, feature_names):
    """Save model files"""
    joblib.dump(model, 'models/calorie_xgb_model.pkl', compress=3)
    joblib.dump(scaler, 'models/calorie_scaler.pkl', compress=3)
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print("Files saved successfully")

def generate_predictions(df, model, scaler, feature_names):
    """Generate predictions"""
    print("Generating predictions...")
    X = df[feature_names]
    X_scaled = scaler.transform(X)
    df['predicted_calories'] = model.predict(X_scaled)
    df['error'] = df['predicted_calories'] - df['Calories']
    df['abs_error'] = np.abs(df['error'])
    df.to_csv('data/calories_with_predictions_full.csv', index=False)
    print("Predictions saved to CSV")

def main():
    print("=" * 60)
    print("CALORIE BURN PREDICTION PIPELINE")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    df = load_data()
    model, scaler, feature_names = train_model(df)
    save_files(model, scaler, feature_names)
    generate_predictions(df, model, scaler, feature_names)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
