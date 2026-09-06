import joblib
import pandas as pd
import numpy as np
import json
import shap
import matplotlib.pyplot as plt

def load_model():
    """Load model and related files"""
    model = joblib.load('models/calorie_xgb_model.pkl')
    scaler = joblib.load('models/calorie_scaler.pkl')
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

def create_background(n=100):
    np.random.seed(42)
    bg = pd.DataFrame({
        'Gender': np.random.choice([0, 1], n),
        'Age': np.random.randint(20, 80, n),
        'Height': np.random.normal(170, 10, n).clip(140, 210),
        'Weight': np.random.normal(72, 15, n).clip(40, 120),
        'Duration': np.random.uniform(1, 30, n),
        'Heart_Rate': np.random.normal(120, 20, n).clip(60, 180),
        'Body_Temp': np.random.normal(38.2, 0.8, n).clip(36, 42),
        'BMI': np.random.normal(24, 4, n).clip(15, 40)
    })
    return bg

def analyze_shap(model, scaler, feature_names, sample_data):
    """Analyze SHAP values for a sample"""
    background_df = create_background(100)
    background_scaled = scaler.transform(background_df[feature_names])
    explainer = shap.TreeExplainer(model, background_scaled, feature_perturbation='tree_path_dependent')
    
    sample_scaled = scaler.transform(sample_data[feature_names])
    shap_values = explainer.shap_values(sample_scaled)
    
    return shap_values, explainer

def plot_shap_summary(shap_values, feature_names):
    """Plot SHAP summary"""
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.show()

if __name__ == "__main__":
    print("Loading model...")
    model, scaler, feature_names = load_model()
    
    # Create sample data
    sample = pd.DataFrame({
        'Gender': [1],
        'Age': [30],
        'Height': [175],
        'Weight': [75],
        'Duration': [25],
        'Heart_Rate': [140],
        'Body_Temp': [38.5],
        'BMI': [24.5]
    })
    
    print("Analyzing SHAP...")
    shap_values, explainer = analyze_shap(model, scaler, feature_names, sample)
    
    print("SHAP values for sample:")
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {shap_values[0][i]:.2f}")
    
    print("\nSHAP analysis complete")
