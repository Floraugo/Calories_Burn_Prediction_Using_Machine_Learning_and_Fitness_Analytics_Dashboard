import joblib
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import shap
import uvicorn

# Load model and scaler
model = joblib.load('models/calorie_xgb_model.pkl')
scaler = joblib.load('models/calorie_scaler.pkl')
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Create SHAP explainer
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

background_df = create_background(100)
background_scaled = scaler.transform(background_df[feature_names])
explainer = shap.TreeExplainer(model, background_scaled, feature_perturbation='tree_path_dependent')

# FastAPI app
app = FastAPI(title="Calorie Burn Prediction API")

class WorkoutInput(BaseModel):
    Gender: int
    Age: float
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float
    BMI: float

class PredictionResponse(BaseModel):
    status: str
    predicted_calories: float
    shap_explanation: Dict[str, float]
    message: str
    llm_insight: Optional[str] = None

@app.get("/")
def root():
    return {"message": "Calorie Burn Prediction API"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict_with_insight", response_model=PredictionResponse)
def predict(workout: WorkoutInput):
    try:
        input_df = pd.DataFrame([workout.model_dump()])
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        pred = float(model.predict(input_scaled)[0])
        
        shap_values = explainer.shap_values(input_scaled)
        shap_dict = {feature_names[i]: float(shap_values[0][i]) for i in range(len(feature_names))}
        
        sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        top = sorted_shap[:2]
        msg = f"Predicted {pred:.1f} kcal. Key contributors: {top[0][0]} (+{top[0][1]:.1f})"
        if len(top) > 1:
            msg += f", {top[1][0]} (+{top[1][1]:.1f})"
        
        return PredictionResponse(
            status="success",
            predicted_calories=round(pred, 1),
            shap_explanation=shap_dict,
            message=msg,
            llm_insight="💡 Your duration and heart rate drove the burn. Try increasing duration by 5 minutes next time."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
