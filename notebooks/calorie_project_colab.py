# 1. Install dependencies
!pip install kagglehub xgboost scikit-learn joblib fastapi uvicorn pyngrok shap nest-asyncio google-generativeai openai -q

# 2. Run pipeline
!python src/calorie_pipeline.py

# 3. Download files
from google.colab import files
files.download('data/calories_with_predictions_full.csv')
files.download('data/calories_test_predictions.csv')

print("Done!")
