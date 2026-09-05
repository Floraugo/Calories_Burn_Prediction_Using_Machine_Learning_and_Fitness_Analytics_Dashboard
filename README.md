
# Calorie Burn Prediction System

## Overview
An end-to-end AI system for predicting calorie expenditure during physical exercise. The system combines machine learning (XGBoost), SHAP explainability, large language models, and Power BI dashboarding.

## Model Performance
| Metric | Value |
| :--- | :--- |
| MAE | 1.08 kcal |
| RMSE | 1.42 kcal |
| R² | 0.9994 |

## Files in this Repository
| File | Description |
| :--- | :--- |
| `calorie_xgb_model.pkl` | Trained XGBoost model |
| `calorie_scaler.pkl` | Feature scaler |
| `feature_names.json` | List of feature names |
| `calories_with_predictions_full.csv` | Predictions for all 15,000 rows |
| `calories_test_predictions.csv` | Test set predictions (~3,000 rows) |
| `calories_test_shap_sample.csv` | 500 rows with SHAP explanations |

## How to Use
1. Open in Google Colab
2. Run the code (no Google Drive required)
3. Download CSV files for Power BI

## Architecture
- **ML Model**: XGBoost
- **Explainability**: SHAP
- **LLM**: Gemini/OpenRouter
- **API**: FastAPI with ngrok
- **Dashboard**: Power BI

## Author
[Your Name] - Birkbeck, University of London
