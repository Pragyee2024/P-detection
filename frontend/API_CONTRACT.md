# API Contract

## Request
POST /predict
Content-Type: multipart/form-data
Field: "audio" (the recorded or uploaded audio file)

## Success response (200)
```json
{
  "risk_score": 0.72,
  "label": "higher_risk",
  "model_name": "xgboost",
  "top_features": [
    {"name": "PPE", "value": 0.31, "shap": 0.18},
    {"name": "spread1", "value": -4.1, "shap": 0.12}
  ],
  "warnings": []
}
```

- risk_score: a number from 0 to 1
- label: only "higher_risk" or "lower_risk"
- top_features: the 5 or 6 features that mattered most, with shap positive if it pushes the score toward higher risk and negative if toward lower risk
- warnings: a list of text messages, empty if nothing is wrong


## Error response (400 or 500)
```json
{ 
    "error": "Audio too short. Please record at least 5 seconds." 
}
```

- value: the feature's raw measured value; shap: its contribution to the score