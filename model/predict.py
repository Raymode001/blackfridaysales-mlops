import json
import pandas as pd
from joblib import load

# Load model and column list
model = load("rf_model.pkl")
with open("model_columns.json") as f:
    COLUMNS = json.load(f)

def predict_one(sample_dict: dict):
    """
    sample_dict: raw feature dictionary, with keys matching any of the one-hot columns
    missing keys will be filled with 0
    """
    # Build a single-row DataFrame
    df = pd.DataFrame([sample_dict])

    # Add any missing columns
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Re-order columns
    df = df[COLUMNS]

    # Predict
    return float(model.predict(df)[0])
