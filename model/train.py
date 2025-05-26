
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

with open("tuned_params.json", "r") as f:
    tuned_list = json.load(f)

best_params = {
    "n_estimators": int(tuned_list[0]),
    "max_depth": int(tuned_list[1]),
    "min_samples_split": int(tuned_list[2]),
    "min_samples_leaf": int(tuned_list[3])
}

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df.fillna(-1, inplace=True)

    # Drop IDs
    df.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])

    X = df.drop('Purchase', axis=1)
    y = df['Purchase']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(data_path="data/salesdata.csv", model_path="rf_model.pkl", columns_path="model_columns.json"):
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved to {model_path}")

    # Save column names
    with open(columns_path, "w") as f:
        json.dump(list(X_train.columns), f)
    print(f"📄 Feature columns saved to {columns_path}")

    # === Debug: Prediction distribution ===
    y_pred = model.predict(X_test)
    print(f"R² score on test set: {model.score(X_test, y_test):.4f}")
    print(f"Mean prediction: {np.mean(y_pred):.2f}, Std Dev: {np.std(y_pred):.2f}")

if __name__ == "__main__":
    train_model()
