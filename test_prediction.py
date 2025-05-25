import joblib
import pandas as pd
import json

# Load the trained model
model = joblib.load("rf_model.pkl")

# Load the feature column names
with open("model_columns.json", "r") as f:
    columns = json.load(f)

# Create a baseline input with all zeros
sample = pd.DataFrame([[0] * len(columns)], columns=columns)

# Set a default value for categorical columns so the model doesn't break
sample["Occupation"] = 1
sample["Marital_Status"] = 0
sample["Product_Category_2"] = 2
sample["Product_Category_3"] = 3
sample["Gender_F"] = 1
sample["Age_26-35"] = 1
sample["City_Category_B"] = 1
sample["Stay_In_Current_City_Years_1"] = 1

# Try different Product_Category_1 values
for pc1 in [1, 5, 10, 15, 20]:
    sample["Product_Category_1"] = pc1
    prediction = model.predict(sample)[0]
    print(f"Prediction for Product_Category_1 = {pc1}: {prediction:.2f}")
