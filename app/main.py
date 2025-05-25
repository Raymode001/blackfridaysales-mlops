from fastapi import FastAPI
from pydantic import BaseModel
from model.predict import predict_one

app = FastAPI(title="BlackFriday RF Predictor")

class PurchaseInput(BaseModel):
    Occupation: int = 0
    Marital_Status: int = 0
    Product_Category_1: int = 0
    Product_Category_2: int = 0
    Product_Category_3: int = 0
    Gender_F: int = 0
    Gender_M: int = 0
    Age_0_17: int = 0
    Age_18_25: int = 0
    Age_26_35: int = 0
    Age_36_45: int = 0
    Age_46_50: int = 0
    Age_51_55: int = 0
    Age_55_plus: int = 0
    City_Category_A: int = 0
    City_Category_B: int = 0
    City_Category_C: int = 0
    Stay_In_Current_City_Years_0: int = 0
    Stay_In_Current_City_Years_1: int = 0
    Stay_In_Current_City_Years_2: int = 0
    Stay_In_Current_City_Years_3: int = 0
    Stay_In_Current_City_Years_4_plus: int = 0

@app.post("/predict")
def predict_endpoint(data: PurchaseInput):
    pred = predict_one(data.dict())
    return {"predicted_purchase": pred}
