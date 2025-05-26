# blackfridaysales-mlops
# BlackFriday Sales Purchase Prediction with Random Forest and SPSA Hyperparameter Tuning

### Overview
This project predicts customer purchase amounts on Black Friday using a Random Forest regression model trained on a real-world sales dataset. It includes data preprocessing, feature engineering with one-hot encoding, model training, and a custom hyperparameter tuning method using Discrete Simultaneous Perturbation Stochastic Approximation (DSPSA). The trained model is deployed as a FastAPI application that serves purchase amount predictions via REST API.

### Features
- Data cleaning and categorical feature one-hot encoding
- Random Forest regression model with adjustable hyperparameters
- Custom DSPSA tuner for efficient hyperparameter optimization
- API for real-time predictions using FastAPI
- Model and feature columns saved for consistent inference
- Comprehensive logging of feature importance and prediction distribution

### Getting Started

#### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

#### Dataset
Place your CSV file in the `data/` folder as `salesdata.csv`.

### Usage
#### Step 1: Hyperparameter Tuning with SPSA (Optional)
  ```bash
  python tune.py
  ```
This generates `tuned_params.json` based on SPSA optimization.

#### Step 2: Train the Model
  ```bash
  python train.py
  ```
This reads `tuned_params.json` and saves the trained model to rf_model.pkl.

#### Step 3: Run the API
  ```bash
  uvicorn main:app --reload
  ```
Then go to http://127.0.0.1:8000/docs to use the interactive Swagger UI.

#### Example JSON Input for /predict
  ```json
  {
  "Gender_M": 1,
  "Age_18-25": 1,
  "City_Category_B": 1,
  "Stay_In_Current_City_Years_2": 1,
  "Product_Category_1": 5,
  ...
  }
  ```

### 📁 Project Structure
- `train.py`: Trains the model using tuned params

- `tune.py`: Runs DSPSA tuner to find best RF hyperparameters

- `main.py`: FastAPI app for serving predictions

- `rf_model.pkl`: Trained model

- `model_columns.json`: Column names used during training

- `tuned_params.json`: Best hyperparameters

- `requirements.txt`: Python dependencies

### 📈 Results
- R² Score ~0.65 on hold-out test set

- Tuning improves performance over defaults

- Top features printed in training logs

### 🙌 Author
Raymode001 — https://github.com/Raymode001



