import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

def dspsa(f, theta0,  max_iter,a=0.5, alpha=0.602,bounds=[[30,2,2,2],[100,20,20,20]]):
    theta = theta0.copy()
    best_theta = theta.copy()
    best_loss = f(theta)

    for k in range(1, max_iter + 1):
        ak = a / (k ** alpha)
        delta = 2 * np.random.randint(0, 2, size=len(theta)) - 1  # ±1

        thetaplus = theta + delta
        thetaminus = theta - delta

        # Clip within bounds
        if bounds is not None:
            thetaplus = np.clip(thetaplus, bounds[0], bounds[1])
            thetaminus = np.clip(thetaminus, bounds[0], bounds[1])

        y_plus = f(thetaplus)
        y_minus = f(thetaminus)

        ghat = (y_plus - y_minus) / (2.0 * delta)
        theta = np.round(theta - ak * ghat).astype(int)

        if bounds is not None:
            theta = np.clip(theta, bounds[0], bounds[1])

        loss = f(theta)
        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()

    return best_theta, best_loss

def rf_loss_given_split(params, X_train, y_train, X_test, y_test):
    # Ensure parameters are within valid bounds
    n_estimators       = max(10, int(params[0]))  # ≥ 10
    max_depth          = max(1, int(params[1]))   # ≥ 1
    min_samples_split  = max(2, int(params[2]))   # ≥ 2
    min_samples_leaf   = max(1, int(params[3]))   # ≥ 1

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)

    return loss

if __name__ == "__main__":
    df = pd.read_csv('data/salesdata.csv')
    df.fillna(-1, inplace=True)
    # Drop IDs
    df = df.drop(['User_ID', 'Product_ID'], axis=1)
    df = pd.get_dummies(df, columns=['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years'])
    X = df.drop('Purchase', axis=1)
    y = df['Purchase']
    hyper_df=pd.DataFrame(df.sample(n=2000))
    X_hyper=hyper_df.drop(columns=['Purchase'])
    y_hyper=hyper_df['Purchase']
    X_hyper_train,X_hyper_test,y_hyper_train,y_hyper_test=train_test_split(X_hyper,y_hyper,test_size=0.2,random_state=42)

    print("Starting SPSA tuning...")
    objective = lambda theta: rf_loss_given_split(
        theta, X_hyper_train, y_hyper_train, X_hyper_test, y_hyper_test
    )

    best_params, best_loss = dspsa(
        objective,
        theta0=np.array([50, 5, 5, 5]),  # Initial guess
        max_iter=50                     # Adjust iterations as you like
    )

    print(f"Best Hyperparameters Found: {best_params}")
    print(f"Best MSE: {best_loss:.2f}")

    with open("tuned_params.json", "w") as f:
        json.dump(best_params.tolist(), f)