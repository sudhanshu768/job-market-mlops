import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error


def run_evaluation():
    print(" Evaluating model...")

    df = pd.read_csv("data/processed/clean_data.csv")

    # Load model + feature list
    model = joblib.load("artifacts/model.pkl")
    features = joblib.load("artifacts/features.pkl")

    X = df.drop(columns=["unemployment"])
    y = df["unemployment"]

    #  CRITICAL FIX (match training features)
    X = X[features]

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)

    print(f" Evaluation complete | MSE: {mse}")