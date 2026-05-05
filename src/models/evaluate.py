import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_PATH = os.path.join(BASE_DIR, "data/processed/clean_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts/model.pkl")


def run_evaluation():
    print(" Evaluating model...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["unemployment", "Date"], errors='ignore')
    y = df["unemployment"]

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    print(" Evaluation complete!")


if __name__ == "__main__":
    run_evaluation()