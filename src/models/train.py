import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/clean_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts/model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "artifacts/features.pkl")


def run_training():
    print(" Training started...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["unemployment", "Date"], errors='ignore')
    y = df["unemployment"]

    # Save feature order
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # MLflow logging
    mlflow.set_experiment("job_market_experiment")

    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")

    print(" Training complete!")


if __name__ == "__main__":
    run_training()