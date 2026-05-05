import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def run_training():
    print(" Training started...")

    df = pd.read_csv("data/processed/clean_data.csv")

    X = df.drop(columns=["unemployment"])
    y = df["unemployment"]

    #  Save feature list (IMPORTANT)
    feature_names = X.columns.tolist()
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(feature_names, "artifacts/features.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("job_market_experiment")

    with mlflow.start_run():

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        # MLflow logs
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        joblib.dump(model, "artifacts/model.pkl")

        print(f" Training complete | MSE: {mse}")