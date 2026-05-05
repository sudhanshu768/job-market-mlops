import pandas as pd
import joblib
import mlflow
from sklearn.metrics import mean_squared_error
import os

#  Latest Evidently imports
from evidently import Report
from evidently.presets import DataDriftPreset


def run_evaluation():
    print(" Evaluating model...")

    #  Load processed data
    df = pd.read_csv("data/processed/clean_data.csv")

    #  Load trained model
    model = joblib.load("artifacts/model.pkl")

    #  Prepare features (match training)
    X = df.drop(columns=["unemployment", "Date"], errors="ignore")
    y = df["unemployment"]

    #  Predictions
    preds = model.predict(X)

    #  Metrics
    mse = mean_squared_error(y, preds)
    rmse = mse ** 0.5

    print(f" MSE: {mse}")
    print(f" RMSE: {rmse}")

    #  Log to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)

    # =========================================================
    #  Evidently Data Drift Report (LATEST API)
    # =========================================================

    report = Report(metrics=[DataDriftPreset()])

    # Split dataset (for demo drift)
    reference = df.sample(frac=0.5, random_state=42)
    current = df.sample(frac=0.5, random_state=1)

    #  IMPORTANT: run() returns new object
    drift_report = report.run(
        reference_data=reference,
        current_data=current
    )

    #  Save report
    os.makedirs("reports", exist_ok=True)

    report_path = os.path.abspath("reports/data_drift_report.html")

    #  Correct method (latest API)
    drift_report.save_html(report_path)

    print(f" Drift report saved at: {report_path}")


if __name__ == "__main__":
    run_evaluation()