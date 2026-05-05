from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import joblib

app = FastAPI(title="Job Market Prediction API")

# Load MLflow model
model = mlflow.pyfunc.load_model("models:/job-market-model/Production")

#  Load feature order (CRITICAL)
features = joblib.load("artifacts/features.pkl")


class InputData(BaseModel):
    Region: int
    employed: float
    labour_rate: float
    Area: int
    lag_1: float
    lag_2: float
    rolling_avg: float
    Year: int
    Month: int


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: InputData):
    print(" Received input:", data)

    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    #  Ensure correct feature order
    df = df[features]

    prediction = model.predict(df)

    print(" Prediction:", prediction)

    return {"prediction": float(prediction[0])}