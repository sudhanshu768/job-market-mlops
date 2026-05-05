from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Job Market Prediction API")

#  Load model from local file (Docker-safe)
model = joblib.load("artifacts/model.pkl")

#  Load feature order (important for consistency)
features = joblib.load("artifacts/features.pkl")


#  Input schema
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


#  Health check
@app.get("/")
def home():
    return {"status": "API is running"}


#  Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    print(" Received input:", data)

    # Convert input → DataFrame
    df = pd.DataFrame([data.dict()])

    #  Ensure correct feature order (VERY IMPORTANT)
    df = df[features]

    # Predict
    prediction = model.predict(df)

    print(" Prediction:", prediction)

    return {"prediction": float(prediction[0])}