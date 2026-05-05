from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

app = FastAPI(title="Job Market Prediction API")

# Load the Production model from MLflow Registry
model = mlflow.pyfunc.load_model("models:/job-market-model/Production")

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
    print("📥 Received input:", data)

    input_array = np.array([[ 
        data.Region,
        data.employed,
        data.labour_rate,
        data.Area,
        data.lag_1,
        data.lag_2,
        data.rolling_avg,
        data.Year,
        data.Month
    ]])

    prediction = model.predict(input_array)

    print("📤 Prediction:", prediction)

    return {"prediction": float(prediction[0])}