import streamlit as st
import requests

st.title("📊 Job Market Prediction")

Region = st.number_input("Region", 0, 50)
employed = st.number_input("Employed", 0.0)
labour_rate = st.number_input("Labour Rate", 0.0)
Area = st.selectbox("Area", [0, 1])
lag_1 = st.number_input("Lag 1")
lag_2 = st.number_input("Lag 2")
rolling_avg = st.number_input("Rolling Avg")
Year = st.number_input("Year", 2000, 2030)
Month = st.number_input("Month", 1, 12)

if st.button("Predict"):
    payload = {
        "Region": int(Region),
        "employed": float(employed),
        "labour_rate": float(labour_rate),
        "Area": int(Area),
        "lag_1": float(lag_1),
        "lag_2": float(lag_2),
        "rolling_avg": float(rolling_avg),
        "Year": int(Year),
        "Month": int(Month)
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()
        prediction = result["prediction"]

        st.success(f"📈 Predicted Unemployment Rate: {prediction:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")