import streamlit as st
import requests

st.set_page_config(page_title="Job Market Prediction", layout="centered")

st.title("📊 Job Market Prediction App")
st.write("Enter the details below to predict unemployment rate")

# Inputs
Region = st.number_input("Region", min_value=0, max_value=50, value=5)
employed = st.number_input("Employed", value=1500000.0)
labour_rate = st.number_input("Labour Rate (%)", value=42.5)
Area = st.selectbox("Area (0 = Rural, 1 = Urban)", [0, 1])
lag_1 = st.number_input("Lag 1", value=6.2)
lag_2 = st.number_input("Lag 2", value=6.0)
rolling_avg = st.number_input("Rolling Average", value=6.1)
Year = st.number_input("Year", min_value=2000, max_value=2030, value=2020)
Month = st.number_input("Month", min_value=1, max_value=12, value=7)

# Button
if st.button("Predict 🚀"):

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

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]

            st.success(f"📈 Predicted Unemployment Rate: {prediction:.2f}%")

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error("🚨 Could not connect to FastAPI. Make sure Docker API is running.")