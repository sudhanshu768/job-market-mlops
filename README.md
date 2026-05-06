#  Job Market Prediction MLOps Pipeline

---

##  Overview

This project implements an **end-to-end MLOps pipeline** to predict unemployment trends using machine learning.

It integrates:

* Data processing
* Model training
* Experiment tracking
* Workflow orchestration
* API deployment
* CI/CD automation
* Monitoring

into a **single unified system**.

---

##  Architecture

```
Raw Data 
   ↓
Preprocessing
   ↓
Model Training
   ↓
Evaluation
   ↓
MLflow Tracking
   ↓
Airflow Orchestration
   ↓
FastAPI (Docker)
   ↓
Streamlit UI
   ↓
Monitoring (Evidently)
```

---

## ⚙️ Tech Stack

* **Python 3.11**
* **Airflow** – Workflow orchestration
* **MLflow** – Experiment tracking
* **FastAPI** – Model deployment
* **Docker** – Containerization
* **Streamlit** – User interface
* **Scikit-learn** – Machine learning
* **Pandas / NumPy** – Data processing
* **GitHub Actions** – CI/CD
* **Evidently AI** – Data drift monitoring

---

## 📁 Project Structure

```
job-market-mlops/
│
├── dags/
│   └── job_market_pipeline.py
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── model.pkl
│   └── features.pkl
│
├── reports/
│   └── data_drift_report.html
│
├── app/
│   ├── main.py        # FastAPI
│   └── app.py         # Streamlit UI
│
├── .github/workflows/
│   └── mlops_pipeline.yml
│
├── requirements.txt
└── README.md
```

---

##  Pipeline Workflow (Airflow DAG)

Airflow orchestrates the following steps:

1. **Ingest Data**
2. **Preprocess Data**
3. **Train Model**
4. **Evaluate Model**

---

##  Model Training

* **Algorithm:** Linear Regression

### Features:

* Region
* employed
* labour_rate
* Area
* lag_1, lag_2
* rolling_avg
* Year, Month

---

##  Experiment Tracking (MLflow)

MLflow is used to track:

* Parameters
* Metrics (MSE, RMSE)
* Model versions

---

##  Deployment

### 🔹 FastAPI (Dockerized)

Run API using Docker:

```bash
docker build -t job-market-api .
docker run -p 8000:8000 job-market-api
```

### API Endpoint:

```
POST /predict
```

### Sample Input:

```json
{
  "Region": 5,
  "employed": 1500000,
  "labour_rate": 42.5,
  "Area": 1,
  "lag_1": 6.2,
  "lag_2": 6.0,
  "rolling_avg": 6.1,
  "Year": 2020,
  "Month": 7
}
```

### Output:

```json
{
  "prediction": 5.65
}
```

---

##  Streamlit UI

Run:

```bash
streamlit run app/app.py
```

Open:

```
http://localhost:8501
```

👉 Provides a user-friendly interface for predictions.

---

## ⚙️ CI/CD Pipeline (GitHub Actions)

### Trigger:

* Push to `main` branch

### Steps:

* Install dependencies
* Run preprocessing
* Train model
* Evaluate model

-> Ensures pipeline reproducibility and early error detection.

---

##  Monitoring (Evidently AI)

Generates data drift report:

```
reports/data_drift_report.html
```

### Insight:

* No drift detected (demo dataset)

---

##  Evaluation Metrics

* **MSE ≈ 0**
* **RMSE ≈ 0**

> Note: Same dataset used for demo → near-perfect score.
> In real-world scenarios, separate test data should be used.

---

##  How to Run the Project

### 1️ Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2️ Run Airflow

```bash
airflow standalone
```

Open:

```
http://localhost:8080
```

---

### 3️ Run MLflow

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

### 4️ Run API (Docker)

```bash
docker run -p 8000:8000 job-market-api
```

---

### 5️ Run Streamlit UI

```bash
streamlit run app/app.py
```

---

##  Key Features

✔ End-to-End MLOps Pipeline
✔ Automated Workflow (Airflow)
✔ Experiment Tracking (MLflow)
✔ Model Deployment (FastAPI + Docker)
✔ User Interface (Streamlit)
✔ CI/CD Integration
✔ Data Drift Monitoring

---

## 🏁 Conclusion

This project demonstrates a **production-style MLOps workflow**, integrating automation, deployment, monitoring, and reproducibility for scalable machine learning systems.

---

## 👤 Author

**Sudhanshu Kandekar**
