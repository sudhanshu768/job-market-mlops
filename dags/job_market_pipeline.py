from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.ingestion import run_ingestion
from src.data.preprocess import run_preprocessing
from src.models.train import run_training
from src.models.evaluate import run_evaluation


with DAG(
    dag_id="job_market_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="ingest_data",
        python_callable=run_ingestion
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocessing
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=run_training
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluation
    )

    t1 >> t2 >> t3 >> t4