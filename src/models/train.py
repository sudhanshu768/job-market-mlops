import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn


def load_data():
    return pd.read_csv("data/processed/clean_data.csv")


def train_model(df):
    # Features & Target
    X = df.drop(columns=['unemployment', 'Date'], errors='ignore')
    y = df['unemployment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, mae, r2


def main():
    # Set experiment
    mlflow.set_experiment("job-market-prediction")

    df = load_data()

    with mlflow.start_run():
        model, mae, r2 = train_model(df)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        #  Log & Register Model
        mlflow.sklearn.log_model(
            model,
            name="model",  # new API (replaces artifact_path)
            registered_model_name="job-market-model"
        )

        print(f" MAE: {mae}")
        print(f" R2 Score: {r2}")
        print(" Model registered as 'job-market-model'")


if __name__ == "__main__":
    main()