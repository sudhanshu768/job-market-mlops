import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

RAW_PATH_1 = os.path.join(BASE_DIR, "data/raw/Unemployment in India.csv")
RAW_PATH_2 = os.path.join(BASE_DIR, "data/raw/Unemployment_Rate_upto_11_2020.csv")

OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/clean_data.csv")


def load_data():
    if not os.path.exists(RAW_PATH_1) or not os.path.exists(RAW_PATH_2):
        print(" Dataset not found. Creating dummy data for CI/CD...")

        data = {
            "Region": ["A", "B", "C"],
            "Date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "Estimated Unemployment Rate (%)": [5.0, 6.0, 7.0],
            "Estimated Employed": [1000000, 1200000, 1100000],
            "Estimated Labour Participation Rate (%)": [40.0, 42.0, 41.0],
            "Area": ["Urban", "Rural", "Urban"]
        }

        df = pd.DataFrame(data)
        return df, df

    df1 = pd.read_csv(RAW_PATH_1)
    df2 = pd.read_csv(RAW_PATH_2)

    return df1, df2


def combine_data(df1, df2):
    df = pd.concat([df1, df2], ignore_index=True)
    return df.drop_duplicates()


def clean_data(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    numeric_cols = [
        'Estimated Unemployment Rate (%)',
        'Estimated Employed',
        'Estimated Labour Participation Rate (%)'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Date', 'Estimated Unemployment Rate (%)'])
    return df


def feature_engineering(df):
    df = df.sort_values(by=['Region', 'Date'])

    df['Area'] = df['Area'].map({'Urban': 1, 'Rural': 0})

    df['lag_1'] = df.groupby('Region')['Estimated Unemployment Rate (%)'].shift(1)
    df['lag_2'] = df.groupby('Region')['Estimated Unemployment Rate (%)'].shift(2)

    df['rolling_avg'] = (
        df.groupby('Region')['Estimated Unemployment Rate (%)']
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    df = df.dropna()
    return df


def clean_columns(df):
    df = df.rename(columns={
        'Estimated Unemployment Rate (%)': 'unemployment',
        'Estimated Employed': 'employed',
        'Estimated Labour Participation Rate (%)': 'labour_rate'
    })

    return df


def encode_features(df):
    df['Region'] = df['Region'].astype('category').cat.codes
    df['Area'] = df['Area'].fillna(0).astype(int)

    return df


def save_data(df):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f" Saved: {OUTPUT_PATH}")


def run_preprocessing():
    print(" Preprocessing started...")

    df1, df2 = load_data()
    df = combine_data(df1, df2)
    df = clean_data(df)
    df = feature_engineering(df)
    df = clean_columns(df)
    df = encode_features(df)

    save_data(df)

    print(" Preprocessing done!")


if __name__ == "__main__":
    run_preprocessing()