import pandas as pd
import os


def load_data():
    file1 = "data/raw/Unemployment in India.csv"
    file2 = "data/raw/Unemployment_Rate_upto_11_2020.csv"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    return df1, df2


def combine_data(df1, df2):
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates()
    return df


def clean_data(df):
    # Fix Date format
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    # Clean numeric columns
    numeric_cols = [
        'Estimated Unemployment Rate (%)',
        'Estimated Employed',
        'Estimated Labour Participation Rate (%)',
        'longitude',
        'latitude'
    ]

    for col in numeric_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop essential missing values
    df = df.dropna(subset=[
        'Date',
        'Estimated Unemployment Rate (%)'
    ])

    return df


def feature_engineering(df):
    # Sort correctly
    df = df.sort_values(by=['Region', 'Date'])

    # Encode Area
    df['Area'] = df['Area'].map({'Urban': 1, 'Rural': 0})

    # Lag features per Region
    df['lag_1'] = df.groupby('Region')['Estimated Unemployment Rate (%)'].shift(1)
    df['lag_2'] = df.groupby('Region')['Estimated Unemployment Rate (%)'].shift(2)

    # Rolling average per Region
    df['rolling_avg'] = (
        df.groupby('Region')['Estimated Unemployment Rate (%)']
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Drop only rows affected by lag/rolling
    df = df.dropna(subset=['lag_1', 'lag_2', 'rolling_avg'])

    return df


def clean_columns(df):
    # Drop unnecessary columns
    df = df.drop(columns=['Region.1', 'Frequency'], errors='ignore')

    # 🔥 Drop geographic columns (FINAL FIX)
    df = df.drop(columns=['longitude', 'latitude'], errors='ignore')

    # Rename columns
    df = df.rename(columns={
        'Estimated Unemployment Rate (%)': 'unemployment',
        'Estimated Employed': 'employed',
        'Estimated Labour Participation Rate (%)': 'labour_rate'
    })

    return df


def encode_features(df):
    # Encode Region
    df['Region'] = df['Region'].astype('category').cat.codes

    # Ensure Area numeric
    df['Area'] = df['Area'].fillna(0).astype(int)

    # Force numeric conversion
    numeric_cols = [
        'unemployment',
        'employed',
        'labour_rate',
        'lag_1',
        'lag_2',
        'rolling_avg',
        'Year',
        'Month'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def save_data(df):
    output_path = "data/processed/clean_data.csv"

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Processed data saved at {output_path}")


def main():
    print("🔄 Starting preprocessing...")

    df1, df2 = load_data()
    df = combine_data(df1, df2)
    df = clean_data(df)
    df = feature_engineering(df)
    df = clean_columns(df)
    df = encode_features(df)

    save_data(df)

    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    main()