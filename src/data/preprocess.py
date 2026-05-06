import pandas as pd
import os

# ==============================
# Base Paths
# ==============================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

RAW_PATH_1 = os.path.join(
    BASE_DIR,
    "data/raw/Unemployment in India.csv"
)

RAW_PATH_2 = os.path.join(
    BASE_DIR,
    "data/raw/Unemployment_Rate_upto_11_2020.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data/processed/clean_data.csv"
)

MAPPING_PATH = os.path.join(
    BASE_DIR,
    "data/processed/region_mapping.csv"
)


# ==============================
# Load Data
# ==============================

def load_data():

    # ==============================
    # CI/CD Fallback Dummy Dataset
    # ==============================

    if not os.path.exists(RAW_PATH_1) or not os.path.exists(RAW_PATH_2):

        print("⚠️ Dataset not found. Using dummy dataset for CI/CD...")

        data = {
            "Region": ["A", "A", "A", "B", "B", "B"],

            "Date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-01-01",
                "2020-02-01",
                "2020-03-01"
            ],

            "Estimated Unemployment Rate (%)": [
                5.0,
                6.0,
                7.0,
                4.5,
                5.5,
                6.5
            ],

            "Estimated Employed": [
                1000000,
                1100000,
                1200000,
                900000,
                950000,
                1000000
            ],

            "Estimated Labour Participation Rate (%)": [
                40.0,
                42.0,
                41.0,
                39.0,
                40.0,
                41.0
            ],

            "Area": [
                "Urban",
                "Urban",
                "Urban",
                "Rural",
                "Rural",
                "Rural"
            ]
        }

        df = pd.DataFrame(data)

        return df, df

    # ==============================
    # Load Real Dataset
    # ==============================

    df1 = pd.read_csv(RAW_PATH_1)
    df2 = pd.read_csv(RAW_PATH_2)

    # Remove extra spaces
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    print("\n Dataset 1 Columns:")
    print(df1.columns)

    print("\n Dataset 2 Columns:")
    print(df2.columns)

    return df1, df2


# ==============================
# Combine Data
# ==============================

def combine_data(df1, df2):

    # Add missing Area column
    if 'Area' not in df2.columns:
        df2['Area'] = 'Unknown'

    df = pd.concat(
        [df1, df2],
        ignore_index=True
    )

    df = df.drop_duplicates()

    return df


# ==============================
# Clean Data
# ==============================

def clean_data(df):

    # Convert Date column
    df['Date'] = pd.to_datetime(
        df['Date'],
        errors='coerce',
        dayfirst=True
    )

    # Numeric columns
    numeric_cols = [
        'Estimated Unemployment Rate (%)',
        'Estimated Employed',
        'Estimated Labour Participation Rate (%)'
    ]

    # Convert numeric columns
    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    # Remove important missing values
    df = df.dropna(
        subset=[
            'Date',
            'Estimated Unemployment Rate (%)'
        ]
    )

    return df


# ==============================
# Feature Engineering
# ==============================

def feature_engineering(df):

    # Sort values
    df = df.sort_values(
        by=['Region', 'Date']
    )

    # Encode Area
    df['Area'] = df['Area'].map({
        'Urban': 1,
        'Rural': 0,
        'Unknown': -1
    })

    # Lag Features
    df['lag_1'] = (
        df.groupby('Region')
        ['Estimated Unemployment Rate (%)']
        .shift(1)
    )

    df['lag_2'] = (
        df.groupby('Region')
        ['Estimated Unemployment Rate (%)']
        .shift(2)
    )

    # Rolling Average
    df['rolling_avg'] = (
        df.groupby('Region')
        ['Estimated Unemployment Rate (%)']
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Extract Year and Month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Remove NaNs created by lag features
    df = df.dropna(
        subset=[
            'lag_1',
            'lag_2',
            'rolling_avg'
        ]
    )

    return df


# ==============================
# Rename Columns
# ==============================

def clean_columns(df):

    df = df.rename(columns={

        'Estimated Unemployment Rate (%)':
            'unemployment',

        'Estimated Employed':
            'employed',

        'Estimated Labour Participation Rate (%)':
            'labour_rate'
    })

    return df


# ==============================
# Encode Features
# ==============================

def encode_features(df):

    # Convert Region to categorical
    region_category = (
        df['Region']
        .astype('category')
    )

    # Store mapping
    region_mapping = dict(
        enumerate(
            region_category.cat.categories
        )
    )

    print("\n Region Encoding Mapping:")
    print(region_mapping)

    # Save mapping
    mapping_df = pd.DataFrame(
        list(region_mapping.items()),
        columns=['Code', 'Region']
    )

    os.makedirs(
        os.path.dirname(MAPPING_PATH),
        exist_ok=True
    )

    mapping_df.to_csv(
        MAPPING_PATH,
        index=False
    )

    print(f"\n Saved Mapping: {MAPPING_PATH}")

    # Encode Region
    df['Region'] = (
        region_category.cat.codes
    )

    # Fill Area nulls
    df['Area'] = (
        df['Area']
        .fillna(-1)
        .astype(int)
    )

    return df


# ==============================
# Save Processed Data
# ==============================

def save_data(df):

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\n Saved Clean Data: {OUTPUT_PATH}")


# ==============================
# Main Pipeline
# ==============================

def run_preprocessing():

    print("\n Preprocessing started...")

    # Load data
    df1, df2 = load_data()

    # Combine
    df = combine_data(df1, df2)

    # Clean
    df = clean_data(df)

    # Feature engineering
    df = feature_engineering(df)

    # Rename columns
    df = clean_columns(df)

    # Encode features
    df = encode_features(df)

    # Save processed dataset
    save_data(df)

    print("\n Preprocessing done!")


# ==============================
# Run Script
# ==============================

if __name__ == "__main__":

    run_preprocessing()