import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import TIME_COL, TARGET_COL, SELECTED_FEATURES, SEQ_LENGTH

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).drop_duplicates().reset_index(drop=True)
    return df


def select_features(df: pd.DataFrame):
    missing_cols = [col for col in SELECTED_FEATURES if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    return SELECTED_FEATURES


def fit_scalers(train_df: pd.DataFrame, feature_cols: list):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(train_df[feature_cols])
    y_scaler.fit(train_df[[TARGET_COL]])

    return x_scaler, y_scaler


def transform_features(df: pd.DataFrame, feature_cols: list, x_scaler, y_scaler):
    X_scaled = x_scaler.transform(df[feature_cols])
    y_scaled = y_scaler.transform(df[[TARGET_COL]])
    return X_scaled, y_scaled


def create_sequences(X_scaled, y_scaled, seq_length=SEQ_LENGTH):
    X_seq, y_seq = [], []

    for i in range(seq_length, len(X_scaled)):
        X_seq.append(X_scaled[i - seq_length:i])
        y_seq.append(y_scaled[i][0])

    return np.array(X_seq), np.array(y_seq)