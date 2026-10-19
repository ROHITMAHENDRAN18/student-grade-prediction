"""Data loading, cleaning checks, splitting and the leakage-safe preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from config import (DATA_PATH, TARGET, SEED, BINARY_COLS, NOMINAL_COLS, GRADE_COLS)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the UCI 'Student Performance' (Portuguese course) CSV (comma separated)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def numeric_cols(df: pd.DataFrame):
    """Numeric features = everything that is neither a binary flag nor a nominal code (target excluded)."""
    return [c for c in df.columns if c not in BINARY_COLS + NOMINAL_COLS + [TARGET]]


def clean_data(df: pd.DataFrame):
    """Remove exact duplicate rows; report missing values. Returns (clean_df, report dict)."""
    report = {"rows_before": len(df), "missing_cells": int(df.isna().sum().sum()),
              "duplicate_rows": int(df.duplicated().sum())}
    df = df.drop_duplicates().reset_index(drop=True)
    report["rows_after"] = len(df)
    return df, report


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] percentile, learned on TRAINING data only."""
    def __init__(self, lower=1, upper=99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.percentile(X, self.lower, axis=0)
        self.hi_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def feature_columns(df, use_grades=True):
    """Input columns for the two experimental settings:
       use_grades=True  -> all 32 features (includes G1, G2)
       use_grades=False -> 30 features (early-warning setting: no earlier grades)."""
    cols = [c for c in df.columns if c != TARGET]
    return cols if use_grades else [c for c in cols if c not in GRADE_COLS]


def build_preprocessor(columns) -> ColumnTransformer:
    """numeric -> winsorise -> standardise | nominal -> one-hot | binary -> 0/1 ordinal encoding."""
    num = [c for c in columns if c not in BINARY_COLS + NOMINAL_COLS]
    return ColumnTransformer([
        ("num", Pipeline([("clip", Winsorizer(1, 99)), ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [c for c in NOMINAL_COLS if c in columns]),
        ("bin", OrdinalEncoder(), [c for c in BINARY_COLS if c in columns]),
    ])


def split_data(df: pd.DataFrame, seed=SEED, val_size=0.15, test_size=0.15):
    """70/15/15 split. Stratified on grade band (fail <10, 10-13, >=14) so every subset has a similar grade mix."""
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(float)
    band = pd.cut(y, [-1, 9, 13, 20], labels=False)
    X_tr, X_tmp, y_tr, y_tmp, b_tr, b_tmp = train_test_split(
        X, y, band, test_size=val_size + test_size, stratify=band, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (val_size + test_size), stratify=b_tmp, random_state=seed)
    return X_tr, X_val, X_te, y_tr, y_val, y_te
