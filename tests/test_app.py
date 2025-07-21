"""Tests for the Music Data Analysis app."""

import os
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "songs_normalize.csv")


@pytest.fixture
def sample_df():
    """Load the dataset for testing."""
    return pd.read_csv(DATA_PATH)


class TestDataLoading:
    def test_csv_loads(self, sample_df):
        assert not sample_df.empty

    def test_required_columns_exist(self, sample_df):
        required = [
            "artist", "song", "year", "popularity", "energy",
            "danceability", "loudness", "speechiness", "acousticness",
            "instrumentalness", "valence", "tempo", "explicit",
        ]
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_no_all_null_rows(self, sample_df):
        assert not sample_df.isnull().all(axis=1).any()

    def test_year_is_numeric(self, sample_df):
        assert pd.api.types.is_numeric_dtype(sample_df["year"])

    def test_popularity_range(self, sample_df):
        assert sample_df["popularity"].min() >= 0
        assert sample_df["popularity"].max() <= 100


class TestPreprocessing:
    def test_dropna_reduces_or_keeps(self, sample_df):
        cleaned = sample_df.dropna()
        assert len(cleaned) <= len(sample_df)

    def test_year_filtering(self, sample_df):
        years = sample_df["year"].unique()[:3]
        filtered = sample_df[sample_df["year"].isin(years)]
        assert all(filtered["year"].isin(years))


class TestModeling:
    def test_linear_regression_runs(self, sample_df):
        features = ["danceability", "energy", "loudness", "valence", "tempo"]
        df = sample_df.dropna(subset=features + ["popularity"])
        X = df[features]
        y = df["popularity"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
