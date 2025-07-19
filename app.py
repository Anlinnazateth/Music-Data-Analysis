"""Spotify Music Data Analysis Dashboard.

An interactive Streamlit dashboard for visualizing Spotify song data
and training ML models on audio features.
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Music Data Analysis",
    page_icon="🎵",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "artist", "song", "year", "popularity", "energy",
    "danceability", "loudness", "speechiness", "acousticness",
    "instrumentalness", "valence", "tempo", "explicit",
]

FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "valence", "tempo",
]


@st.cache_data
def load_data(path: str = "songs_normalize.csv") -> pd.DataFrame:
    """Load and validate the Spotify songs dataset."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Dataset not found at `{path}`. Please place the CSV file in the project root.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()

    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🎵 Spotify Dashboard")
page = st.sidebar.radio("Navigate", ["Visualization", "Modeling"])

# ---------------------------------------------------------------------------
# Visualization page
# ---------------------------------------------------------------------------
if page == "Visualization":
    st.title("Spotify Music Data Analysis")

    # Dataset preview
    if st.checkbox("Show raw data"):
        st.dataframe(df.head(50), use_container_width=True)

    # --- Filters ---
    years = sorted(df["year"].unique())
    selected_years = st.multiselect(
        "Select years:", options=years, default=years[:5]
    )
    filtered_df = df[df["year"].isin(selected_years)]

    # --- Top artists by popularity ---
    st.header("Top 10 Artists by Popularity")
    if not filtered_df.empty:
        top_artists = (
            filtered_df.groupby("artist")["popularity"]
            .mean()
            .nlargest(10)
            .reset_index()
        )
        fig = px.bar(
            top_artists,
            x="popularity",
            y="artist",
            orientation="h",
            color="popularity",
            color_continuous_scale="YlGnBu",
            labels={"artist": "Artist", "popularity": "Avg Popularity"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data for the selected years.")

    # --- Artist popularity trend ---
    target_artist = st.selectbox(
        "Select an artist for popularity trend:", df["artist"].unique()
    )
    artist_df = df[df["artist"] == target_artist]
    if not artist_df.empty:
        st.header(f"Popularity Trend — {target_artist}")
        fig = px.line(
            artist_df,
            x="year",
            y="popularity",
            color_discrete_sequence=px.colors.sequential.Plasma,
            labels={"year": "Year", "popularity": "Popularity"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Energy histogram ---
    st.header("Energy Distribution")
    selected_year = st.selectbox("Select a year:", df["year"].unique())
    year_df = df[df["year"] == selected_year]
    if not year_df.empty:
        fig = px.histogram(
            year_df,
            x="energy",
            nbins=20,
            color_discrete_sequence=["#1DB954"],
            labels={"energy": "Energy"},
            title=f"Energy Distribution ({selected_year})",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Song count by artist ---
    multi_artists = st.multiselect(
        "Select artists for song count:",
        options=df["artist"].unique(),
        default=list(df["artist"].unique()[:5]),
    )
    if multi_artists:
        song_counts = (
            df[df["artist"].isin(multi_artists)]
            .groupby("artist")["song"]
            .count()
            .reset_index()
            .rename(columns={"song": "song_count"})
        )
        st.header("Song Count by Artist")
        fig = px.bar(
            song_counts,
            x="artist",
            y="song_count",
            color="song_count",
            color_continuous_scale="Viridis",
            labels={"artist": "Artist", "song_count": "Songs"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Explicit content pie chart ---
    st.header("Explicit vs Non-Explicit")
    explicit_counts = df["explicit"].value_counts().reset_index()
    explicit_counts.columns = ["explicit", "count"]
    fig = px.pie(
        explicit_counts,
        names="explicit",
        values="count",
        color_discrete_sequence=px.colors.sequential.Pinkyl,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Popularity heatmap ---
    st.header("Popularity Heatmap (Artist x Year)")
    heatmap_artists = st.multiselect(
        "Select artists for heatmap:",
        options=df["artist"].unique(),
        default=list(df["artist"].unique()[:5]),
    )
    hm_df = df[
        (df["artist"].isin(heatmap_artists)) & (df["year"].isin(selected_years))
    ]
    if not hm_df.empty:
        pivot = hm_df.pivot_table(
            index="artist", columns="year", values="popularity", aggfunc="mean"
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data for the selected artists and years.")

# ---------------------------------------------------------------------------
# Modeling page
# ---------------------------------------------------------------------------
elif page == "Modeling":
    st.title("Audio Feature Modeling")

    if not all(f in df.columns for f in FEATURE_COLUMNS):
        st.error("Dataset does not contain required audio feature columns.")
        st.stop()

    X = df[FEATURE_COLUMNS]

    model_choice = st.selectbox(
        "Select a model:",
        ["Linear Regression", "Logistic Regression", "Decision Tree Classifier"],
    )

    if model_choice == "Linear Regression":
        st.header("Linear Regression — Predict Popularity")
        y = df["popularity"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mean_squared_error(y_test, preds):.2f}")
        col2.metric("R-Squared", f"{r2_score(y_test, preds):.2f}")

        results = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
        st.subheader("Predictions vs Actual")
        st.dataframe(results.head(20), use_container_width=True)

    elif model_choice == "Logistic Regression":
        st.header("Logistic Regression — Predict Explicit Content")
        y = df["explicit"].apply(lambda x: 1 if x else 0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2%}")
        st.subheader("Classification Report")
        st.text(classification_report(y_test, preds))

    elif model_choice == "Decision Tree Classifier":
        st.header("Decision Tree — Predict Explicit Content")
        y = df["explicit"].apply(lambda x: 1 if x else 0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2%}")
        st.subheader("Classification Report")
        st.text(classification_report(y_test, preds))
