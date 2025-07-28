import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Apple Music Data Analysis Dashboard", layout="wide")

# Load the dataset
data_path = "./songs_normalize.csv"
df = pd.read_csv(data_path)

# Ensure data integrity
df = df.dropna()

# Sidebar for navigation
st.sidebar.title("Apple Music Dashboard")
option = st.sidebar.radio("Select an Option:", ("Visualization", "Modeling"))

if option == "Visualization":
    # Visualization Section

    st.title("Apple Music Data Analysis Dashboard")

    # Display dataset information
    if st.checkbox("Show raw data"):
        st.write(df.head())

    # Dropdown to select years
    years = st.multiselect("Select Time Period (Years):", options=df["year"].unique(), default=df["year"].unique()[:5])
    filtered_year_df = df[df["year"].isin(years)]

    # Horizontal bar chart for top artists by popularity
    st.header("Top Artists by Popularity")
    if not filtered_year_df.empty:
        top_artists = filtered_year_df.groupby("artist")["popularity"].mean().nlargest(10).reset_index()
        fig = px.bar(top_artists, x="popularity", y="artist", orientation="h", color="popularity", color_continuous_scale="YlGnBu", labels={"artist": "Artist", "popularity": "Popularity"})
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected years.")

    # Dropdown to select a single artist for popularity trend
    target_artist = st.selectbox("Select an Artist for Popularity Trend:", options=df["artist"].unique())
    artist_df = df[df["artist"] == target_artist]

    if not artist_df.empty:
        st.header(f"Popularity Trend for {target_artist}")
        fig = px.line(artist_df, x="year", y="popularity", color_discrete_sequence=px.colors.sequential.Plasma, labels={"year": "Year", "popularity": "Popularity"})
        st.plotly_chart(fig)
    else:
        st.warning(f"No data available for {target_artist}.")

    # Histogram of energy for selected year
    st.header("Histogram of Energy")
    selected_year = st.selectbox("Select a Year:", options=df["year"].unique())
    year_df = df[df["year"] == selected_year]

    if not year_df.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(year_df["energy"], bins=20, color="red", edgecolor="black")
        plt.title(f"Energy Distribution for {selected_year}")
        plt.xlabel("Energy")
        plt.ylabel("Frequency")
        st.pyplot(plt)
    else:
        st.warning(f"No data available for {selected_year}.")

    # Bar plot for songs by selected artists
    multi_artists = st.multiselect("Select Artists for Song Count:", options=df["artist"].unique(), default=df["artist"].unique()[:5])
    artist_song_count = df[df["artist"].isin(multi_artists)].groupby("artist")["song"].count().reset_index().rename(columns={"song": "song_count"})

    if not artist_song_count.empty:
        st.header("Number of Songs by Selected Artists")
        fig = px.bar(artist_song_count, x="artist", y="song_count", color="song_count", color_continuous_scale="viridis", labels={"artist": "Artist", "song_count": "Number of Songs"})
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected artists.")

    # Pie chart for explicit content
    st.header("Songs with Explicit Content")
    explicit_counts = df["explicit"].value_counts().reset_index()
    explicit_counts.columns = ["explicit", "count"]  # Rename columns to 'explicit' and 'count'
    fig = px.pie(explicit_counts, names="explicit", values="count", color_discrete_sequence=px.colors.sequential.Pinkyl, labels={"explicit": "Explicit Content", "count": "Count"})
    st.plotly_chart(fig)

    # Heatmap for popularity by countries and years
    st.header("Popularity Heatmap by Artist and Year")
    selected_artists = st.multiselect("Select artists:", options=df["artist"].unique(), default=df["artist"].unique()[:5])
    filtered_artists_df = df[(df["artist"].isin(selected_artists)) & (df["year"].isin(years))]

    if not filtered_artists_df.empty:
        heatmap_data = filtered_artists_df.pivot_table(index="artist", columns="year", values="popularity", aggfunc="mean").fillna(0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
        st.pyplot(plt)
    else:
        st.warning("No data available for the selected artists and years.")

elif option == "Modeling":
    # Modeling Section

    st.title("Apple Music Data Modeling")

    # Prepare the data for modeling
    features = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "valence", "tempo"]

    if all(feature in df.columns for feature in features):
        X = df[features]

        model_option = st.selectbox("Select a Model:", ("Linear Regression", "Logistic Regression", "Decision Tree Classifier"))

        if model_option == "Linear Regression":
            st.header("Linear Regression")
            target = "popularity"

            if target in df.columns:
                y = df[target]

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Linear Regression Model
                linear_model = LinearRegression()
                linear_model.fit(X_train, y_train)

                y_pred_linear = linear_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred_linear)
                r2 = r2_score(y_test, y_pred_linear)

                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R-Squared: {r2:.2f}")

        elif model_option == "Logistic Regression":
            st.header("Logistic Regression")
            df["explicit_binary"] = df["explicit"].apply(lambda x: 1 if x else 0)
            y_logistic = df["explicit_binary"]

            # Split the data
            X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_logistic, test_size=0.3, random_state=42)

            # Logistic Regression Model
            logistic_model = LogisticRegression(max_iter=1000)
            logistic_model.fit(X_train_log, y_train_log)

            y_pred_logistic = logistic_model.predict(X_test_log)
            accuracy = accuracy_score(y_test_log, y_pred_logistic)

            st.write(f"Accuracy: {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test_log, y_pred_logistic))

        elif model_option == "Decision Tree Classifier":
            st.header("Decision Tree Classifier")
            df["explicit_binary"] = df["explicit"].apply(lambda x: 1 if x else 0)
            y_tree = df["explicit_binary"]

            # Split the data
            X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X, y_tree, test_size=0.3, random_state=42)

            # Decision Tree Classifier Model
            tree_model = DecisionTreeClassifier(random_state=42)
            tree_model.fit(X_train_tree, y_train_tree)

            y_pred_tree = tree_model.predict(X_test_tree)
            accuracy_tree = accuracy_score(y_test_tree, y_pred_tree)

            st.write(f"Accuracy: {accuracy_tree:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test_tree, y_pred_tree))

    else:
        st.error("Dataset does not contain required features for modeling.")

