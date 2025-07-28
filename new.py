import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Set Streamlit page configuration
st.set_page_config(page_title="Spotify Data Analysis Dashboard", layout="wide")

# Load the dataset
data_path = "./songs_normalize.csv"
df = pd.read_csv(data_path)

# Ensure data integrity
df = df.dropna()

# Display a title for the dashboard
st.title("Spotify Data Analysis Dashboard")

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
st.header("Popularity Heatmap by artist and Year")
selected_artists = st.multiselect("Select artists:", options=df["artist"].unique(), default=df["artist"].unique()[:5])
filtered_artists_df = df[(df["artist"].isin(selected_artists)) & (df["year"].isin(years))]

if not filtered_artists_df.empty:
    heatmap_data = filtered_artists_df.pivot_table(index="artist", columns="year", values="popularity", aggfunc="mean").fillna(0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
    st.pyplot(plt)
else:
    st.warning("No data available for the selected countries and years.")


