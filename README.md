# Spotify Music Data Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive Streamlit dashboard for exploring Spotify song data. Visualize trends in artist popularity, energy levels, explicit content, and more using dynamic charts. Includes ML models for predicting popularity and explicit content from audio features.

## Features

### Visualizations
- **Top Artists** — Horizontal bar chart of the 10 most popular artists
- **Popularity Trend** — Line chart tracking an artist's popularity over time
- **Energy Distribution** — Histogram of energy levels by year
- **Song Count** — Bar chart comparing song output across artists
- **Explicit Content** — Pie chart of explicit vs non-explicit tracks
- **Popularity Heatmap** — Artist x Year heatmap showing popularity patterns

### ML Models
- **Linear Regression** — Predict song popularity from audio features (MSE & R2)
- **Logistic Regression** — Classify explicit content (accuracy & classification report)
- **Decision Tree** — Alternative explicit content classifier

## Tech Stack

- **Frontend:** Streamlit, Plotly, Seaborn, Matplotlib
- **Data:** Pandas, NumPy
- **ML:** scikit-learn

## Installation

```bash
git clone https://github.com/Anlinnazateth/Music-Data-Analysis.git
cd Music-Data-Analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Use the sidebar to switch between **Visualization** and **Modeling** modes.

## Dataset

The `songs_normalize.csv` file contains 2,000 Spotify tracks with 18 attributes:

| Column | Description |
|--------|-------------|
| `artist` | Artist name |
| `song` | Track title |
| `duration_ms` | Duration in milliseconds |
| `explicit` | Explicit content flag |
| `year` | Release year |
| `popularity` | Popularity score (0-100) |
| `danceability` | Danceability (0-1) |
| `energy` | Energy level (0-1) |
| `key` | Musical key |
| `loudness` | Loudness (dB) |
| `speechiness` | Speech content (0-1) |
| `acousticness` | Acoustic quality (0-1) |
| `instrumentalness` | Instrumental content (0-1) |
| `valence` | Musical positivity (0-1) |
| `tempo` | Beats per minute |
| `genre` | Musical genre |

## Project Structure

```
Music-Data-Analysis/
├── app.py                  # Main Streamlit application
├── final.py                # Legacy app (reference)
├── songs_normalize.csv     # Dataset
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── .gitignore
├── .streamlit/
│   └── config.toml         # Streamlit theme
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
└── tests/
    └── test_app.py         # Unit tests
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
