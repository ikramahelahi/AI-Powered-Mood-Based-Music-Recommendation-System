"""
feature_store.py

Responsible for loading offline audio features from songs.csv.
This replaces all direct dependency on Spotify's audio_features endpoint.
"""

import pandas as pd

# These are the numeric features we expect in songs.csv
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
    "tempo",
]


def load_global_audio_features(songs_csv_path: str = "songs.csv") -> pd.DataFrame:
    """
    Load global audio features from an offline CSV file (songs.csv).

    Expected minimum columns in songs.csv (case-insensitive):
        - track_id  (or 'id' or 'track_uri' that we can convert)
        - danceability
        - energy
        - valence
        - acousticness
        - instrumentalness
        - speechiness
        - liveness
        - loudness
        - tempo

    Returns:
        DataFrame with at least: ['track_id'] + FEATURE_COLUMNS
    """
    try:
        df = pd.read_csv(songs_csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find {songs_csv_path}. "
            f"Make sure your offline features CSV is prepared."
        ) from e

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle track_id column
    if "track_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "track_id"})
        elif "track_uri" in df.columns:
            df["track_id"] = df["track_uri"].str.replace(
                "spotify:track:", "", regex=False
            )
        else:
            raise ValueError(
                "songs.csv must contain a track identifier column. "
                "Expected one of: 'track_id', 'id', or 'track_uri'."
            )

    # Validate presence of required feature columns
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"songs.csv is missing required audio feature columns: {missing}. "
            f"Columns present are: {list(df.columns)}"
        )

    # Keep only what we need
    keep_cols = ["track_id"] + FEATURE_COLUMNS
    df = df[keep_cols].dropna(subset=FEATURE_COLUMNS)

    # Drop duplicates on track_id
    df = df.drop_duplicates(subset=["track_id"])

    return df
