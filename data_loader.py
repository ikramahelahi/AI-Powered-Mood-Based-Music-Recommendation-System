"""
Responsible for:
- Loading user-level CSV exports (top tracks, saved tracks, recently played)
- Optional language-based filtering (English vs Urdu/Hindi)
- Joining with offline audio features from songs.csv

This version is backward compatible and also provides:
- load_user_history_from_folder(user_folder, songs_csv_path, language_filter)
"""
import os
import pandas as pd

# Audio feature columns from Kaggle dataset
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
    "tempo"
]


def _read_csv_if_exists(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            return df
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None
    return None


def load_user_history(language_filter=None, songs_csv_path="songs.csv"):
    """
    Original API (keeps previous behaviour) - looks for CSVs in current working dir.
    """
    return load_user_history_from_folder(".", songs_csv_path=songs_csv_path, language_filter=language_filter)


def load_user_history_from_folder(user_folder: str, songs_csv_path: str = "songs.csv", language_filter=None):
    """
    Loads one user's listening history from a folder containing:
        - spotify_top_tracks.csv
        - spotify_saved_tracks.csv
        - spotify_recently_played.csv

    Then merges with Kaggle songs.csv on track_id and returns merged DataFrame
    with both metadata and numeric audio features.

    Returns:
        pd.DataFrame or None
    """
    print(f"Loading Spotify history from folder: {user_folder}")

    csv_files = [
        os.path.join(user_folder, "spotify_top_tracks.csv"),
        os.path.join(user_folder, "spotify_saved_tracks.csv"),
        os.path.join(user_folder, "spotify_recently_played.csv"),
    ]

    dfs = []
    for f in csv_files:
        df = _read_csv_if_exists(f)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("✗ No user CSV files found in", user_folder)
        return None

    user_df = pd.concat(dfs, ignore_index=True, sort=False)
    user_df.columns = [c.lower().strip() for c in user_df.columns]

    # Normalize expected columns
    required_cols = ["track_id", "track_name", "artists", "album_name"]
    for col in required_cols:
        if col not in user_df.columns:
            print(f"✗ Missing required column in user CSVs ({user_folder}): {col}")
            return None

    # Remove duplicates by track_id
    user_df = user_df.drop_duplicates(subset=["track_id"])

    # Optional language filtering (keeps your existing behaviour)
    if language_filter and "artist_genres" in user_df.columns:
        print(f"Applying language filter: {language_filter}")
        genre_keywords = [
            "desi", "hindi", "urdu", "bollywood", "ghazal", "sufi",
            "qawwali", "punjabi", "bhangra", "indian",
            "desi pop", "hindi indie", "indian indie",
            "desi hip hop", "punjabi pop"
        ]
        user_df["artist_genres"] = user_df["artist_genres"].fillna("").str.lower()
        if language_filter == "english":
            mask = ~user_df["artist_genres"].str.contains("|".join(genre_keywords))
        elif language_filter == "urdu":
            mask = user_df["artist_genres"].str.contains("|".join(genre_keywords))
        else:
            mask = [True] * len(user_df)
        user_df = user_df[mask]

    # ---------------------------------------------------------
    # Load Kaggle features (songs.csv)
    # ---------------------------------------------------------
    if not os.path.exists(songs_csv_path):
        print(f"✗ Kaggle features file not found: {songs_csv_path}")
        return None

    feat_df = pd.read_csv(songs_csv_path)
    feat_df.columns = [c.lower().strip() for c in feat_df.columns]

    # Ensure Kaggle file has required audio columns
    for col in FEATURE_COLUMNS:
        if col not in feat_df.columns:
            print(f"✗ Feature missing in Kaggle dataset: {col}")
            return None

    # Keep only track_id + features (prevents accidental large joins)
    feat_df = feat_df[["track_id"] + FEATURE_COLUMNS]

    print("Merging user metadata with Kaggle audio features...")
    merged = user_df.merge(feat_df, on="track_id", how="left")

    before = len(merged)
    merged = merged.dropna(subset=FEATURE_COLUMNS)
    after = len(merged)
    print(f"✓ Tracks with audio features: {after}/{before}")

    if merged.empty:
        print("✗ No tracks left after merging with audio features.")
        return None

    # Reset index for safety
    merged = merged.reset_index(drop=True)

    return merged
