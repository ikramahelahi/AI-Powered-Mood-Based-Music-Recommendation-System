# user_profiles.py
"""
Build user embeddings (mean of top-N tracks) and compute pairwise similarities.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_user_history_from_folder, FEATURE_COLUMNS

def load_global_scaler(songs_csv_path: str = "songs.csv"):
    """
    Fit a StandardScaler on all Kaggle audio features.
    This ensures all users are transformed the same way.
    """
    df = pd.read_csv(songs_csv_path)
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def compute_user_embedding(df: pd.DataFrame, scaler: StandardScaler, top_n: int = 10, weight_by_recency: bool = False) -> np.ndarray:
    """
    Compute a user embedding vector as the mean of top_n track feature vectors.
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe provided for user embedding computation.")

    if weight_by_recency and "played_at" in df.columns:
        df = df.sort_values("played_at", ascending=False)

    top = df.head(top_n) if len(df) >= top_n else df
    features = top[FEATURE_COLUMNS].astype(float).values
    if features.size == 0:
        raise ValueError("No feature columns found to compute embedding.")

    # Apply global scaler
    features_scaled = scaler.transform(features)

    # Compute mean embedding
    embedding = features_scaled.mean(axis=0)

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    return embedding

def load_all_user_profiles(users_root_dir: str = "users", 
                           top_n: int = 10,
                           language_filter=None, 
                           songs_csv_path: str = "songs.csv", weight_by_recency=False):
    """
    Walk through each subfolder under users_root_dir. For each subfolder (user),
    try to load user history and compute embedding. Returns a dict:
        { username: { 'df': df, 'embedding': np.array, 'num_tracks': int } }
    """
    users = {}
    scaler = load_global_scaler(songs_csv_path=songs_csv_path)

    for entry in os.listdir(users_root_dir):
        user_folder = os.path.join(users_root_dir, entry)
        if not os.path.isdir(user_folder):
            continue
        try:
            df = load_user_history_from_folder(user_folder, songs_csv_path=songs_csv_path, language_filter=language_filter)
            if df is None or df.empty:
                print(f"Skipping user '{entry}': no valid merged tracks.")
                continue
            emb = compute_user_embedding(df, scaler, top_n=top_n, weight_by_recency=weight_by_recency)
            users[entry] = {"df": df, "embedding": emb, "num_tracks": len(df)}
            print(f"Loaded user '{entry}': {len(df)} tracks, embedding shape {emb.shape}")
        except Exception as e:
            print(f"Error loading user '{entry}': {e}")
            continue

    return users

def pairwise_user_similarity(embeddings_dict: Dict[str, dict]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute cosine similarity matrix between all users.
    """
    users = list(embeddings_dict.keys())
    if not users:
        return pd.DataFrame(), np.array([[]])

    mats = np.stack([embeddings_dict[u]["embedding"] for u in users], axis=0)
    sims = cosine_similarity(mats)
    sim_df = pd.DataFrame(sims, index=users, columns=users)

    return sim_df, sims

def save_embeddings(embeddings_dict: Dict[str, dict], out_path: str = "user_embeddings.json"):
    """
    Save embeddings to a JSON-friendly dict (list for numpy arrays).
    """
    out = {}
    for user, info in embeddings_dict.items():
        out[user] = {
            "embedding": info["embedding"].tolist(),
            "num_tracks": int(info["num_tracks"])
        }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved embeddings to", out_path)


# sanity checked user_profiles.py
# """
# Build user embeddings (mean of top-N tracks) and compute pairwise similarities.
# Includes sanity checks for raw features, scaled features, and embeddings.
# """

# import os
# import json
# import numpy as np
# import pandas as pd
# from typing import Dict, Tuple
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity
# from data_loader import load_user_history_from_folder, FEATURE_COLUMNS

# def load_global_scaler(songs_csv_path: str = "songs.csv"):
#     """
#     Fit a StandardScaler on all Kaggle audio features.
#     This ensures all users are transformed the same way.
#     """
#     df = pd.read_csv(songs_csv_path)
#     X = df[FEATURE_COLUMNS].values
#     scaler = StandardScaler()
#     scaler.fit(X)
#     return scaler

# def compute_user_embedding(df: pd.DataFrame, scaler: StandardScaler, top_n: int = 10, weight_by_recency: bool = False) -> np.ndarray:
#     """
#     Compute a user embedding vector as the mean of top_n track feature vectors.
#     If the dataframe contains a 'played_at' column, you can optionally weight by recency.
#     """
#     if df is None or df.empty:
#         raise ValueError("Empty dataframe provided for user embedding computation.")

#     if weight_by_recency and "played_at" in df.columns:
#         df = df.sort_values("played_at", ascending=False)

#     top = df.head(top_n) if len(df) >= top_n else df

#     features = top[FEATURE_COLUMNS].astype(float).values
#     if features.size == 0:
#         raise ValueError("No feature columns found to compute embedding.")

#     # --- Sanity Check 1: raw features ---
#     print(f"\n[Sanity] First 3 rows of raw features for this user:")
#     print(top[FEATURE_COLUMNS].head(3))

#     # Apply global scaler
#     features_scaled = scaler.transform(features)

#     # --- Sanity Check 2: scaled features ---
#     print(f"[Sanity] Mean of scaled features (should be ~0): {np.round(features_scaled.mean(axis=0),3)}")

#     # Compute mean embedding
#     embedding = features_scaled.mean(axis=0)

#     # --- Sanity Check 3: embedding vector ---
#     print(f"[Sanity] User embedding vector: {np.round(embedding,3)}\n")

#     # L2 normalize
#     norm = np.linalg.norm(embedding)
#     if norm > 0:
#         embedding /= norm

#     return embedding

# def load_all_user_profiles(users_root_dir: str = "users", 
#                            top_n: int = 10,
#                            language_filter=None, 
#                            songs_csv_path: str = "songs.csv", weight_by_recency=False):
#     """
#     Walk through each subfolder under users_root_dir. For each subfolder (user),
#     try to load user history and compute embedding. Returns a dict:
#         { username: { 'df': df, 'embedding': np.array, 'num_tracks': int } }
#     """
#     users = {}
#     scaler = load_global_scaler(songs_csv_path=songs_csv_path)

#     for entry in os.listdir(users_root_dir):
#         user_folder = os.path.join(users_root_dir, entry)
#         if not os.path.isdir(user_folder):
#             continue
#         try:
#             df = load_user_history_from_folder(user_folder, songs_csv_path=songs_csv_path, language_filter=language_filter)
#             if df is None or df.empty:
#                 print(f"Skipping user '{entry}': no valid merged tracks.")
#                 continue
#             emb = compute_user_embedding(df, scaler, top_n=top_n, weight_by_recency=weight_by_recency)
#             users[entry] = {"df": df, "embedding": emb, "num_tracks": len(df)}
#             print(f"Loaded user '{entry}': {len(df)} tracks, embedding shape {emb.shape}")
#         except Exception as e:
#             print(f"Error loading user '{entry}': {e}")
#             continue

#     return users

# def pairwise_user_similarity(embeddings_dict: Dict[str, dict]) -> Tuple[pd.DataFrame, np.ndarray]:
#     """
#     Given users dict (from load_all_user_profiles), compute cosine similarity matrix.
#     Returns a pandas DataFrame (rows/cols = usernames) and the raw numpy matrix.
#     """
#     users = list(embeddings_dict.keys())
#     if not users:
#         return pd.DataFrame(), np.array([[]])

#     mats = np.stack([embeddings_dict[u]["embedding"] for u in users], axis=0)
#     sims = cosine_similarity(mats)  # shape (num_users, num_users)
#     sim_df = pd.DataFrame(sims, index=users, columns=users)

#     # --- Sanity Check 4: manual cosine similarity for first pair ---
#     if len(users) >= 2:
#         a = embeddings_dict[users[0]]['embedding']
#         b = embeddings_dict[users[1]]['embedding']
#         manual_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#         print(f"[Sanity] Manual cosine similarity between '{users[0]}' and '{users[1]}': {manual_sim:.3f}")

#     return sim_df, sims

# def save_embeddings(embeddings_dict: Dict[str, dict], out_path: str = "user_embeddings.json"):
#     """
#     Save embeddings to a JSON-friendly dict (list for numpy arrays).
#     """
#     out = {}
#     for user, info in embeddings_dict.items():
#         out[user] = {
#             "embedding": info["embedding"].tolist(),
#             "num_tracks": int(info["num_tracks"])
#         }
#     with open(out_path, "w") as fh:
#         json.dump(out, fh, indent=2)
#     print("Saved embeddings to", out_path)

