# mood_recommender.py
"""
KMeans Clustering-based Music Recommendation Engine

Pipeline:
1. User selects their profile and mood
2. Cluster all songs into pseudo-genres using KMeans
3. Find user's preferred clusters (weighted by play frequency)
4. Recommend songs from preferred clusters that match the mood
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_loader import FEATURE_COLUMNS
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Get script directory for file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(script_dir, "users")
SONGS_CSV = os.path.join(script_dir, "songs.csv")

# Mood profiles with target audio feature values
MOOD_PROFILES = {
    "happy": {
        "valence": 0.9,
        "energy": 0.8,
        "danceability": 0.8,
        "acousticness": 0.2,
        "instrumentalness": 0.1,
        "speechiness": 0.2,
        "liveness": 0.3,
        "loudness": -5.0,
        "tempo": 120.0,
    },
    "sad": {
        "valence": 0.2,
        "energy": 0.3,
        "danceability": 0.3,
        "acousticness": 0.6,
        "instrumentalness": 0.3,
        "speechiness": 0.1,
        "liveness": 0.2,
        "loudness": -12.0,
        "tempo": 80.0,
    },
    "energetic": {
        "valence": 0.7,
        "energy": 0.95,
        "danceability": 0.85,
        "acousticness": 0.1,
        "instrumentalness": 0.1,
        "speechiness": 0.15,
        "liveness": 0.4,
        "loudness": -4.0,
        "tempo": 135.0,
    },
    "calm": {
        "valence": 0.6,
        "energy": 0.3,
        "danceability": 0.4,
        "acousticness": 0.7,
        "instrumentalness": 0.4,
        "speechiness": 0.1,
        "liveness": 0.2,
        "loudness": -14.0,
        "tempo": 70.0,
    },
}

# Source weights for user track importance
SOURCE_WEIGHTS = {
    'liked': 3.0,      # Strongest preference signal
    'top': 2.0,        # Medium preference
    'recent': 1.0      # Recency indicator
}

# Number of mood clusters to learn from user
N_MOOD_CLUSTERS = 6

# Fuzzy matching threshold for title similarity
FUZZY_THRESHOLD = 0.85

# Artist familiarity boost multipliers
ARTIST_BOOST_FREQUENT = 1.5  # +50% for frequently played artists
ARTIST_BOOST_KNOWN = 1.25     # +25% for known artists


def build_mood_vector(mood_name):
    """Build a feature vector from mood profile."""
    mood_key = mood_name.lower()
    if mood_key not in MOOD_PROFILES:
        raise ValueError(
            f"Unknown mood '{mood_name}'. Available: {', '.join(MOOD_PROFILES.keys())}"
        )
    
    profile = MOOD_PROFILES[mood_key]
    # Ensure all 9 features are present in the correct order
    return np.array([profile[feat] for feat in FEATURE_COLUMNS])


def fuzzy_match_score(str1, str2):
    """Calculate fuzzy string similarity score (0 to 1)."""
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    return SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()


def normalize_string(s):
    """Normalize string for matching."""
    if pd.isna(s):
        return ""
    # Remove special characters, extra spaces, convert to lowercase
    import re
    s = re.sub(r'[^a-z0-9\s]', '', s.lower())
    return ' '.join(s.split())


def load_user_data(user_folder):
    """Load and combine all user track sources with source weighting."""
    print("✓ Loading user track sources...")
    
    user_tracks = []
    
    # Load liked tracks (highest weight)
    liked_path = os.path.join(user_folder, "spotify_saved_tracks.csv")
    if os.path.exists(liked_path):
        liked_df = pd.read_csv(liked_path)
        liked_df.columns = [c.lower().strip() for c in liked_df.columns]
        liked_df['source'] = 'liked'
        liked_df['source_weight'] = SOURCE_WEIGHTS['liked']
        user_tracks.append(liked_df)
        print(f"  Loaded {len(liked_df)} liked tracks")
    
    # Load top tracks (medium weight)
    top_path = os.path.join(user_folder, "spotify_top_tracks.csv")
    if os.path.exists(top_path):
        top_df = pd.read_csv(top_path)
        top_df.columns = [c.lower().strip() for c in top_df.columns]
        top_df['source'] = 'top'
        top_df['source_weight'] = SOURCE_WEIGHTS['top']
        user_tracks.append(top_df)
        print(f"  Loaded {len(top_df)} top tracks")
    
    # Load recently played (recency weight)
    recent_path = os.path.join(user_folder, "spotify_recently_played.csv")
    if os.path.exists(recent_path):
        recent_df = pd.read_csv(recent_path)
        recent_df.columns = [c.lower().strip() for c in recent_df.columns]
        recent_df['source'] = 'recent'
        recent_df['source_weight'] = SOURCE_WEIGHTS['recent']
        user_tracks.append(recent_df)
        print(f"  Loaded {len(recent_df)} recent tracks")
    
    if not user_tracks:
        return None
    
    # Combine all sources
    combined = pd.concat(user_tracks, ignore_index=True)
    
    # Ensure required columns exist
    if 'track_name' not in combined.columns or 'artists' not in combined.columns:
        print("✗ Missing required columns (track_name, artists)")
        return None
    
    # Normalize names for matching
    combined['track_name_norm'] = combined['track_name'].apply(normalize_string)
    combined['artists_norm'] = combined['artists'].apply(normalize_string)
    
    print(f"✓ Combined {len(combined)} total tracks from all sources")
    return combined


def load_global_scaler(songs_csv_path):
    """Fit a StandardScaler on the entire Kaggle dataset."""
    df = pd.read_csv(songs_csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def load_kaggle_songs(songs_csv_path):
    """Load the Kaggle dataset with all available songs."""
    df = pd.read_csv(songs_csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Keep only necessary columns
    keep_cols = ["track_id", "track_name", "artists"] + FEATURE_COLUMNS
    # Add album_name if it exists
    if "album_name" in df.columns:
        keep_cols.insert(3, "album_name")
    
    df = df[keep_cols].dropna(subset=FEATURE_COLUMNS)
    
    # Add normalized columns for fuzzy matching
    df['track_name_norm'] = df['track_name'].apply(normalize_string)
    df['artists_norm'] = df['artists'].apply(normalize_string)
    
    return df


def match_audio_features(user_df, kaggle_df):
    """Match user tracks to Kaggle audio features using optimized fuzzy matching."""
    print("✓ Matching user tracks to audio features...")
    
    matched_tracks = []
    unmatched_count = 0
    
    # Create a lookup dictionary for exact matches (much faster)
    exact_match_dict = {}
    for idx, row in kaggle_df.iterrows():
        key = (row['track_name_norm'], row['artists_norm'])
        if key not in exact_match_dict:  # Keep first occurrence
            exact_match_dict[key] = row
    
    print(f"  Built exact match index with {len(exact_match_dict)} unique tracks")
    
    for idx, user_row in user_df.iterrows():
        user_title_norm = user_row['track_name_norm']
        user_artist_norm = user_row['artists_norm']
        
        if not user_title_norm or not user_artist_norm:
            unmatched_count += 1
            continue
        
        # Try exact match first using dictionary lookup (O(1) instead of O(n))
        exact_key = (user_title_norm, user_artist_norm)
        if exact_key in exact_match_dict:
            match = exact_match_dict[exact_key]
        else:
            # Pre-filter candidates for fuzzy matching (massive speedup)
            # Filter by first character and similar length
            title_len = len(user_title_norm)
            title_first = user_title_norm[0] if user_title_norm else ''
            
            candidates = kaggle_df[
                (kaggle_df['track_name_norm'].str.len().between(title_len - 5, title_len + 5)) &
                (kaggle_df['track_name_norm'].str.startswith(title_first, na=False))
            ]
            
            # If too few candidates, expand search
            if len(candidates) < 50:
                candidates = kaggle_df[
                    kaggle_df['track_name_norm'].str.len().between(title_len - 10, title_len + 10)
                ]
            
            # Limit to max 500 candidates to keep it fast
            if len(candidates) > 500:
                candidates = candidates.sample(n=500, random_state=42)
            
            if candidates.empty:
                unmatched_count += 1
                continue
            
            # Fuzzy matching only on filtered candidates
            title_scores = candidates['track_name_norm'].apply(
                lambda x: fuzzy_match_score(x, user_title_norm)
            )
            artist_scores = candidates['artists_norm'].apply(
                lambda x: fuzzy_match_score(x, user_artist_norm)
            )
            combined_scores = title_scores * 0.7 + artist_scores * 0.3
            
            best_idx = combined_scores.idxmax()
            best_score = combined_scores.loc[best_idx]
            
            if best_score < FUZZY_THRESHOLD:
                unmatched_count += 1
                continue
            
            match = candidates.loc[best_idx]
        
        # Combine user metadata with Kaggle features
        matched_row = {
            'track_name': user_row.get('track_name', match['track_name']),
            'artists': user_row.get('artists', match['artists']),
            'source': user_row['source'],
            'source_weight': user_row['source_weight']
        }
        
        # Add audio features
        for feat in FEATURE_COLUMNS:
            matched_row[feat] = match[feat]
        
        matched_tracks.append(matched_row)
    
    if not matched_tracks:
        print("✗ No tracks could be matched to audio features")
        return None
    
    matched_df = pd.DataFrame(matched_tracks)
    print(f"✓ Matched {len(matched_df)}/{len(user_df)} tracks ({unmatched_count} unmatched)")
    
    return matched_df


def build_user_mood_clusters(user_df, scaler, n_clusters=N_MOOD_CLUSTERS):
    """Build K-means clusters representing user's natural listening moods."""
    print(f"✓ Building {n_clusters} mood clusters from user data...")
    
    # Extract and scale features
    features = user_df[FEATURE_COLUMNS].values
    features_scaled = scaler.transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Calculate cluster importance (weighted by source)
    cluster_importance = np.zeros(n_clusters)
    for cluster_id, weight in zip(clusters, user_df['source_weight'].values):
        cluster_importance[cluster_id] += weight
    cluster_importance = cluster_importance / cluster_importance.sum()
    
    print(f"  Cluster distribution: {cluster_importance}")
    
    return {
        'model': kmeans,
        'centroids': kmeans.cluster_centers_,
        'assignments': clusters,
        'importance': cluster_importance
    }


def cluster_songs(kaggle_df, scaler, n_clusters=N_MOOD_CLUSTERS, plot=False):
    """Cluster songs into pseudo-genres using KMeans."""
    features = kaggle_df[FEATURE_COLUMNS].values
    features_scaled = scaler.transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Visualize clustering if requested
    if plot:
        print("✓ Generating cluster visualization...")
        # Reduce to 2D using PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_scaled)
        
        # Sample 5000 points for clearer visualization (full dataset is too dense)
        sample_size = min(5000, len(features_2d))
        sample_idx = np.random.choice(len(features_2d), sample_size, replace=False)
        
        plt.figure(figsize=(14, 10))
        
        # Create a colormap
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = clusters[sample_idx] == cluster_id
            cluster_points = features_2d[sample_idx][cluster_mask]
            
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[cluster_id]], 
                       label=f'Cluster {cluster_id}',
                       alpha=0.6, 
                       s=30,
                       edgecolors='none')
        
        # Plot cluster centers
        centers_2d = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='black', 
                   marker='X', 
                   s=200, 
                   edgecolors='white',
                   linewidths=2,
                   label='Cluster Centers',
                   zorder=5)
        
        plt.title(f'Song Clustering Visualization ({n_clusters} Pseudo-Genres)\nPCA 2D Projection', 
                 fontsize=16, fontweight='bold')
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=12)
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=12)
        
        # Legend with smaller font
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=8, ncol=2, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(script_dir, 'cluster_visualization.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Cluster visualization saved to: {plot_path}")
        plt.close()
    
    return clusters, kmeans





def map_mood_to_cluster(mood_name, mood_clusters, scaler):
    """Map user-selected mood to the closest user cluster centroid."""
    print(f"✓ Mapping '{mood_name}' mood to user clusters...")
    
    # Get mood target vector
    mood_vec = build_mood_vector(mood_name)
    mood_vec_scaled = scaler.transform([mood_vec])[0]
    
    # Find closest cluster centroid
    centroids = mood_clusters['centroids']
    distances = np.linalg.norm(centroids - mood_vec_scaled, axis=1)
    best_cluster = np.argmin(distances)
    
    print(f"  Selected cluster {best_cluster} (distance: {distances[best_cluster]:.3f})")
    
    return best_cluster, centroids[best_cluster]


def extract_user_artists(user_df, weights=None):
    """
    Extract and count artists from user's listening history.
    Returns a dict mapping artist -> weighted play count.
    """
    artist_counts = {}
    
    if weights is None:
        weights = np.ones(len(user_df))
    
    for idx, artists_str in enumerate(user_df["artists"].fillna("")):
        # Handle multiple artists separated by semicolons or commas
        for artist in str(artists_str).split(";"):
            artist = artist.strip().lower()
            if artist:
                artist_counts[artist] = artist_counts.get(artist, 0) + weights[idx]
    
    return artist_counts


def compute_artist_familiarity(candidate_df, artist_counts):
    """
    Compute artist familiarity scores for candidate tracks.
    Returns a numpy array of scores (0 to 1), graduated by play frequency.
    """
    familiarity_scores = []
    
    # Get max count for normalization
    max_count = max(artist_counts.values()) if artist_counts else 1.0
    
    for artists_str in candidate_df["artists"].fillna(""):
        # Parse artists for this track
        track_artists = []
        for artist in str(artists_str).split(";"):
            artist = artist.strip().lower()
            if artist:
                track_artists.append(artist)
        
        # Get max familiarity among all artists on this track
        max_familiarity = 0.0
        for artist in track_artists:
            if artist in artist_counts:
                # Normalize by max count with square root for smoothing
                familiarity = np.sqrt(artist_counts[artist] / max_count)
                max_familiarity = max(max_familiarity, familiarity)
        
        familiarity_scores.append(max_familiarity)
    
    return np.array(familiarity_scores)


def recommend_mood_tracks(target_user, mood_name, top_n=10):
    """Generate mood-based recommendations using user's listening patterns.
    
    Args:
        target_user: Username
        mood_name: Selected mood
        top_n: Number of recommendations
    
    Returns:
        DataFrame of recommended tracks with scores
    """
    print(f"\n{'='*60}")
    print(f"MOOD-BASED RECOMMENDATION FOR '{target_user}'")
    print(f"Mood: {mood_name.upper()}")
    print(f"{'='*60}\n")
    
    # Step 1: Load all user data sources
    user_folder = os.path.join(USERS_DIR, target_user)
    user_df = load_user_data(user_folder)
    
    if user_df is None or user_df.empty:
        print(f"✗ No listening history found for user '{target_user}'")
        return pd.DataFrame()
    
    # Step 2: Load Kaggle dataset and scaler
    print("✓ Loading Kaggle dataset...")
    scaler = load_global_scaler(SONGS_CSV)
    kaggle_df = load_kaggle_songs(SONGS_CSV)
    
    # Step 3: Match user tracks to audio features
    user_matched = match_audio_features(user_df, kaggle_df)
    
    if user_matched is None or user_matched.empty:
        print("✗ Could not match user tracks to audio features")
        return pd.DataFrame()
    
    # Step 4: Build user mood clusters
    mood_clusters = build_user_mood_clusters(user_matched, scaler, n_clusters=N_MOOD_CLUSTERS)
    
    # Step 5: Map selected mood to best cluster
    target_cluster_id, target_centroid = map_mood_to_cluster(mood_name, mood_clusters, scaler)
    
    # Step 6: Build artist familiarity map
    print("✓ Building artist familiarity map...")
    artist_counts = {}
    for idx, row in user_matched.iterrows():
        artists = str(row['artists']).lower().split(';')
        for artist in artists:
            artist = artist.strip()
            if artist:
                artist_counts[artist] = artist_counts.get(artist, 0) + row['source_weight']
    
    # Normalize artist counts
    max_artist_count = max(artist_counts.values()) if artist_counts else 1.0
    print(f"  Found {len(artist_counts)} unique artists")
    
    # Step 7: Get user track names to exclude
    user_track_names_norm = set(user_matched['track_name'].apply(normalize_string))
    
    # Step 8: Filter candidates (exclude already heard)
    print("✓ Filtering candidate songs...")
    kaggle_df['track_norm_for_filter'] = kaggle_df['track_name'].apply(normalize_string)
    candidates = kaggle_df[~kaggle_df['track_norm_for_filter'].isin(user_track_names_norm)].copy()
    print(f"  {len(candidates)} candidates after filtering")
    
    if candidates.empty:
        print("✗ No new tracks available")
        return pd.DataFrame()
    
    # Step 9: Calculate mood distance scores
    print("✓ Computing mood match scores...")
    candidate_features = candidates[FEATURE_COLUMNS].values
    candidate_features_scaled = scaler.transform(candidate_features)
    
    # FIXED: Use the actual mood vector, not just cluster centroid
    # This ensures different moods get different results
    mood_vec = build_mood_vector(mood_name)
    mood_vec_scaled = scaler.transform([mood_vec])[0]
    
    # Distance to target mood vector (not cluster)
    distances = np.linalg.norm(candidate_features_scaled - mood_vec_scaled, axis=1)
    max_dist = distances.max() if distances.max() > 0 else 1
    mood_scores = 1 - (distances / max_dist)  # Convert to similarity
    
    # Step 10: Calculate artist familiarity boosts
    print("✓ Applying artist familiarity boosts...")
    artist_boosts = []
    for artists_str in candidates['artists'].fillna(''):
        artists = str(artists_str).lower().split(';')
        max_boost = 1.0  # No boost by default
        
        for artist in artists:
            artist = artist.strip()
            if artist in artist_counts:
                count = artist_counts[artist]
                if count >= max_artist_count * 0.3:  # Frequent artist
                    max_boost = max(max_boost, ARTIST_BOOST_FREQUENT)
                else:  # Known artist
                    max_boost = max(max_boost, ARTIST_BOOST_KNOWN)
        
        artist_boosts.append(max_boost)
    
    artist_boosts = np.array(artist_boosts)
    
    # Step 11: Final scoring (mood-first with artist boost)
    final_scores = mood_scores * artist_boosts
    
    # Step 12: Add scores and rank
    candidates['mood_score'] = mood_scores
    candidates['artist_boost'] = artist_boosts
    candidates['final_score'] = final_scores
    
    # Sort by final score
    candidates = candidates.sort_values('final_score', ascending=False)
    
    # Remove duplicates
    candidates = candidates.drop_duplicates(subset=['track_name', 'artists'], keep='first')
    
    # Apply artist diversity filter - only one song per artist
    seen_artists = set()
    diverse_recommendations = []
    
    for idx, row in candidates.iterrows():
        # Normalize artist names for comparison
        artists = str(row['artists']).lower().strip()
        
        # Check if we've already recommended this artist
        if artists not in seen_artists:
            diverse_recommendations.append(row)
            seen_artists.add(artists)
            
            # Stop when we have enough recommendations
            if len(diverse_recommendations) >= top_n:
                break
    
    # Convert to DataFrame
    recommendations = pd.DataFrame(diverse_recommendations)
    
    print(f"\n✓ Generated {len(recommendations)} diverse recommendations (1 per artist)\n")
    
    return recommendations


def main():
    print("\n" + "="*60)
    print("MOOD-BASED PERSONALIZED MUSIC RECOMMENDER")
    print("="*60)
    
    # Step 1: Select user
    if not os.path.exists(USERS_DIR):
        print(f"Error: Users directory not found at {USERS_DIR}")
        return
    
    available_users = [d for d in os.listdir(USERS_DIR) 
                      if os.path.isdir(os.path.join(USERS_DIR, d))]
    
    if not available_users:
        print("No users found in the users directory!")
        return
    
    print("\nAvailable users:")
    for i, user in enumerate(available_users, 1):
        print(f"{i}) {user}")
    
    try:
        user_choice = int(input("\nSelect a user (enter number): ").strip())
        if user_choice < 1 or user_choice > len(available_users):
            print("Invalid selection.")
            return
        target_user = available_users[user_choice - 1]
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    # Step 2: Select mood
    print(f"\nAvailable moods:")
    mood_list = list(MOOD_PROFILES.keys())
    for i, mood in enumerate(mood_list, 1):
        print(f"{i}) {mood}")
    
    mood_input = input("\nEnter mood name (or number): ").strip().lower()
    
    # Allow either mood name or number
    if mood_input.isdigit():
        mood_idx = int(mood_input)
        if 1 <= mood_idx <= len(mood_list):
            selected_mood = mood_list[mood_idx - 1]
        else:
            print("Invalid mood number.")
            return
    elif mood_input in MOOD_PROFILES:
        selected_mood = mood_input
    else:
        print(f"Unknown mood. Available: {', '.join(MOOD_PROFILES.keys())}")
        return
    
    # Step 3: Generate recommendations
    recommendations = recommend_mood_tracks(target_user, selected_mood, top_n=10)
    
    if recommendations.empty:
        print("\nNo recommendations generated.")
        return
    
    # Step 4: Display results
    print(f"\n{'='*60}")
    print(f"TOP 10 RECOMMENDATIONS FOR '{target_user}' - MOOD: '{selected_mood.upper()}'")
    print(f"{'='*60}\n")
    
    for idx, row in enumerate(recommendations.itertuples(), 1):
        track_name = row.track_name
        artists = row.artists
        final_score = row.final_score
        mood_score = row.mood_score
        artist_boost = row.artist_boost
        
        # Add indicator for artist familiarity
        if artist_boost >= ARTIST_BOOST_FREQUENT:
            artist_indicator = "⭐⭐"  # Frequent artist
        elif artist_boost >= ARTIST_BOOST_KNOWN:
            artist_indicator = "⭐"     # Known artist
        else:
            artist_indicator = ""        # New artist
        
        print(f"{idx}. {track_name} {artist_indicator}")
        print(f"   Artist: {artists}")
        print(f"   Score: {final_score:.3f} (Mood Match: {mood_score:.2f} × Artist Boost: {artist_boost:.2f}x)")
        print()


if __name__ == "__main__":
    main()