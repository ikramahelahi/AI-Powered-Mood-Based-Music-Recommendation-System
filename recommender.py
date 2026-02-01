# recommender.py
"""
User Similarity-based Recommendation Engine

- Uses cosine similarity between user embeddings
- Recommends top-N tracks to a user based on what similar users like
"""

from user_profiles import load_all_user_profiles, pairwise_user_similarity
import numpy as np

class UserSimilarityRecommender:
    def __init__(self, similarity_dict, user_track_history, track_metadata):
        """
        similarity_dict: { user1: {user2: sim, ...}, ... }
        user_track_history: { user: [track_id, ...] }
        track_metadata: { track_id: {"track_name":..., "artists":...} }
        """
        self.sim = similarity_dict
        self.user_track_history = user_track_history
        self.track_metadata = track_metadata

    def recommend(self, target_user, top_n=10):
        if target_user not in self.sim:
            print(f"User '{target_user}' not found in similarity matrix.")
            return []

        # Get other users sorted by similarity
        other_users = [(u, s) for u, s in self.sim[target_user].items() if u != target_user]
        other_users.sort(key=lambda x: x[1], reverse=True)  # descending similarity

        track_scores = {}
        for u, sim_score in other_users:
            for track_id in self.user_track_history.get(u, []):
                # Do not recommend tracks the target user already has
                if track_id in self.user_track_history.get(target_user, []):
                    continue
                if track_id not in track_scores:
                    track_scores[track_id] = 0
                track_scores[track_id] += sim_score  # weighted by similarity

        # Sort tracks by cumulative score
        sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
        top_tracks = sorted_tracks[:top_n]

        # Convert to readable format
        recommendations = []
        for track_id, score in top_tracks:
            info = self.track_metadata.get(track_id, {})
            track_name = info.get("track_name", "Unknown Track")
            artists = info.get("artists", "Unknown Artist")
            recommendations.append({
                "track_id": track_id,
                "track_name": track_name,
                "artists": artists,
                "score": round(score, 3)
            })

        return recommendations

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import os

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load users and embeddings
    users = load_all_user_profiles(
        users_root_dir=os.path.join(script_dir, "users"), 
        songs_csv_path=os.path.join(script_dir, "songs.csv")
    )
    if not users:
        print("No user profiles found. Please check your 'users/' folder.")
        exit()

    sim_df, _ = pairwise_user_similarity(users)

    # Build user → track history
    user_track_history = {uname: info["df"]["track_id"].tolist() for uname, info in users.items()}

    # Build track metadata
    track_metadata = {}
    for uname, info in users.items():
        df = info["df"]
        for _, row in df.iterrows():
            track_metadata[row["track_id"]] = {
                "track_name": row.get("track_name", ""),
                "artists": row.get("artists", "")
            }

    # Initialize recommender
    recommender = UserSimilarityRecommender(sim_df.to_dict(), user_track_history, track_metadata)

    # CLI menu
    user_list = list(users.keys())
    print("\nSelect a user to get recommendations:\n")
    for idx, uname in enumerate(user_list, 1):
        print(f"{idx}) {uname}")

    while True:
        try:
            choice = int(input("\nEnter number: ").strip())
            if 1 <= choice <= len(user_list):
                target_user = user_list[choice - 1]
                break
            else:
                print(f"Enter a number between 1 and {len(user_list)}")
        except ValueError:
            print("Invalid input, enter a number.")

    print(f"\nTop 10 recommended tracks for '{target_user}':\n")
    recs = recommender.recommend(target_user=target_user, top_n=10)
    for idx, rec in enumerate(recs, 1):
        print(f"{idx}) {rec['track_name']} – {rec['artists']} (score={rec['score']})")

