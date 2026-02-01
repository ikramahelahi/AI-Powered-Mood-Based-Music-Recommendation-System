"""
Quick demo script for Phase 1:
- Loads user profiles from ./users/
- Computes embeddings and similarity matrix
- Optionally saves embeddings to disk
"""

import argparse
from user_profiles import load_all_user_profiles, pairwise_user_similarity, save_embeddings
import numpy as np
import pandas as pd

def main(args):
    users = load_all_user_profiles(
        users_root_dir=args.users_dir,
        top_n=args.top_n,
        language_filter=None,
        songs_csv_path=args.songs_csv,
        weight_by_recency=args.weighted
    )
    if not users:
        print("No user profiles found. Place user folders under", args.users_dir)
        return

    sim_df, raw = pairwise_user_similarity(users)
    print("\nPairwise user similarity (cosine):")
    print(sim_df.round(3).to_string())

    if args.save:
        save_embeddings(users, out_path=args.out_json)
        print("Embeddings saved to", args.out_json)

    # Print summary per user
    print("\nUser summary:")
    for u, info in users.items():
        print(f"- {u}: {info['num_tracks']} tracks, embedding mean (first 3 dims): {np.round(info['embedding'][:3], 3)}")

if __name__ == "__main__":
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--users-dir", default=os.path.join(script_dir, "users"), help="Root directory containing user subfolders")
    parser.add_argument("--songs-csv", default=os.path.join(script_dir, "songs.csv"), help="Path to Kaggle songs.csv with features")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top tracks to use for embedding")
    parser.add_argument("--save", action="store_true", help="Save embeddings to JSON")
    parser.add_argument("--out-json", default="user_embeddings.json", help="Where to save embeddings if --save used")
    parser.add_argument("--weighted", action="store_true", help="Weight by recency if played_at exists")
    args = parser.parse_args()
    main(args)
