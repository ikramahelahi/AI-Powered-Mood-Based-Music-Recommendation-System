# model_evaluation.py
"""
Comprehensive Model Evaluation Framework

Evaluates and compares music recommendation models across multiple metrics:
- Precision@K: Accuracy of top-K recommendations
- Diversity: Genre/artist variety in recommendations
- Novelty: How much recommendations differ from user's history
- Coverage: Percentage of catalog recommended across all users
- User Satisfaction Score: Composite metric

Supports all recommendation models:
1. Mood-based KMeans clustering
2. User similarity (collaborative filtering)
3. Track-level similarity (hybrid)
4. Gradient boosting (content-based ML)
"""

import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import recommender modules
from data_loader import FEATURE_COLUMNS
from mood_recommender import recommend_mood_tracks, MOOD_PROFILES
from user_profiles import load_all_user_profiles, pairwise_user_similarity
from recommender import UserSimilarityRecommender

# Script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(script_dir, "users")
SONGS_CSV = os.path.join(script_dir, "songs.csv")
RESULTS_DIR = os.path.join(script_dir, "evaluation_results")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)


class ModelEvaluator:
    """Unified evaluation framework for all recommendation models."""
    
    def __init__(self, users_dir=USERS_DIR, songs_csv=SONGS_CSV):
        """Initialize evaluator with data paths."""
        self.users_dir = users_dir
        self.songs_csv = songs_csv
        self.available_users = self._get_available_users()
        self.kaggle_df = self._load_kaggle_data()
        
        print(f"✓ Evaluator initialized")
        print(f"  Users: {len(self.available_users)}")
        print(f"  Catalog size: {len(self.kaggle_df)}")
    
    def _get_available_users(self):
        """Get list of available users."""
        return [d for d in os.listdir(self.users_dir) 
                if os.path.isdir(os.path.join(self.users_dir, d))]
    
    def _load_kaggle_data(self):
        """Load the full Kaggle dataset."""
        df = pd.read_csv(self.songs_csv)
        df.columns = [c.lower().strip() for c in df.columns]
        return df.dropna(subset=FEATURE_COLUMNS)
    
    def _load_user_tracks(self, user_name):
        """Load a user's listening history (track IDs)."""
        from gradient_boosting_recommender import GradientBoostingRecommender
        
        gbm = GradientBoostingRecommender()
        user_folder = os.path.join(self.users_dir, user_name)
        user_df = gbm._load_user_data(user_folder, self.songs_csv)
        
        if user_df is None or user_df.empty:
            return set(), pd.DataFrame()
        
        track_ids = set(user_df['track_id'].values) if 'track_id' in user_df.columns else set()
        return track_ids, user_df
    
    def precision_at_k(self, recommendations, user_tracks, k=10):
        """
        Calculate Precision@K.
        
        For offline evaluation, we use a proxy:
        - Higher score if recommended tracks are similar to user's liked tracks
        - Based on audio feature similarity
        
        Args:
            recommendations: DataFrame with recommendations
            user_tracks: DataFrame with user's history
            k: Top-K to evaluate
        
        Returns:
            Precision score (0-1)
        """
        if recommendations.empty or user_tracks.empty:
            return 0.0
        
        # Take top K recommendations
        top_k_recs = recommendations.head(k)
        
        # Calculate average similarity to user's tracks
        rec_features = top_k_recs[FEATURE_COLUMNS].values
        user_features = user_tracks[FEATURE_COLUMNS].values
        
        # Compute similarities
        similarities = cosine_similarity(rec_features, user_features)
        
        # For each recommendation, find max similarity to any user track
        max_similarities = similarities.max(axis=1)
        
        # Precision = average of max similarities
        precision = max_similarities.mean()
        
        return precision
    
    def diversity_score(self, recommendations):
        """
        Calculate diversity of recommendations.
        
        Measures:
        - Artist diversity (unique artists / total tracks)
        - Feature diversity (std dev of audio features)
        
        Returns:
            Diversity score (0-1)
        """
        if recommendations.empty:
            return 0.0
        
        # Artist diversity
        if 'artists' in recommendations.columns:
            artists = recommendations['artists'].str.split(';').explode()
            unique_artists = artists.nunique()
            total_tracks = len(recommendations)
            artist_diversity = unique_artists / total_tracks
        else:
            artist_diversity = 0.5
        
        # Feature diversity (normalized std dev)
        feature_std = recommendations[FEATURE_COLUMNS].std().mean()
        feature_diversity = min(feature_std / 0.3, 1.0)  # Normalize by typical std
        
        # Combined diversity
        diversity = (artist_diversity * 0.6 + feature_diversity * 0.4)
        
        return diversity
    
    def novelty_score(self, recommendations, user_tracks):
        """
        Calculate novelty of recommendations.
        
        Novelty = how different recommendations are from user's existing tracks
        Higher score = more novel/exploratory recommendations
        
        Returns:
            Novelty score (0-1)
        """
        if recommendations.empty or user_tracks.empty:
            return 0.0
        
        rec_features = recommendations[FEATURE_COLUMNS].values
        user_features = user_tracks[FEATURE_COLUMNS].values
        
        # Compute similarities
        similarities = cosine_similarity(rec_features, user_features)
        
        # Novelty = 1 - average max similarity
        max_similarities = similarities.max(axis=1)
        novelty = 1.0 - max_similarities.mean()
        
        return max(0.0, novelty)  # Ensure non-negative
    
    def catalog_coverage(self, all_recommendations):
        """
        Calculate catalog coverage across all users.
        
        Args:
            all_recommendations: Dict of {user: recommendations_df}
        
        Returns:
            Coverage percentage (0-100)
        """
        recommended_tracks = set()
        
        for recs in all_recommendations.values():
            if 'track_id' in recs.columns:
                recommended_tracks.update(recs['track_id'].values)
        
        total_catalog = len(self.kaggle_df)
        coverage = (len(recommended_tracks) / total_catalog) * 100
        
        return coverage
    
    def user_satisfaction_score(self, precision, diversity, novelty, 
                                weights={'precision': 0.5, 'diversity': 0.25, 'novelty': 0.25}):
        """
        Calculate composite user satisfaction score.
        
        Weighted combination of precision, diversity, and novelty.
        
        Returns:
            Satisfaction score (0-1)
        """
        score = (weights['precision'] * precision + 
                weights['diversity'] * diversity + 
                weights['novelty'] * novelty)
        return score
    
    def evaluate_model(self, model_name, recommendation_generator, k=10):
        """
        Evaluate a single model across all users.
        
        Args:
            model_name: Name of the model for reporting
            recommendation_generator: Function that takes (user_name) and returns recommendations DataFrame
            k: Top-K for evaluation
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}\n")
        
        results = {
            'model': model_name,
            'users': [],
            'precision_at_k': [],
            'diversity': [],
            'novelty': [],
            'satisfaction': []
        }
        
        all_recommendations = {}
        
        for user_name in self.available_users:
            print(f"  Evaluating for user: {user_name}...", end=' ')
            
            try:
                # Generate recommendations
                recommendations = recommendation_generator(user_name)
                
                if recommendations is None or recommendations.empty:
                    print("No recommendations generated. Skipping.")
                    continue
                
                # Load user tracks for comparison
                user_track_ids, user_df = self._load_user_tracks(user_name)
                
                if user_df.empty:
                    print("No user history. Skipping.")
                    continue
                
                # Calculate metrics
                precision = self.precision_at_k(recommendations, user_df, k=k)
                diversity = self.diversity_score(recommendations)
                novelty = self.novelty_score(recommendations, user_df)
                satisfaction = self.user_satisfaction_score(precision, diversity, novelty)
                
                # Store results
                results['users'].append(user_name)
                results['precision_at_k'].append(precision)
                results['diversity'].append(diversity)
                results['novelty'].append(novelty)
                results['satisfaction'].append(satisfaction)
                
                all_recommendations[user_name] = recommendations
                
                print(f"✓ (P@{k}={precision:.3f}, Div={diversity:.3f}, Nov={novelty:.3f})")
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Calculate coverage
        coverage = self.catalog_coverage(all_recommendations)
        results['coverage'] = coverage
        
        # Calculate averages
        if results['users']:
            results['avg_precision'] = np.mean(results['precision_at_k'])
            results['avg_diversity'] = np.mean(results['diversity'])
            results['avg_novelty'] = np.mean(results['novelty'])
            results['avg_satisfaction'] = np.mean(results['satisfaction'])
        else:
            results['avg_precision'] = 0.0
            results['avg_diversity'] = 0.0
            results['avg_novelty'] = 0.0
            results['avg_satisfaction'] = 0.0
        
        print(f"\n✓ Evaluation complete!")
        print(f"  Avg Precision@{k}: {results['avg_precision']:.3f}")
        print(f"  Avg Diversity:     {results['avg_diversity']:.3f}")
        print(f"  Avg Novelty:       {results['avg_novelty']:.3f}")
        print(f"  Avg Satisfaction:  {results['avg_satisfaction']:.3f}")
        print(f"  Catalog Coverage:  {results['coverage']:.2f}%")
        
        return results
    
    def compare_models(self, model_results_list, save_plots=True):
        """
        Compare multiple models and generate visualization.
        
        Args:
            model_results_list: List of results dictionaries from evaluate_model()
            save_plots: Whether to save plots to disk
        
        Returns:
            Comparison DataFrame
        """
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}\n")
        
        # Create comparison DataFrame
        comparison_data = []
        for results in model_results_list:
            comparison_data.append({
                'Model': results['model'],
                'Precision@10': results['avg_precision'],
                'Diversity': results['avg_diversity'],
                'Novelty': results['avg_novelty'],
                'Satisfaction': results['avg_satisfaction'],
                'Coverage %': results['coverage']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison table
        print(comparison_df.to_string(index=False))
        print()
        
        # Identify best model for each metric
        print("Best Performers:")
        print(f"  Precision:    {comparison_df.loc[comparison_df['Precision@10'].idxmax(), 'Model']}")
        print(f"  Diversity:    {comparison_df.loc[comparison_df['Diversity'].idxmax(), 'Model']}")
        print(f"  Novelty:      {comparison_df.loc[comparison_df['Novelty'].idxmax(), 'Model']}")
        print(f"  Satisfaction: {comparison_df.loc[comparison_df['Satisfaction'].idxmax(), 'Model']}")
        print(f"  Coverage:     {comparison_df.loc[comparison_df['Coverage %'].idxmax(), 'Model']}")
        
        # Generate visualizations
        if save_plots:
            self._plot_model_comparison(comparison_df)
            self._plot_radar_chart(model_results_list)
        
        # Save comparison to CSV
        csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved to: {csv_path}")
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_df):
        """Generate bar chart comparing models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison Across Metrics', fontsize=16, fontweight='bold')
        
        metrics = [
            ('Precision@10', 'Precision@10', axes[0, 0]),
            ('Diversity', 'Diversity Score', axes[0, 1]),
            ('Novelty', 'Novelty Score', axes[1, 0]),
            ('Satisfaction', 'User Satisfaction', axes[1, 1])
        ]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
        
        for metric_col, title, ax in metrics:
            comparison_df.plot(
                x='Model', 
                y=metric_col, 
                kind='bar', 
                ax=ax, 
                legend=False,
                color=colors,
                edgecolor='black',
                linewidth=1.2
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(comparison_df[metric_col]):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, 'model_comparison_bars.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Bar chart saved to: {plot_path}")
        plt.close()
    
    def _plot_radar_chart(self, model_results_list):
        """Generate radar chart for model comparison."""
        from math import pi
        
        # Prepare data
        categories = ['Precision', 'Diversity', 'Novelty', 'Satisfaction']
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, results in enumerate(model_results_list):
            values = [
                results['avg_precision'],
                results['avg_diversity'],
                results['avg_novelty'],
                results['avg_satisfaction']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=results['model'], color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
        
        # Fix axis to go from 0 to 1
        ax.set_ylim(0, 1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plot_path = os.path.join(RESULTS_DIR, 'model_comparison_radar.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Radar chart saved to: {plot_path}")
        plt.close()


def create_recommendation_generators():
    """
    Create recommendation generator functions for each model.
    
    Returns:
        Dictionary of {model_name: generator_function}
    """
    generators = {}
    
    # 1. Mood-based recommender (using "happy" mood as default)
    def mood_generator(user_name):
        """Generate mood-based recommendations."""
        try:
            return recommend_mood_tracks(user_name, mood_name='happy', top_n=10)
        except Exception as e:
            print(f"Error in mood generator: {e}")
            return pd.DataFrame()
    
    generators['Mood-Based (KMeans)'] = mood_generator
    
    # 2. User similarity recommender
    def user_sim_generator(user_name):
        """Generate user similarity recommendations."""
        try:
            # Load all user profiles
            users = load_all_user_profiles(users_root_dir=USERS_DIR, songs_csv_path=SONGS_CSV)
            if not users or user_name not in users:
                return pd.DataFrame()
            
            sim_df, _ = pairwise_user_similarity(users)
            
            # Build user track history
            user_track_history = {u: info["df"]["track_id"].tolist() for u, info in users.items()}
            
            # Build track metadata
            track_metadata = {}
            for u, info in users.items():
                df = info["df"]
                for _, row in df.iterrows():
                    track_metadata[row["track_id"]] = {
                        "track_name": row.get("track_name", ""),
                        "artists": row.get("artists", "")
                    }
            
            # Initialize recommender
            recommender = UserSimilarityRecommender(sim_df.to_dict(), user_track_history, track_metadata)
            
            # Get recommendations
            recs = recommender.recommend(target_user=user_name, top_n=10)
            
            # Convert to DataFrame format
            if recs:
                rec_df = pd.DataFrame(recs)
                # Add audio features from Kaggle dataset
                kaggle_df = pd.read_csv(SONGS_CSV)
                kaggle_df.columns = [c.lower().strip() for c in kaggle_df.columns]
                rec_df = rec_df.merge(kaggle_df[['track_id'] + FEATURE_COLUMNS], on='track_id', how='left')
                return rec_df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error in user similarity generator: {e}")
            return pd.DataFrame()
    
    generators['User Similarity (Collaborative)'] = user_sim_generator
    
    # 3. Gradient Boosting recommender
    def gbm_generator(user_name):
        """Generate gradient boosting recommendations."""
        try:
            from gradient_boosting_recommender import GradientBoostingRecommender
            
            user_folder = os.path.join(USERS_DIR, user_name)
            recommender = GradientBoostingRecommender(user_name=user_name)
            
            return recommender.recommend(user_folder, SONGS_CSV, top_n=10)
            
        except Exception as e:
            print(f"Error in GBM generator: {e}")
            return pd.DataFrame()
    
    generators['Gradient Boosting (XGBoost)'] = gbm_generator
    
    return generators


def main():
    """Run comprehensive model evaluation."""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Create recommendation generators
    print("\n✓ Setting up models...")
    generators = create_recommendation_generators()
    
    # Evaluate each model
    all_results = []
    for model_name, generator in generators.items():
        results = evaluator.evaluate_model(model_name, generator, k=10)
        all_results.append(results)
    
    # Compare models
    print("\n" + "="*60)
    comparison_df = evaluator.compare_models(all_results, save_plots=True)
    
    print("\n✓ Evaluation complete! Check the 'evaluation_results' folder for plots.")


if __name__ == "__main__":
    main()
