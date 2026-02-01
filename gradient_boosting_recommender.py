# gradient_boosting_recommender.py
"""
Gradient Boosting Music Recommendation Engine

Uses XGBoost/LightGBM to learn user preferences from listening history.
Trains on positive examples (user's tracks) vs negative samples (random tracks).

Key Features:
- Feature engineering: audio features + derived features
- Positive/negative sampling with class weighting
- Model persistence for faster inference
- Probability-based ranking for recommendations
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_loader import FEATURE_COLUMNS, load_user_history_from_folder
import warnings
warnings.filterwarnings('ignore')

# Try to import both XGBoost and LightGBM, use whichever is available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

if not HAS_XGBOOST and not HAS_LIGHTGBM:
    raise ImportError("Please install either xgboost or lightgbm: pip install xgboost lightgbm")

# Get script directory for file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(script_dir, "users")
SONGS_CSV = os.path.join(script_dir, "songs.csv")
MODELS_DIR = os.path.join(script_dir, "models")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Configuration
DEFAULT_MODEL_TYPE = 'xgboost' if HAS_XGBOOST else 'lightgbm'
NEGATIVE_SAMPLE_RATIO = 3  # 3 negative samples per positive sample
SOURCE_WEIGHTS = {
    'liked': 3.0,
    'top': 2.0,
    'recent': 1.0
}


class GradientBoostingRecommender:
    """Gradient Boosting-based Music Recommender"""
    
    def __init__(self, model_type=DEFAULT_MODEL_TYPE, user_name=None):
        """
        Initialize the recommender.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            user_name: Username for personalized model
        """
        self.model_type = model_type
        self.user_name = user_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.trained = False
        
        # Validate model type
        if model_type == 'xgboost' and not HAS_XGBOOST:
            raise ValueError("XGBoost not installed. Use 'lightgbm' or install xgboost.")
        if model_type == 'lightgbm' and not HAS_LIGHTGBM:
            raise ValueError("LightGBM not installed. Use 'xgboost' or install lightgbm.")
    
    def engineer_features(self, df, user_mean_features=None):
        """
        Engineer additional features from audio features.
        
        Args:
            df: DataFrame with audio features
            user_mean_features: Mean feature vector from user's history (for deviation features)
        
        Returns:
            DataFrame with engineered features
        """
        features_df = df[FEATURE_COLUMNS].copy()
        
        # Derived features: ratios and interactions
        features_df['energy_valence_ratio'] = features_df['energy'] / (features_df['valence'] + 0.01)
        features_df['danceability_energy_product'] = features_df['danceability'] * features_df['energy']
        features_df['acoustic_instrumental_ratio'] = features_df['acousticness'] / (features_df['instrumentalness'] + 0.01)
        features_df['vocal_intensity'] = features_df['speechiness'] * features_df['loudness']
        features_df['tempo_energy_product'] = features_df['tempo'] * features_df['energy'] / 100
        
        # If user mean features provided, add deviation features
        if user_mean_features is not None:
            for feat in FEATURE_COLUMNS:
                features_df[f'{feat}_deviation'] = features_df[feat] - user_mean_features[feat]
        
        return features_df
    
    def load_and_prepare_data(self, user_folder, songs_csv_path):
        """
        Load user data and prepare training dataset with positive/negative sampling.
        
        Returns:
            X_train, X_val, y_train, y_val, user_df, kaggle_df
        """
        print(f"✓ Loading data for user: {os.path.basename(user_folder)}")
        
        # Load user listening history
        user_df = self._load_user_data(user_folder, songs_csv_path)
        if user_df is None or user_df.empty:
            raise ValueError(f"No user data found in {user_folder}")
        
        # Load full Kaggle dataset
        kaggle_df = pd.read_csv(songs_csv_path)
        kaggle_df.columns = [c.lower().strip() for c in kaggle_df.columns]
        kaggle_df = kaggle_df.dropna(subset=FEATURE_COLUMNS)
        
        print(f"  User tracks: {len(user_df)}")
        print(f"  Total tracks in catalog: {len(kaggle_df)}")
        
        # Calculate user mean features for deviation features
        user_mean_features = user_df[FEATURE_COLUMNS].mean()
        
        # Prepare positive samples (user's tracks)
        user_features = self.engineer_features(user_df, user_mean_features)
        user_labels = user_df['source_weight'].values  # Use source weights as labels (1-3 range)
        
        # Normalize labels to 0-1 range for better training
        user_labels = (user_labels - 1) / 2  # Maps [1, 2, 3] -> [0, 0.5, 1]
        
        # Prepare negative samples (random tracks not in user's library)
        user_track_ids = set(user_df['track_id'].values) if 'track_id' in user_df.columns else set()
        negative_pool = kaggle_df[~kaggle_df['track_id'].isin(user_track_ids)] if user_track_ids else kaggle_df
        
        n_negative = min(len(user_df) * NEGATIVE_SAMPLE_RATIO, len(negative_pool))
        negative_samples = negative_pool.sample(n=n_negative, random_state=42)
        
        negative_features = self.engineer_features(negative_samples, user_mean_features)
        negative_labels = np.zeros(len(negative_samples))  # Label 0 for negative samples
        
        print(f"  Positive samples: {len(user_features)}")
        print(f"  Negative samples: {len(negative_features)}")
        
        # Combine positive and negative samples
        X = pd.concat([user_features, negative_features], ignore_index=True)
        y = np.concatenate([user_labels, negative_labels])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=(y > 0).astype(int)
        )
        
        print(f"  Train set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val, user_df, kaggle_df, user_mean_features
    
    def _load_user_data(self, user_folder, songs_csv_path):
        """Load user data with source weighting."""
        user_tracks = []
        
        # Load liked tracks
        liked_path = os.path.join(user_folder, "spotify_saved_tracks.csv")
        if os.path.exists(liked_path):
            liked_df = pd.read_csv(liked_path)
            liked_df.columns = [c.lower().strip() for c in liked_df.columns]
            liked_df['source'] = 'liked'
            liked_df['source_weight'] = SOURCE_WEIGHTS['liked']
            user_tracks.append(liked_df)
        
        # Load top tracks
        top_path = os.path.join(user_folder, "spotify_top_tracks.csv")
        if os.path.exists(top_path):
            top_df = pd.read_csv(top_path)
            top_df.columns = [c.lower().strip() for c in top_df.columns]
            top_df['source'] = 'top'
            top_df['source_weight'] = SOURCE_WEIGHTS['top']
            user_tracks.append(top_df)
        
        # Load recent tracks
        recent_path = os.path.join(user_folder, "spotify_recently_played.csv")
        if os.path.exists(recent_path):
            recent_df = pd.read_csv(recent_path)
            recent_df.columns = [c.lower().strip() for c in recent_df.columns]
            recent_df['source'] = 'recent'
            recent_df['source_weight'] = SOURCE_WEIGHTS['recent']
            user_tracks.append(recent_df)
        
        if not user_tracks:
            return None
        
        # Combine and deduplicate
        combined = pd.concat(user_tracks, ignore_index=True)
        combined = combined.drop_duplicates(subset=['track_id'] if 'track_id' in combined.columns else ['track_name', 'artists'])
        
        # Match to Kaggle features
        kaggle_df = pd.read_csv(songs_csv_path)
        kaggle_df.columns = [c.lower().strip() for c in kaggle_df.columns]
        
        # Merge on track_id if available
        if 'track_id' in combined.columns and 'track_id' in kaggle_df.columns:
            merged = combined.merge(
                kaggle_df[['track_id'] + FEATURE_COLUMNS],
                on='track_id',
                how='inner'
            )
        else:
            # Fallback to name matching (simplified)
            merged = combined.merge(
                kaggle_df[['track_name', 'artists'] + FEATURE_COLUMNS],
                on=['track_name', 'artists'],
                how='inner'
            )
        
        return merged
    
    def train(self, user_folder, songs_csv_path=SONGS_CSV, **model_params):
        """
        Train the gradient boosting model for a specific user.
        
        Args:
            user_folder: Path to user's data folder
            songs_csv_path: Path to songs.csv
            **model_params: Additional parameters for the model
        """
        print(f"\n{'='*60}")
        print(f"TRAINING GRADIENT BOOSTING MODEL ({self.model_type.upper()})")
        print(f"{'='*60}\n")
        
        # Load and prepare data
        X_train, X_val, y_train, y_val, user_df, kaggle_df, user_mean_features = \
            self.load_and_prepare_data(user_folder, songs_csv_path)
        
        # Store user mean features for later use
        self.user_mean_features = user_mean_features
        
        print("\n✓ Training model...")
        
        # Train based on model type
        if self.model_type == 'xgboost':
            self._train_xgboost(X_train, X_val, y_train, y_val, **model_params)
        else:
            self._train_lightgbm(X_train, X_val, y_train, y_val, **model_params)
        
        self.trained = True
        print(f"\n✓ Model trained successfully!")
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_binary = (y_val > 0).astype(int)
        pred_binary = (val_pred > 0.3).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        accuracy = accuracy_score(val_binary, pred_binary)
        precision = precision_score(val_binary, pred_binary, zero_division=0)
        recall = recall_score(val_binary, pred_binary, zero_division=0)
        
        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
    
    def _train_xgboost(self, X_train, X_val, y_train, y_val, **params):
        """Train XGBoost model."""
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'random_state': 42,
            'eval_metric': 'rmse'
        }
        default_params.update(params)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    def _train_lightgbm(self, X_train, X_val, y_train, y_val, **params):
        """Train LightGBM model."""
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1
        }
        default_params.update(params)
        
        self.model = lgb.LGBMRegressor(**default_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
    
    def recommend(self, user_folder, songs_csv_path=SONGS_CSV, top_n=10, retrain=False):
        """
        Generate recommendations for a user.
        
        Args:
            user_folder: Path to user's data folder
            songs_csv_path: Path to songs.csv
            top_n: Number of recommendations
            retrain: Whether to retrain the model
        
        Returns:
            DataFrame with recommendations
        """
        # Train if not already trained or if retrain requested
        if not self.trained or retrain:
            self.train(user_folder, songs_csv_path)
        
        print(f"\n✓ Generating recommendations...")
        
        # Load user's existing tracks
        user_df = self._load_user_data(user_folder, songs_csv_path)
        user_track_ids = set(user_df['track_id'].values) if 'track_id' in user_df.columns else set()
        
        # Load all candidate tracks
        kaggle_df = pd.read_csv(songs_csv_path)
        kaggle_df.columns = [c.lower().strip() for c in kaggle_df.columns]
        kaggle_df = kaggle_df.dropna(subset=FEATURE_COLUMNS)
        
        # Filter out already-heard tracks
        candidates = kaggle_df[~kaggle_df['track_id'].isin(user_track_ids)].copy()
        print(f"  Evaluating {len(candidates)} candidate tracks...")
        
        # Engineer features for candidates
        candidate_features = self.engineer_features(candidates, self.user_mean_features)
        
        # Scale features
        candidate_features_scaled = self.scaler.transform(candidate_features)
        
        # Predict scores
        predictions = self.model.predict(candidate_features_scaled)
        candidates['predicted_score'] = predictions
        
        # Sort by predicted score
        candidates = candidates.sort_values('predicted_score', ascending=False)
        
        # Remove duplicates by (track_name, artists) to ensure unique recommendations
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
        
        print(f"✓ Generated {len(recommendations)} unique recommendations (1 per artist)\n")
        
        return recommendations
    
    def save_model(self, filepath=None):
        """Save trained model to disk."""
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f"gbm_{self.user_name}_{self.model_type}.pkl")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'user_mean_features': self.user_mean_features,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.user_mean_features = model_data['user_mean_features']
        self.model_type = model_data['model_type']
        self.trained = True
        
        print(f"✓ Model loaded from: {filepath}")
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features."""
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        else:  # lightgbm
            importance = self.model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feature_importance


def main():
    """CLI for testing the gradient boosting recommender."""
    print("\n" + "="*60)
    print("GRADIENT BOOSTING MUSIC RECOMMENDER")
    print("="*60)
    
    # Select user
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
    
    # Select model type
    print(f"\nAvailable models:")
    if HAS_XGBOOST:
        print("1) XGBoost")
    if HAS_LIGHTGBM:
        print("2) LightGBM")
    
    model_type = DEFAULT_MODEL_TYPE
    if HAS_XGBOOST and HAS_LIGHTGBM:
        try:
            model_choice = int(input("\nSelect model type (enter number): ").strip())
            model_type = 'xgboost' if model_choice == 1 else 'lightgbm'
        except ValueError:
            print(f"Using default: {model_type}")
    
    # Create and train recommender
    user_folder = os.path.join(USERS_DIR, target_user)
    recommender = GradientBoostingRecommender(model_type=model_type, user_name=target_user)
    
    # Train and recommend
    recommendations = recommender.recommend(user_folder, SONGS_CSV, top_n=10)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"TOP 10 RECOMMENDATIONS FOR '{target_user}' ({model_type.upper()})")
    print(f"{'='*60}\n")
    
    for idx, row in enumerate(recommendations.itertuples(), 1):
        track_name = row.track_name
        artists = row.artists
        score = row.predicted_score
        
        print(f"{idx}. {track_name}")
        print(f"   Artist: {artists}")
        print(f"   Predicted Score: {score:.3f}")
        print()
    
    # Show feature importance
    print(f"\n{'='*60}")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print(f"{'='*60}\n")
    
    feature_importance = recommender.get_feature_importance(top_n=10)
    for idx, row in enumerate(feature_importance.itertuples(), 1):
        print(f"{idx}. {row.feature}: {row.importance:.4f}")
    
    # Ask if user wants to save the model
    save_choice = input("\n\nSave this model? (y/n): ").strip().lower()
    if save_choice == 'y':
        recommender.save_model()


if __name__ == "__main__":
    main()
