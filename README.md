# 🎵 Music Recommendation System

A production-ready **AI-powered music recommendation system** that compares **three different recommendation approaches**—unsupervised learning, supervised learning, and collaborative filtering—using real Spotify data and rigorous evaluation metrics.

Built with **Python, Scikit-learn, XGBoost, and Streamlit**, this project demonstrates how modern recommender systems balance **accuracy, diversity, and user satisfaction**.

---

## 🚀 Project Overview

With music catalogs exceeding **89,000 tracks**, finding songs users actually enjoy is non-trivial. Generic playlists fail because musical taste is deeply personal.

This project answers a key question:

> **Which AI recommendation technique best balances precision, diversity, and user satisfaction?**

To explore this, we implemented and compared:
- **KMeans Clustering** (Unsupervised Learning)
- **XGBoost Gradient Boosting** (Supervised Learning)
- **Collaborative Filtering** (User Similarity Baseline)

All models are integrated into a **professional Streamlit web application** with live comparison, metrics, and insights.

---

## 📊 Dataset

### 🎧 Music Data
- **Source:** Kaggle Spotify Dataset
- **Tracks:** 89,741 songs
- **Audio Features (9):**
  - Danceability
  - Energy
  - Valence (positivity)
  - Acousticness
  - Instrumentalness
  - Speechiness
  - Liveness
  - Loudness
  - Tempo

### 👤 User Data
- Real Spotify listening data from **3 users**
- Includes saved tracks, top tracks, and recently played songs
- ~700 matched tracks per user after preprocessing

---

## 🧠 Recommendation Models

### 1️⃣ KMeans Clustering (Unsupervised)
- Clusters all songs into **6 natural groups** using audio features
- Users select a **mood** (happy, sad, energetic, calm)
- Mood is mapped to a **target audio feature vector**
- Songs ranked by distance to the mood vector (not cluster centroid)
- **Artist diversity constraint** (max 1 song per artist)

**Strengths:** Fast, interpretable, no labeled data required

---

### 2️⃣ XGBoost Gradient Boosting (Supervised)
- Trained using:
  - **Positive samples:** user’s liked tracks
  - **Negative samples:** unheard tracks (3:1 ratio)
- **Feature engineering:** 9 → **23 features**
  - Interaction features (e.g., danceability × energy)
  - **Deviation features** measuring distance from user’s average taste
- Learns **personalized listening patterns**

**Best-performing model in user satisfaction**

---

### 3️⃣ Collaborative Filtering (Baseline)
- Computes **user embeddings** from averaged audio features
- Uses **cosine similarity** to find similar users
- Recommends tracks liked by similar users
- Limited by small user count, included as baseline

---

## 📐 Evaluation Metrics

| Metric | Description |
|------|------------|
| Precision@10 | Accuracy of top-10 recommendations |
| Diversity | Variety in recommendations |
| Novelty | Degree of exploration |
| User Satisfaction | Composite performance score |
| Coverage | Portion of catalog utilized |

---

## 🏆 Results Summary

| Model | Precision@10 | Diversity | User Satisfaction |
|------|------------|----------|------------------|
| KMeans | 0.999 | 1.02 | 0.755 |
| XGBoost | 0.999 | **1.14** | **0.785** |
| Collaborative Filtering | 0.999 | 0.94 | 0.735 |

**Key Insights:**
- All models achieve ~99.9% precision
- XGBoost performs best overall due to advanced feature engineering
- Low novelty is intentional to maximize user satisfaction

---

## 🖥️ System Architecture

- Modular pipeline:
  - Data loading
  - Feature processing
  - Model selection
  - Recommendation generation
  - Evaluation
- **Streamlit GUI** with:
  - Home
  - Generate Recommendations
  - Model Comparison
  - Metrics Visualization
  - Model Insights
- Persistent caching for fast reloads
- Easily extensible architecture

---

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** Streamlit

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

