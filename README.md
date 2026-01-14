# Cross-Platform Movie Recommendation System

A unified movie recommendation system that integrates data from Netflix, Hulu, Prime Video, and Disney+ to overcome limitations of service-segregated recommendations.

## Problem

Each streaming platform operates its own recommendation system based only on its available content. This segregation limits recommendation quality since users' preferences often span across multiple platforms.

## Solution

Built a content-based recommender using cosine similarity across integrated datasets from four major streaming services. The system analyzes movie metadata including genres, production details, runtime, and descriptions to find similar movies regardless of platform.

## Results

Integrated recommendations significantly outperformed platform-specific ones:

- **Disney+**: 20% RMSE improvement (2.02 → 1.60)
- **Hulu**: 44% RMSE improvement (2.10 → 1.18) 
- **Prime Video**: 29% RMSE improvement (3.48 → 2.48)
- **Netflix**: Comparable performance (already strong baseline)

## Technical Approach

**Data Processing**
- Cleaned and integrated 45,466 movies from The Movies Dataset
- Mapped 9,515 entries across streaming platforms
- Validated with 4.6M user ratings from IMDb

**Feature Engineering**
- StandardScaler for numerical features (release date, runtime)
- OneHotEncoder for categorical features (language, collections)
- TF-IDF vectorization for text features (overview, title)
- CountVectorizer for multi-categorical features (genres, production companies)

**Model**
- Pairwise cosine similarity matrix
- Top-10 recommendations per query
- Evaluated using RMSE and MAE metrics

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (feature extraction, similarity metrics)
- Jupyter Notebook

## Usage
```bash
pip install -r requirements.txt
jupyter notebook model.ipynb
```

## Project Structure
```
├── model.ipynb          # Main implementation
├── Final_Report.pdf     # Detailed methodology and results
└── requirements.txt     # Dependencies
```

## Course Information

HKUST MSBD 5001 - Foundations of Data Analytics (2024)

scikit-learn
scipy
jupyter
