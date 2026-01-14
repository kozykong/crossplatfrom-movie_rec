"""
Recommender system module.
Implements content-based movie recommendation using cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity


class Recommender(BaseEstimator):
    """
    Content-based movie recommender using pairwise similarity.
    
    Compares movies using cosine similarity on their feature vectors
    and returns the most similar movies as recommendations.
    
    Attributes:
        similarity_metric (callable): Function to compute pairwise similarity
        num_recommendations (int): Number of recommendations to return
        kernel (pd.DataFrame): Pairwise similarity matrix
    """
    
    def __init__(self, similarity_metric=cosine_similarity, num_recommendations=10):
        """
        Initialize the recommender system.
        
        Args:
            similarity_metric (callable): Function to compute pairwise similarity.
                Default is cosine_similarity.
            num_recommendations (int): Number of movies to recommend.
                Default is 10.
        """
        self.similarity_metric = similarity_metric
        self.num_recommendations = num_recommendations
        self.kernel = None
    
    def fit(self, X, y=None):
        """
        Compute pairwise similarity matrix for all movies.
        
        Args:
            X (pd.DataFrame or array-like): Feature matrix with movie features.
                If DataFrame, index should be IMDb IDs.
            y: Ignored, present for sklearn compatibility
            
        Returns:
            self: Fitted recommender instance
        """
        # Get index (IMDb IDs) if available
        if isinstance(X, pd.DataFrame):
            index = X.index
        else:
            index = range(len(X))
        
        # Compute pairwise similarity matrix
        similarity_matrix = self.similarity_metric(X, X)
        
        # Store as DataFrame for easy lookup
        self.kernel = pd.DataFrame(
            similarity_matrix,
            index=index,
            columns=index
        )
        
        return self
    
    def predict(self, movie_id):
        """
        Get top-N recommendations for a given movie.
        
        Args:
            movie_id (str): IMDb ID of the target movie
            
        Returns:
            list: List of tuples (movie_id, similarity_score) for top recommendations
        """
        if self.kernel is None:
            raise ValueError("Recommender must be fitted before making predictions")
        
        if movie_id not in self.kernel.index:
            raise ValueError(f"Movie {movie_id} not found in the dataset")
        
        # Get similarity scores for the target movie
        sim_scores = self.kernel.loc[movie_id]
        
        # Sort by similarity (descending) and exclude the movie itself
        sim_scores = sim_scores.sort_values(ascending=False)
        top_similar = sim_scores.iloc[1:self.num_recommendations + 1]
        
        # Return as list of (movie_id, score) tuples
        return list(top_similar.items())
    
    def get_recommendations_batch(self, movie_ids):
        """
        Get recommendations for multiple movies at once.
        
        Args:
            movie_ids (list): List of IMDb IDs
            
        Returns:
            dict: Dictionary mapping each movie_id to its recommendations
        """
        recommendations = {}
        for movie_id in movie_ids:
            try:
                recommendations[movie_id] = self.predict(movie_id)
            except ValueError:
                recommendations[movie_id] = []
        
        return recommendations
    
    def get_similarity_score(self, movie_id1, movie_id2):
        """
        Get similarity score between two specific movies.
        
        Args:
            movie_id1 (str): IMDb ID of first movie
            movie_id2 (str): IMDb ID of second movie
            
        Returns:
            float: Similarity score between the two movies
        """
        if self.kernel is None:
            raise ValueError("Recommender must be fitted before computing similarity")
        
        return self.kernel.loc[movie_id1, movie_id2]
    
    def get_most_similar_movies(self, movie_id, threshold=0.5):
        """
        Get all movies above a similarity threshold.
        
        Args:
            movie_id (str): IMDb ID of the target movie
            threshold (float): Minimum similarity score (0-1)
            
        Returns:
            list: List of tuples (movie_id, similarity_score) above threshold
        """
        if self.kernel is None:
            raise ValueError("Recommender must be fitted before making predictions")
        
        sim_scores = self.kernel.loc[movie_id]
        similar_movies = sim_scores[sim_scores >= threshold].sort_values(ascending=False)
        
        # Exclude the movie itself
        similar_movies = similar_movies[similar_movies.index != movie_id]
        
        return list(similar_movies.items())
    
    def save_recommendations(self, output_path):
        """
        Save all pairwise recommendations to a CSV file.
        
        Args:
            output_path (str): Path to save the recommendations CSV
        """
        if self.kernel is None:
            raise ValueError("Recommender must be fitted before saving")
        
        recommendations = []
        
        for movie_id in self.kernel.index:
            recs = self.predict(movie_id)
            for rec_id, score in recs:
                recommendations.append({
                    'movie_id': movie_id,
                    'recommended_movie_id': rec_id,
                    'similarity_score': score
                })
        
        df = pd.DataFrame(recommendations)
        df.to_csv(output_path, index=False)
        print(f"Recommendations saved to {output_path}")


def build_recommender(feature_df, similarity_metric=cosine_similarity, num_recommendations=10):
    """
    Convenience function to build and fit a recommender in one step.
    
    Args:
        feature_df (pd.DataFrame): DataFrame with movie features (IMDb ID as index)
        similarity_metric (callable): Similarity function to use
        num_recommendations (int): Number of recommendations per movie
        
    Returns:
        Recommender: Fitted recommender instance
    """
    recommender = Recommender(
        similarity_metric=similarity_metric,
        num_recommendations=num_recommendations
    )
    recommender.fit(feature_df)
    
    return recommender
