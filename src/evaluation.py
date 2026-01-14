"""
Evaluation module for recommendation system.
Implements performance metrics (RMSE, MAE) for evaluating recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_rmse(actual_ratings, predicted_ratings):
    """
    Calculate Root Mean Square Error between actual and predicted ratings.
    
    RMSE measures the average magnitude of prediction errors, with larger
    errors weighted more heavily due to squaring.
    
    Args:
        actual_ratings (array-like): Actual user ratings
        predicted_ratings (array-like): Predicted ratings from recommendations
        
    Returns:
        float: RMSE value (lower is better)
    """
    return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))


def calculate_mae(actual_ratings, predicted_ratings):
    """
    Calculate Mean Absolute Error between actual and predicted ratings.
    
    MAE measures the average absolute difference between actual and predicted
    ratings, treating all errors equally.
    
    Args:
        actual_ratings (array-like): Actual user ratings
        predicted_ratings (array-like): Predicted ratings from recommendations
        
    Returns:
        float: MAE value (lower is better)
    """
    return mean_absolute_error(actual_ratings, predicted_ratings)


def find_valid_user_movie_pairs(ratings_df, recommendations_df, target_movie_id):
    """
    Find users who rated both the target movie and all its recommended movies.
    
    This is necessary to evaluate recommendation quality using actual user ratings.
    
    Args:
        ratings_df (pd.DataFrame): User ratings data (userID, movieID, rating)
        recommendations_df (pd.DataFrame): Recommendations (movie_id, recommended_movie_id, similarity_score)
        target_movie_id (str): IMDb ID of the target movie
        
    Returns:
        tuple: (user_id, target_movie_id, recommended_movie_ids) or None if not found
    """
    # Get recommendations for target movie
    recs = recommendations_df[recommendations_df['movie_id'] == target_movie_id]
    
    if len(recs) == 0:
        return None
    
    recommended_movie_ids = recs['recommended_movie_id'].tolist()
    
    # Find users who rated the target movie
    target_ratings = ratings_df[ratings_df['movieID'] == target_movie_id]
    
    # Check each user to see if they rated all recommended movies
    for user_id in target_ratings['userID'].unique():
        user_ratings = ratings_df[ratings_df['userID'] == user_id]
        rated_movies = set(user_ratings['movieID'].values)
        
        # Check if user rated all recommended movies
        if all(rec_id in rated_movies for rec_id in recommended_movie_ids):
            return user_id, target_movie_id, recommended_movie_ids
    
    return None


def evaluate_recommendations(ratings_df, recommendations_df, target_movie_id):
    """
    Evaluate recommendation quality for a specific target movie.
    
    Finds a user who rated both the target and all recommended movies,
    then calculates RMSE and MAE between the target movie rating and
    recommended movie ratings.
    
    Args:
        ratings_df (pd.DataFrame): User ratings data
        recommendations_df (pd.DataFrame): Generated recommendations
        target_movie_id (str): IMDb ID of target movie
        
    Returns:
        dict: Dictionary containing RMSE, MAE, and metadata, or None if no valid user found
    """
    result = find_valid_user_movie_pairs(ratings_df, recommendations_df, target_movie_id)
    
    if result is None:
        return None
    
    user_id, target_id, recommended_ids = result
    
    # Get user's ratings
    user_ratings = ratings_df[ratings_df['userID'] == user_id]
    
    # Get target movie rating
    target_rating = user_ratings[user_ratings['movieID'] == target_id]['rating'].values[0]
    
    # Get recommended movies' ratings
    rec_ratings = []
    for rec_id in recommended_ids:
        rating = user_ratings[user_ratings['movieID'] == rec_id]['rating'].values
        if len(rating) > 0:
            rec_ratings.append(rating[0])
    
    # Calculate metrics
    # Compare target rating to recommended ratings
    predicted = [target_rating] * len(rec_ratings)
    
    rmse = calculate_rmse(rec_ratings, predicted)
    mae = calculate_mae(rec_ratings, predicted)
    
    return {
        'target_movie_id': target_id,
        'user_id': user_id,
        'target_rating': target_rating,
        'recommended_ratings': rec_ratings,
        'rmse': rmse,
        'mae': mae,
        'num_recommendations': len(rec_ratings)
    }


def compare_platforms(ratings_df, platform_recommendations, full_recommendations):
    """
    Compare performance metrics between platform-specific and integrated recommendations.
    
    Args:
        ratings_df (pd.DataFrame): User ratings data
        platform_recommendations (pd.DataFrame): Recommendations from single platform
        full_recommendations (pd.DataFrame): Recommendations from integrated dataset
        
    Returns:
        dict: Comparison results with RMSE and MAE for both systems
    """
    # Get common target movies
    common_movies = set(platform_recommendations['movie_id']) & set(full_recommendations['movie_id'])
    
    results = {
        'platform': {'rmse': [], 'mae': []},
        'integrated': {'rmse': [], 'mae': []}
    }
    
    for movie_id in common_movies:
        # Evaluate platform-specific recommendations
        platform_eval = evaluate_recommendations(ratings_df, platform_recommendations, movie_id)
        if platform_eval is not None:
            results['platform']['rmse'].append(platform_eval['rmse'])
            results['platform']['mae'].append(platform_eval['mae'])
        
        # Evaluate integrated recommendations
        full_eval = evaluate_recommendations(ratings_df, full_recommendations, movie_id)
        if full_eval is not None:
            results['integrated']['rmse'].append(full_eval['rmse'])
            results['integrated']['mae'].append(full_eval['mae'])
    
    # Calculate average metrics
    comparison = {
        'platform': {
            'avg_rmse': np.mean(results['platform']['rmse']) if results['platform']['rmse'] else None,
            'avg_mae': np.mean(results['platform']['mae']) if results['platform']['mae'] else None,
            'num_evaluated': len(results['platform']['rmse'])
        },
        'integrated': {
            'avg_rmse': np.mean(results['integrated']['rmse']) if results['integrated']['rmse'] else None,
            'avg_mae': np.mean(results['integrated']['mae']) if results['integrated']['mae'] else None,
            'num_evaluated': len(results['integrated']['rmse'])
        }
    }
    
    return comparison


def calculate_improvement(platform_metric, integrated_metric):
    """
    Calculate percentage improvement from platform-specific to integrated system.
    
    Args:
        platform_metric (float): Metric value from platform-specific system
        integrated_metric (float): Metric value from integrated system
        
    Returns:
        float: Percentage improvement (negative means degradation)
    """
    if platform_metric == 0:
        return 0
    
    improvement = ((platform_metric - integrated_metric) / platform_metric) * 100
    return improvement


def print_evaluation_summary(comparison_results, platform_name):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        comparison_results (dict): Results from compare_platforms
        platform_name (str): Name of the platform being compared
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {platform_name} vs Integrated")
    print(f"{'='*60}\n")
    
    platform = comparison_results['platform']
    integrated = comparison_results['integrated']
    
    print(f"{platform_name} Dataset:")
    print(f"  RMSE: {platform['avg_rmse']:.4f}")
    print(f"  MAE:  {platform['avg_mae']:.4f}")
    print(f"  Evaluated: {platform['num_evaluated']} movies\n")
    
    print(f"Fully Integrated Dataset:")
    print(f"  RMSE: {integrated['avg_rmse']:.4f}")
    print(f"  MAE:  {integrated['avg_mae']:.4f}")
    print(f"  Evaluated: {integrated['num_evaluated']} movies\n")
    
    if platform['avg_rmse'] and integrated['avg_rmse']:
        rmse_improvement = calculate_improvement(platform['avg_rmse'], integrated['avg_rmse'])
        mae_improvement = calculate_improvement(platform['avg_mae'], integrated['avg_mae'])
        
        print(f"Improvement:")
        print(f"  RMSE: {rmse_improvement:+.2f}%")
        print(f"  MAE:  {mae_improvement:+.2f}%")
    
    print(f"{'='*60}\n")
