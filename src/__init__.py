"""
Movie Recommendation System
A content-based recommender that integrates data from multiple streaming platforms.
"""

__version__ = "1.0.0"
__author__ = "Jun Hyuk Lee"

from .recommender import Recommender
from .feature_engineering import prepare_features
from .data_preprocessing import load_and_integrate_data
from .evaluation import calculate_rmse, calculate_mae

__all__ = [
    'Recommender',
    'prepare_features',
    'load_and_integrate_data',
    'calculate_rmse',
    'calculate_mae'
]
