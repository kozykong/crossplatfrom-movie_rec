"""
Feature engineering module for movie recommendation system.
Handles vectorization and transformation of movie features.
"""

import string
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select specific columns from a DataFrame.
    """
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]


class ListToStringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert list-type columns to space-separated strings.
    Used for multi-categorical features like genres and production companies.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.applymap(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
        elif isinstance(X, pd.Series):
            return X.apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
        return X


def custom_tokenizer(text):
    """
    Custom tokenizer that handles punctuation flexibly for movie titles.
    Preserves acronyms, dashes, and unconventional punctuation in movie titles.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        list: List of tokens
    """
    # Remove most punctuation but keep hyphens and apostrophes
    translator = str.maketrans('', '', string.punctuation.replace('-', '').replace("'", ''))
    text = text.translate(translator)
    return text.lower().split()


def build_numerical_pipeline():
    """
    Build pipeline for numerical features (release_date, runtime).
    Uses StandardScaler to normalize values.
    
    Returns:
        Pipeline: Numerical feature pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler())
    ])


def build_single_categorical_pipeline():
    """
    Build pipeline for single-categorical features (belongs_to_collection, original_language).
    Uses OneHotEncoder with frequency filtering to reduce dimensionality.
    
    Returns:
        Pipeline: Single-categorical feature pipeline
    """
    return Pipeline([
        ('encoder', OneHotEncoder(
            min_frequency=0.01,  # Drop categories appearing in <1% of documents
            sparse_output=True,
            handle_unknown='ignore'
        ))
    ])


def build_multi_categorical_pipeline():
    """
    Build pipeline for multi-categorical features (genres, production_companies, spoken_languages).
    Uses CountVectorizer to handle multiple categories per entry.
    
    Returns:
        Pipeline: Multi-categorical feature pipeline
    """
    return Pipeline([
        ('list_to_string', ListToStringTransformer()),
        ('vectorizer', CountVectorizer(
            lowercase=False,
            token_pattern=r'(?u)\b\w+\b'
        ))
    ])


def build_text_pipeline():
    """
    Build pipeline for text features (overview, title).
    Uses TF-IDF vectorization with frequency constraints to manage dimensionality.
    
    Returns:
        Pipeline: Text feature pipeline
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_df=0.8,  # Filter words appearing in >80% of documents
            min_df=0.025,  # Filter words appearing in <2.5% of documents
            tokenizer=custom_tokenizer,
            lowercase=True,
            stop_words=None  # Custom tokenizer handles this
        ))
    ])


def build_feature_union():
    """
    Build complete feature engineering pipeline using FeatureUnion.
    Combines all feature types into a single vectorized representation.
    
    Returns:
        FeatureUnion: Complete feature engineering pipeline
    """
    return FeatureUnion([
        # Numerical features
        ('numerical', Pipeline([
            ('select', ColumnSelector(['release_date', 'runtime'])),
            ('scale', build_numerical_pipeline())
        ])),
        
        # Single-categorical features
        ('collection', Pipeline([
            ('select', ColumnSelector(['belongs_to_collection'])),
            ('encode', build_single_categorical_pipeline())
        ])),
        
        ('language', Pipeline([
            ('select', ColumnSelector(['original_language'])),
            ('encode', build_single_categorical_pipeline())
        ])),
        
        # Multi-categorical features
        ('genres', Pipeline([
            ('select', ColumnSelector(['genres'])),
            ('vectorize', build_multi_categorical_pipeline())
        ])),
        
        ('production_companies', Pipeline([
            ('select', ColumnSelector(['production_companies'])),
            ('vectorize', build_multi_categorical_pipeline())
        ])),
        
        ('spoken_languages', Pipeline([
            ('select', ColumnSelector(['spoken_languages'])),
            ('vectorize', build_multi_categorical_pipeline())
        ])),
        
        ('production_countries', Pipeline([
            ('select', ColumnSelector(['production_countries'])),
            ('vectorize', build_multi_categorical_pipeline())
        ])),
        
        # Text features
        ('overview', Pipeline([
            ('select', ColumnSelector(['overview'])),
            ('flatten', FunctionTransformer(lambda x: x.values.ravel(), validate=False)),
            ('tfidf', build_text_pipeline().named_steps['tfidf'])
        ])),
        
        ('title', Pipeline([
            ('select', ColumnSelector(['title'])),
            ('flatten', FunctionTransformer(lambda x: x.values.ravel(), validate=False)),
            ('tfidf', build_text_pipeline().named_steps['tfidf'])
        ]))
    ])


def prepare_features(df):
    """
    Apply feature engineering pipeline to a dataframe.
    
    Args:
        df (pd.DataFrame): Preprocessed movie dataset
        
    Returns:
        tuple: (transformed_features, fitted_pipeline, original_index)
            - transformed_features: Vectorized feature matrix
            - fitted_pipeline: Fitted FeatureUnion pipeline
            - original_index: IMDb IDs for indexing
    """
    # Store IMDb IDs as index
    imdb_ids = df['imdb_id'].values
    
    # Build and fit pipeline
    pipeline = build_feature_union()
    transformed = pipeline.fit_transform(df)
    
    return transformed, pipeline, imdb_ids


def wrap_df(transformed_array, original_df):
    """
    Wrap transformed features back into a DataFrame with IMDb ID index.
    
    Args:
        transformed_array: Transformed feature array
        original_df (pd.DataFrame): Original dataframe with IMDb IDs
        
    Returns:
        pd.DataFrame: Features as DataFrame with IMDb ID index
    """
    # Convert sparse matrix to dense if needed
    if hasattr(transformed_array, 'toarray'):
        transformed_array = transformed_array.toarray()
    
    # Create DataFrame with IMDb ID as index
    feature_df = pd.DataFrame(
        transformed_array,
        index=original_df['imdb_id'].values
    )
    
    return feature_df
