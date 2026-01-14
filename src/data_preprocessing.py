"""
Data preprocessing module for movie recommendation system.
Handles data loading, cleaning, transformation, and integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_movies_dataset(filepath):
    """
    Load The Movies Dataset with initial preprocessing.
    
    Args:
        filepath (str): Path to the movies CSV file
        
    Returns:
        pd.DataFrame: Loaded movies dataset
    """
    df = pd.read_csv(filepath)
    return df


def load_streaming_platforms_dataset(filepath):
    """
    Load The Streaming Platforms Dataset.
    
    Args:
        filepath (str): Path to the streaming platforms CSV file
        
    Returns:
        pd.DataFrame: Loaded streaming platforms dataset
    """
    df = pd.read_csv(filepath)
    return df


def select_relevant_features(df):
    """
    Select only relevant features from The Movies Dataset.
    Drops redundant identifiers, unreliable financial data, and unnecessary fields.
    
    Args:
        df (pd.DataFrame): Raw movies dataset
        
    Returns:
        pd.DataFrame: Dataset with selected features
    """
    relevant_columns = [
        'imdb_id',
        'title',
        'belongs_to_collection',
        'genres',
        'original_language',
        'overview',
        'production_countries',
        'production_companies',
        'release_date',
        'runtime',
        'spoken_languages'
    ]
    
    return df[relevant_columns].copy()


def clean_missing_values(df):
    """
    Handle missing values in the dataset.
    - Drops entries with missing IMDb IDs that couldn't be recovered
    - Drops entries with whitespace-only overviews
    - Fills missing runtime values (should be done via web scraping in production)
    
    Args:
        df (pd.DataFrame): Dataset with missing values
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Drop rows with missing IMDb IDs
    df = df.dropna(subset=['imdb_id'])
    
    # Drop rows where overview is only whitespace
    df = df[df['overview'].str.strip().astype(bool)]
    
    # Drop rows with missing runtime (in production, scrape from IMDb)
    df = df.dropna(subset=['runtime'])
    
    return df


def parse_json_field(json_str, key='name'):
    """
    Parse JSON-formatted string fields and extract specified key values.
    
    Args:
        json_str (str): JSON string
        key (str): Key to extract from JSON objects
        
    Returns:
        list: List of extracted values
    """
    if pd.isna(json_str):
        return []
    
    try:
        import json
        parsed = json.loads(json_str.replace("'", '"'))
        return [item[key] for item in parsed]
    except:
        return []


def transform_json_fields(df):
    """
    Transform JSON-formatted fields into list format.
    Handles: genres, production_companies, production_countries, spoken_languages
    
    Args:
        df (pd.DataFrame): Dataset with JSON fields
        
    Returns:
        pd.DataFrame: Dataset with transformed fields
    """
    # Transform genres
    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(lambda x: parse_json_field(x, 'id'))
    
    # Transform production companies
    if 'production_companies' in df.columns:
        df['production_companies'] = df['production_companies'].apply(
            lambda x: parse_json_field(x, 'name')
        )
    
    # Transform production countries (convert to ISO codes)
    if 'production_countries' in df.columns:
        df['production_countries'] = df['production_countries'].apply(
            lambda x: parse_json_field(x, 'iso_3166_1')
        )
    
    # Transform spoken languages
    if 'spoken_languages' in df.columns:
        df['spoken_languages'] = df['spoken_languages'].apply(
            lambda x: parse_json_field(x, 'iso_639_1')
        )
    
    # Transform collection
    if 'belongs_to_collection' in df.columns:
        df['belongs_to_collection'] = df['belongs_to_collection'].apply(
            lambda x: parse_json_field(x, 'id')
        )
        df['belongs_to_collection'] = df['belongs_to_collection'].apply(
            lambda x: x[0] if len(x) > 0 else np.nan
        )
    
    return df


def convert_date_to_seconds(date_str):
    """
    Convert date string to seconds since epoch.
    
    Args:
        date_str (str): Date string in YYYY-MM-DD format
        
    Returns:
        float: Seconds since epoch
    """
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.timestamp()
    except:
        return np.nan


def preprocess_dates(df):
    """
    Convert release_date to seconds for numerical processing.
    
    Args:
        df (pd.DataFrame): Dataset with date fields
        
    Returns:
        pd.DataFrame: Dataset with converted dates
    """
    if 'release_date' in df.columns:
        df['release_date'] = df['release_date'].apply(convert_date_to_seconds)
    
    return df


def integrate_streaming_data(movies_df, streaming_df):
    """
    Integrate streaming platform availability data with movies dataset.
    Uses IMDb ID as the linking key.
    
    Args:
        movies_df (pd.DataFrame): Preprocessed movies dataset
        streaming_df (pd.DataFrame): Streaming platforms dataset
        
    Returns:
        pd.DataFrame: Integrated dataset with platform availability columns
    """
    # Merge on IMDb ID
    integrated_df = movies_df.merge(
        streaming_df[['imdb_id', 'Netflix', 'Hulu', 'Prime Video', 'Disney+']],
        on='imdb_id',
        how='left'
    )
    
    return integrated_df


def split_by_platform(df):
    """
    Split integrated dataset by streaming platform.
    
    Args:
        df (pd.DataFrame): Integrated dataset
        
    Returns:
        dict: Dictionary containing dataframes for each platform and 'all'
    """
    datasets = {}
    
    # Full integrated dataset (drop platform columns)
    datasets['all'] = df.drop(columns=['Netflix', 'Hulu', 'Prime Video', 'Disney+'], errors='ignore')
    
    # Platform-specific datasets
    for platform in ['Netflix', 'Hulu', 'Prime Video', 'Disney+']:
        if platform in df.columns:
            datasets[platform.lower().replace(' ', '_')] = df[df[platform] == 1].drop(
                columns=['Netflix', 'Hulu', 'Prime Video', 'Disney+'],
                errors='ignore'
            )
    
    return datasets


def load_and_integrate_data(movies_path, streaming_path):
    """
    Complete preprocessing pipeline: load, clean, transform, and integrate data.
    
    Args:
        movies_path (str): Path to movies dataset
        streaming_path (str): Path to streaming platforms dataset
        
    Returns:
        dict: Dictionary of preprocessed dataframes by platform
    """
    # Load datasets
    movies_df = load_movies_dataset(movies_path)
    streaming_df = load_streaming_platforms_dataset(streaming_path)
    
    # Preprocess movies dataset
    movies_df = select_relevant_features(movies_df)
    movies_df = clean_missing_values(movies_df)
    movies_df = transform_json_fields(movies_df)
    movies_df = preprocess_dates(movies_df)
    
    # Integrate with streaming data
    integrated_df = integrate_streaming_data(movies_df, streaming_df)
    
    # Split by platform
    datasets = split_by_platform(integrated_df)
    
    return datasets
