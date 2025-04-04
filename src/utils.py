import os
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.base import BaseEstimator, TransformerMixin


def ensure_directories():
    """
    Ensure all required directories exist
    """
    directories = [
        'data',
        'models',
        'static/images',
        'static/css',
        'static/js',
        'templates',
        'logs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")


def load_model_and_preprocessor(model_path='models/best_model.pkl', preprocessor_path='models/preprocessor.pkl'):
    """
    Load the trained model and preprocessor
    """
    try:
        model = joblib.load(model_path)
        preprocessor = pickle.load(open(preprocessor_path, 'rb'))
        print(f"Model loaded from {model_path}")
        print(f"Preprocessor loaded from {preprocessor_path}")
        return model, preprocessor
    except Exception as e:
        print(f"Error loading model or preprocessor: {str(e)}")
        return None, None


def prepare_input_data(form_data):
    """
    Prepare input data from form submission for prediction
    """
    # Define default values for fields not in the form
    defaults = {
        'contact': 'cellular',
        'month': 'may',
        'day_of_week': 'mon',
        'duration': 0,
        'campaign': 1,
        'pdays': 999,
        'previous': 0,
        'poutcome': 'nonexistent',
        'emp.var.rate': 0,
        'cons.price.idx': 0,
        'cons.conf.idx': 0,
        'euribor3m': 0,
        'nr.employed': 0
    }

    # Combine form data with defaults
    data = {**defaults, **form_data}

    # Convert to appropriate types
    if 'age' in data:
        data['age'] = int(data['age'])

    # Create DataFrame
    input_df = pd.DataFrame([data])

    return input_df


def log_prediction(input_data, prediction, probability, log_file='logs/predictions.csv'):
    """
    Log prediction details to a CSV file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create a log entry
    log_entry = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': 'Yes' if prediction == 1 else 'No',
        'probability': probability,
        **input_data
    }

    # Convert to DataFrame
    log_df = pd.DataFrame([log_entry])

    # Append to log file or create if it doesn't exist
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    print(f"Prediction logged to {log_file}")


class MissingIndicator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create indicators for missing values
    """

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for feature in self.features:
            X_copy[f'{feature}_missing'] = X_copy[feature].isna().astype(int)
        return X_copy


if __name__ == "__main__":
    ensure_directories()
    print("Utility functions loaded successfully!")