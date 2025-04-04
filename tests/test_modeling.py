import pytest
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modeling import train_logistic_regression, train_svm, evaluate_model


@pytest.fixture
def sample_processed_data():
    """Create a small sample processed dataset for testing"""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    # Create features with some correlation to the target
    X = np.random.randn(n_samples, n_features)

    # Create a target variable with some pattern
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split into train and test
    train_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
    test_idx = np.array(list(set(range(n_samples)) - set(train_idx)))

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def test_train_logistic_regression(sample_processed_data, tmp_path):
    """Test the logistic regression training function"""
    X_train, _, y_train, _ = sample_processed_data
    output_dir = tmp_path / "models"
    output_dir.mkdir()

    # Train the model
    model = train_logistic_regression(X_train, y_train, output_dir=str(output_dir))

    # Check that the model was created and saved
    assert isinstance(model, LogisticRegression)
    assert os.path.exists(output_dir / "logistic_regression_model.pkl")

    # Load the model and check it's the same type
    loaded_model = joblib.load(output_dir / "logistic_regression_model.pkl")
    assert isinstance(loaded_model, LogisticRegression)


def test_train_svm(sample_processed_data, tmp_path):
    """Test the SVM training function"""
    X_train, _, y_train, _ = sample_processed_data
    output_dir = tmp_path / "models"
    output_dir.mkdir()

    # Train the model
    model = train_svm(X_train, y_train, output_dir=str(output_dir))

    # Check that the model was created and saved
    assert isinstance(model, SVC)
    assert os.path.exists(output_dir / "svm_model.pkl")

    # Load the model and check it's the same type
    loaded_model = joblib.load(output_dir / "svm_model.pkl")
    assert isinstance(loaded_model, SVC)


def test_evaluate_model(sample_processed_data, tmp_path):
    """Test the model evaluation function"""
    X_train, X_test, y_train, y_test = sample_processed_data
    output_dir = tmp_path / "images"
    output_dir.mkdir()

    # Train a simple model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy, auc = evaluate_model(
        model, X_train, X_test, y_train, y_test,
        "Test Model", output_dir=str(output_dir)
    )

    # Check that the metrics were calculated
    assert 0 <= accuracy <= 1
    assert 0 <= auc <= 1

    # Check that the visualizations were created
    assert os.path.exists(output_dir / "test_model_confusion_matrix.png")
    assert os.path.exists(output_dir / "test_model_roc_curve.png")
    assert os.path.exists(output_dir / "test_model_pr_curve.png")