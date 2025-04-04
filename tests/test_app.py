import pytest
from app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test that the home page loads correctly"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Bank Term Deposit Prediction' in response.data

def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_prediction_endpoint(client):
    """Test the prediction endpoint with form data"""
    test_data = {
        'age': '35',
        'job': 'management',
        'marital': 'married',
        'education': 'university.degree',
        'default': 'no',
        'housing': 'yes',
        'loan': 'no'
    }
    response = client.post('/predict', data=test_data)
    assert response.status_code == 200
    assert b'Prediction Result' in response.data

def test_api_prediction_endpoint(client):
    """Test the API prediction endpoint with JSON data"""
    test_data = {
        'age': 35,
        'job': 'management',
        'marital': 'married',
        'education': 'university.degree',
        'default': 'no',
        'housing': 'yes',
        'loan': 'no'
    }
    response = client.post('/api/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability' in data
    assert 'timestamp' in data