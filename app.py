from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib  # Use joblib instead of pickle
import os
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize global variables
model = None
preprocessor = None
expected_columns = []


# Load the model and preprocessor
def load_model_and_preprocessor():
    global model, preprocessor, expected_columns

    try:
        # Try to load with joblib first
        model_paths = ['models/best_model.joblib', 'models/best_model.pkl']
        preprocessor_paths = ['models/preprocessor.joblib', 'models/preprocessor.pkl']

        # Try to load the model
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    logger.info(f"Attempting to load model from {model_path}")
                    model = joblib.load(model_path)
                    logger.info(f"Successfully loaded model from {model_path}")
                    break
                except Exception as e:
                    logger.error(f"Error loading model from {model_path}: {str(e)}")

        # Try to load the preprocessor
        for preprocessor_path in preprocessor_paths:
            if os.path.exists(preprocessor_path):
                try:
                    logger.info(f"Attempting to load preprocessor from {preprocessor_path}")
                    preprocessor = joblib.load(preprocessor_path)
                    logger.info(f"Successfully loaded preprocessor from {preprocessor_path}")
                    break
                except Exception as e:
                    logger.error(f"Error loading preprocessor from {preprocessor_path}: {str(e)}")

        # If model or preprocessor is still None, run preprocessing script
        if model is None or preprocessor is None:
            logger.error("Model or preprocessor could not be loaded. Running preprocessing script.")
            # Run preprocessing script to create the model and preprocessor
            import subprocess
            try:
                subprocess.run(['python', '-m', 'src.preprocessing'], check=True)
                subprocess.run(['python', '-m', 'src.modeling'], check=True)

                # Try to load again
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        try:
                            model = joblib.load(model_path)
                            logger.info(f"Successfully loaded model from {model_path} after preprocessing")
                            break
                        except Exception as e:
                            logger.error(f"Error loading model from {model_path} after preprocessing: {str(e)}")

                for preprocessor_path in preprocessor_paths:
                    if os.path.exists(preprocessor_path):
                        try:
                            preprocessor = joblib.load(preprocessor_path)
                            logger.info(
                                f"Successfully loaded preprocessor from {preprocessor_path} after preprocessing")
                            break
                        except Exception as e:
                            logger.error(
                                f"Error loading preprocessor from {preprocessor_path} after preprocessing: {str(e)}")
            except Exception as e:
                logger.error(f"Error running preprocessing script: {str(e)}")
                create_fallback_model_and_preprocessor()
    except Exception as e:
        logger.error(f"Error in load_model_and_preprocessor: {str(e)}")
        traceback.print_exc()
        create_fallback_model_and_preprocessor()

    # Get the expected columns from the preprocessor
    if preprocessor is not None and hasattr(preprocessor, 'transformers_'):
        for _, _, cols in preprocessor.transformers_:
            if isinstance(cols, list):
                expected_columns.extend(cols)

    logger.info(f"Expected columns: {expected_columns}")


def create_fallback_model_and_preprocessor():
    """Create a simple model and preprocessor as fallback"""
    global model, preprocessor

    logger.info("Creating fallback model and preprocessor")

    try:
        # Create a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        # Fit the model with dummy data
        model.fit(np.array([[0], [1]]), np.array([0, 1]))

        # Save the model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/best_model.joblib')

        # Create a simple preprocessor
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        # Define categorical and numerical features
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                                'contact', 'month', 'day_of_week', 'poutcome']
        numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous',
                              'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit the preprocessor with dummy data
        dummy_data = pd.DataFrame({
            'age': [30, 40],
            'job': ['admin.', 'blue-collar'],
            'marital': ['married', 'single'],
            'education': ['secondary', 'tertiary'],
            'default': ['no', 'no'],
            'housing': ['yes', 'no'],
            'loan': ['no', 'no'],
            'contact': ['cellular', 'telephone'],
            'month': ['may', 'jun'],
            'day_of_week': ['mon', 'tue'],
            'duration': [100, 200],
            'campaign': [1, 2],
            'pdays': [999, 999],
            'previous': [0, 0],
            'poutcome': ['nonexistent', 'nonexistent'],
            'emp.var.rate': [1.1, 1.1],
            'cons.price.idx': [93.994, 93.994],
            'cons.conf.idx': [-36.4, -36.4],
            'euribor3m': [4.857, 4.857],
            'nr.employed': [5191, 5191]
        })

        preprocessor.fit(dummy_data)

        # Save the preprocessor
        joblib.dump(preprocessor, 'models/preprocessor.joblib')

        logger.info("Fallback model and preprocessor created and saved")
    except Exception as e:
        logger.error(f"Error creating fallback model and preprocessor: {str(e)}")
        traceback.print_exc()


# Load the model and preprocessor at startup
load_model_and_preprocessor()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model, preprocessor, expected_columns

    try:
        # Get input data from form
        input_data = {
            'age': request.form.get('age'),
            'job': request.form.get('job'),
            'marital': request.form.get('marital'),
            'education': request.form.get('education'),
            'default': request.form.get('default'),
            'housing': request.form.get('housing'),
            'loan': request.form.get('loan'),
            'contact': request.form.get('contact', 'telephone'),
            'month': request.form.get('month', 'may'),
            'day_of_week': request.form.get('day_of_week', 'mon'),
            'duration': request.form.get('duration', '0'),
            'campaign': request.form.get('campaign', '1'),
            'pdays': request.form.get('pdays', '999'),
            'previous': request.form.get('previous', '0'),
            'poutcome': request.form.get('poutcome', 'nonexistent'),
            'emp.var.rate': request.form.get('emp.var.rate', '1.1'),
            'cons.price.idx': request.form.get('cons.price.idx', '93.994'),
            'cons.conf.idx': request.form.get('cons.conf.idx', '-36.4'),
            'euribor3m': request.form.get('euribor3m', '4.857'),
            'nr.employed': request.form.get('nr.employed', '5191')
        }

        logger.info(f"Received prediction request with data: {input_data}")

        # If model or preprocessor is not available, reload them
        if model is None or preprocessor is None:
            logger.warning("Model or preprocessor is not available. Reloading...")
            load_model_and_preprocessor()

            if model is None or preprocessor is None:
                logger.error("Failed to load model or preprocessor. Creating fallback.")
                create_fallback_model_and_preprocessor()

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert numeric columns to float
        numeric_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(float)

        # Check if we have all the expected columns
        missing_cols = [col for col in expected_columns if col not in input_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in input data: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                if col in numeric_cols:
                    input_df[col] = 0.0
                else:
                    input_df[col] = 'unknown'

        # Process input data
        try:
            logger.info(f"Input DataFrame: {input_df}")
            input_processed = preprocessor.transform(input_df)

            # Make prediction
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0][1]

            result = {
                'prediction': 'yes' if prediction == 1 else 'no',
                'probability': f"{probability:.2f}"
            }

            logger.info(f"Prediction result: {result}")
            return render_template('result.html', result=result)

        except Exception as ve:
            logger.error(f"Error making prediction: {str(ve)}")
            traceback.print_exc()

            # Try to reload the model and preprocessor
            logger.warning("Trying to reload the model and preprocessor...")
            load_model_and_preprocessor()

            if model is not None and preprocessor is not None:
                try:
                    # Try prediction again
                    input_processed = preprocessor.transform(input_df)
                    prediction = model.predict(input_processed)[0]
                    probability = model.predict_proba(input_processed)[0][1]

                    result = {
                        'prediction': 'yes' if prediction == 1 else 'no',
                        'probability': f"{probability:.2f}"
                    }

                    logger.info(f"Prediction result after reload: {result}")
                    return render_template('result.html', result=result)
                except Exception as e2:
                    logger.error(f"Error making prediction after reload: {str(e2)}")

            # If all else fails, create a fallback model and preprocessor
            create_fallback_model_and_preprocessor()

            try:
                # Try prediction with fallback model
                input_processed = preprocessor.transform(input_df)
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0][1]

                result = {
                    'prediction': 'yes' if prediction == 1 else 'no',
                    'probability': f"{probability:.2f}",
                    'message': 'Note: This prediction was made with a fallback model.'
                }

                logger.info(f"Prediction result with fallback model: {result}")
                return render_template('result.html', result=result)
            except Exception as e3:
                logger.error(f"Error making prediction with fallback model: {str(e3)}")

                result = {
                    'error': str(ve),
                    'message': 'An error occurred during prediction. Please try again with different input values.'
                }

                return render_template('result.html', result=result)

    except Exception as e:
        logger.error(f"Error in prediction route: {str(e)}")
        traceback.print_exc()

        result = {
            'error': str(e),
            'message': 'An error occurred during prediction.'
        }

        return render_template('result.html', result=result)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)