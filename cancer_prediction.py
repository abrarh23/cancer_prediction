"""Cancer Prediction API"""
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import pandas as pd
from flask import Flask, request, jsonify, Response
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from models import PredictionResponse, ErrorResponse

# Initialize Flask app
app = Flask(__name__)

MODEL_PATH: str = 'cancer_rf_model.pkl'

def train_and_save_model() -> RandomForestClassifier:
    """Train the Random Forest model and save it"""
    # Load the dataset
    raw_df: pd.DataFrame = pd.read_csv('cancer.csv')

    # Prepare features and target
    x: pd.DataFrame = raw_df.drop('Diagnosis', axis=1)
    y: pd.Series = raw_df['Diagnosis']

    # Split the data
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest model
    model: RandomForestClassifier = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

    # Print model performance
    y_pred: npt.NDArray[np.int_] = model.predict(x_test)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")

    return model

@lru_cache(maxsize=1)
def get_model() -> RandomForestClassifier:
    """Load the trained model from disk (cached to avoid reloading)"""
    if os.path.exists(MODEL_PATH):
        model: RandomForestClassifier = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    else:
        print("Model not found. Training new model...")
        return train_and_save_model()

@app.route('/predict', methods=['POST'])
def predict() -> Tuple[Response, int]:
    """API endpoint to predict cancer diagnosis"""
    try:
        # Get the model (cached, so it only loads once)
        model: RandomForestClassifier = get_model()
        # Get JSON data from request
        data: Optional[Dict[str, Any]] = request.get_json()

        if data is None:
            return jsonify({
                'error': 'Invalid JSON in request body',
                'required_fields': [
                    'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
                    'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'
                ]
            }), 400

        # Extract features in the correct order
        features: List[Optional[Union[int, float]]] = [
            data.get('Age'),
            data.get('Gender'),
            data.get('BMI'),
            data.get('Smoking'),
            data.get('GeneticRisk'),
            data.get('PhysicalActivity'),
            data.get('AlcoholIntake'),
            data.get('CancerHistory')
        ]

        # Check if all features are provided
        if None in features:
            error_response: ErrorResponse = {
                'error': 'Missing required fields',
                'required_fields': [
                    'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
                    'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'
                ]
            }
            return jsonify(error_response), 400

        # Get feature names from model or use default order
        feature_names: List[str]
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            # Fallback to expected feature order
            feature_names = [
                'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
                'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'
            ]

        # Convert to DataFrame with correct feature names to avoid warnings
        features_df: pd.DataFrame = pd.DataFrame(
            [features],
            columns=feature_names,
            dtype=np.float64
        )

        # Make prediction
        prediction: int = int(model.predict(features_df)[0])
        prediction_proba: npt.NDArray[np.float64] = model.predict_proba(features_df)[0]

        # Return result
        result: PredictionResponse = {
            'prediction': prediction,
            'diagnosis': 'Malignant' if prediction == 1 else 'Benign',
            'probability': {
                'benign': float(prediction_proba[0]),
                'malignant': float(prediction_proba[1])
            }
        }

        return jsonify(result), 200

    except (ValueError, TypeError, KeyError) as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health() -> Tuple[Response, int]:
    """Health check endpoint"""
    try:
        get_model()
        model_loaded: bool = True
    except Exception:
        model_loaded = False
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded}), 200

@app.route('/', methods=['GET'])
def home() -> Tuple[Response, int]:
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Cancer Prediction API',
        'endpoints': {
            '/predict': {
                'method': 'POST',
                'description': 'Predict cancer diagnosis',
                'required_fields': {
                    'Age': 'float or int',
                    'Gender': 'int (0 or 1)',
                    'BMI': 'float',
                    'Smoking': 'int (0 or 1)',
                    'GeneticRisk': 'int (0, 1, or 2)',
                    'PhysicalActivity': 'float',
                    'AlcoholIntake': 'float',
                    'CancerHistory': 'int (0 or 1)'
                },
                'example_request': {
                    'Age': 58,
                    'Gender': 1,
                    'BMI': 16.08,
                    'Smoking': 0,
                    'GeneticRisk': 1,
                    'PhysicalActivity': 8.14,
                    'AlcoholIntake': 4.14,
                    'CancerHistory': 1
                }
            },
            '/health': {
                'method': 'GET',
                'description': 'Check API health status'
            }
        }
    }), 200

if __name__ == '__main__':
    # Load or train the model on startup (this will cache it)
    get_model()

    # Run the Flask app
    print("\n" + "="*50)
    print("Cancer Prediction API is running!")
    print("="*50)
    print("\nEndpoints:")
    print("  GET  /          - API documentation")
    print("  GET  /health    - Health check")
    print("  POST /predict   - Make a prediction")
    print("\nExample curl command:")
    print('curl -X POST http://localhost:5000/predict \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"Age": 58, "Gender": 1, "BMI": 16.08, "Smoking": 0, "GeneticRisk": 1, "PhysicalActivity": 8.14, "AlcoholIntake": 4.14, "CancerHistory": 1}\'')
    print("\n" + "="*50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
