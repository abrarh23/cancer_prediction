"""Streamlit Frontend for Cancer Prediction"""
import streamlit as st
import pandas as pd
import numpy as np
from functools import lru_cache
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH: str = 'cancer_rf_model.pkl'

@lru_cache(maxsize=1)
def get_model() -> RandomForestClassifier:
    """Load the trained model from disk (cached to avoid reloading)"""
    if os.path.exists(MODEL_PATH):
        model: RandomForestClassifier = joblib.load(MODEL_PATH)
        return model
    else:
        st.error("Model not found. Please train the model first using cancer_prediction.py")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Cancer Prediction App",
    page_icon="🏥",
    layout="centered"
)

# Title and description
st.title("Cancer Prediction App")
st.markdown("Enter patient information below to predict cancer diagnosis using Random Forest model.")

# Sidebar for model info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses a **Random Forest Classifier** to predict cancer diagnosis based on:
    - Patient demographics
    - Lifestyle factors
    - Medical history
    """)
    
    st.info("""
    **Note:** Model training is handled by the backend API.
    To retrain the model, run `cancer_prediction.py`.
    """)

# Main form
with st.form("prediction_form"):
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=50,
            help="Patient's age in years"
        )
        
        gender = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="0 = Female, 1 = Male"
        )
        
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0,
            max_value=50.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index"
        )
        
        smoking = st.selectbox(
            "Smoking Status",
            options=[0, 1],
            format_func=lambda x: "Non-smoker" if x == 0 else "Smoker",
            help="0 = Non-smoker, 1 = Smoker"
        )
    
    with col2:
        genetic_risk = st.selectbox(
            "Genetic Risk",
            options=[0, 1, 2],
            format_func=lambda x: ["Low", "Medium", "High"][x],
            help="0 = Low, 1 = Medium, 2 = High"
        )
        
        physical_activity = st.number_input(
            "Physical Activity Level",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Physical activity level (0-20)"
        )
        
        alcohol_intake = st.number_input(
            "Alcohol Intake",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
            help="Alcohol intake level (0-20)"
        )
        
        cancer_history = st.selectbox(
            "Family Cancer History",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="0 = No family history, 1 = Family history"
        )
    
    submitted = st.form_submit_button("Predict", use_container_width=True)

# Prediction logic
if submitted:
    try:
        # Get the model
        model = get_model()
        
        # Get feature names from model or use default order
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [
                'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
                'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'
            ]
        
        # Prepare features
        features = [age, gender, bmi, smoking, genetic_risk, 
                   physical_activity, alcohol_intake, cancer_history]
        
        # Convert to DataFrame
        features_df = pd.DataFrame(
            [features],
            columns=feature_names,
            dtype=np.float64
        )
        
        # Make prediction
        prediction = int(model.predict(features_df)[0])
        prediction_proba = model.predict_proba(features_df)[0]
        
        # Display results
        st.divider()
        st.subheader("Prediction Results")
        
        # Diagnosis with color coding
        if prediction == 1:
            st.error("**Diagnosis: Malignant**")
        else:
            st.success("**Diagnosis: Benign**")
        
        # Probability bars
        col1, col2 = st.columns(2)
        
        with col1:
            benign_prob = float(prediction_proba[0]) * 100
            st.metric("Benign Probability", f"{benign_prob:.2f}%")
            st.progress(benign_prob / 100)
        
        with col2:
            malignant_prob = float(prediction_proba[1]) * 100
            st.metric("Malignant Probability", f"{malignant_prob:.2f}%")
            st.progress(malignant_prob / 100)
        
        # Additional info
        with st.expander("📋 View Input Summary"):
            input_data = {
                "Age": age,
                "Gender": "Female" if gender == 0 else "Male",
                "BMI": bmi,
                "Smoking": "Non-smoker" if smoking == 0 else "Smoker",
                "Genetic Risk": ["Low", "Medium", "High"][genetic_risk],
                "Physical Activity": physical_activity,
                "Alcohol Intake": alcohol_intake,
                "Cancer History": "No" if cancer_history == 0 else "Yes"
            }
            st.json(input_data)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure the model file exists. You can train it by running `cancer_prediction.py`.")

