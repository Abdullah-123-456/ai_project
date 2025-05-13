import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load models
chi2_dnn = load_model("chi2_dnn_model.h5")
chi2_ann = load_model("chi2_ann_model.h5")
dnn = load_model("dnn_model.h5")
ann = load_model("ann_model.h5")

# Load saved scalers
chi2_scaler = joblib.load("chi2_scaler.pkl")
full_scaler = joblib.load("full_scaler.pkl")

# Chi²-selected features
chi2_selected_features = ['sex', 'cp', 'trestbps', 'fbs', 'restecg',
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# All features
all_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Prediction")
st.markdown("Enter patient data to get predictions from trained models.")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest pain type", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250)
    chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
    thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=250)
    exang = st.selectbox("Exercise induced angina", options=[0, 1])
    oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of the ST segment", options=[0, 1, 2])
    ca = st.selectbox("Number of major vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[x])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input dataframe
    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    df = pd.DataFrame([input_data])

    # Prepare scaled inputs
    chi2_df = df[chi2_selected_features]
    full_df = df[all_features]

    chi2_scaled = chi2_scaler.transform(chi2_df)
    full_scaled = full_scaler.transform(full_df)

    # Prediction function
    def predict(model, X):
        prob = model.predict(X, verbose=0)[0][0]
        label = "Heart Disease" if prob > 0.5 else "Healthy"
        return label, prob

    st.subheader("Predictions")
    col1, col2 = st.columns(2)

    with col1:
        label, prob = predict(chi2_dnn, chi2_scaled)
        st.metric("χ²-DNN", label, f"{prob:.2f}")
        label, prob = predict(dnn, full_scaled)
        st.metric("DNN", label, f"{prob:.2f}")

    with col2:
        label, prob = predict(chi2_ann, chi2_scaled)
        st.metric("χ²-ANN", label, f"{prob:.2f}")
        label, prob = predict(ann, full_scaled)
        st.metric("ANN", label, f"{prob:.2f}")
