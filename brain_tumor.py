import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model

# ============ PAGE CONFIG ============
st.set_page_config(page_title="Brain Tumor Prediction", layout="centered")

# ============ CUSTOM CSS ============
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
        text-align: center;
        text-shadow: 3px 3px 8px #000000;
        font-weight: 700;
    }
    p {
        color: #e6e6e6;
        font-size: 18px;
        line-height: 1.6;
        text-align: justify;
        text-shadow: 1px 1px 3px black;
    }
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(8px);
        color: white;
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 10px;
        border: none;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        font-size: 16px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #00b4d8;
        color: black;
        transform: scale(1.05);
    }
    .result-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        margin-top: 30px;
        text-align: center;
        box-shadow: 0 4px 25px rgba(0,0,0,0.6);
    }
    </style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.header('🧠 Brain Tumor Prediction Using Machine Learning')

st.markdown('''
Brain tumor prediction is critical for early diagnosis and treatment.  
Machine learning models can analyze patient data and MRI findings to detect potential brain tumors accurately.  
This app uses a trained machine learning model to predict brain tumor presence based on clinical and diagnostic features.
''')

st.image(
    'https://www.yashodahospitals.com/wp-content/uploads/2018/03/Understanding-brain-tumours-1.jpg',
    use_container_width=True
)

# ============ CACHE LOADING FUNCTIONS ============
@st.cache_resource
def load_encoders(path='encoders.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler(path='scaler.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model(path='brain_tumor_pred_model.keras'):
    return load_model(path)

@st.cache_data
def load_dataset(path='data/brain_tumor_prediction_dataset.csv.gzip'):

# ============ LOAD FILES ============
try:
    encoders = load_encoders()
    scaler = load_scaler()
    model = load_keras_model()
    df = load_dataset()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}. Please upload it or check file paths.")
    st.stop()

# ============ FEATURES ============
feature_columns = [
    'Age', 'Gender', 'Country', 'Tumor_Size', 'Tumor_Location', 'MRI_Findings',
    'Genetic_Risk', 'Smoking_History', 'Alcohol_Consumption', 'Radiation_Exposure',
    'Head_Injury_History', 'Chronic_Illness', 'Diabetes', 'Tumor_Type',
    'Treatment_Received', 'Survival_Rate(%)', 'Tumor_Growth_Rate',
    'Family_History', 'Symptom_Severity', 'High_BP', 'Low_BP'
]

st.sidebar.header('🩺 Enter Patient Details')
st.sidebar.image('https://mdwestone.com/wp-content/uploads/2024/07/brain-tumor.jpg')

# ============ HELPER FUNCTION ============
def fraction_to_float(value):
    """Convert fraction strings like '120/80' to tuple of floats (120.0, 80.0)"""
    if isinstance(value, str) and '/' in value:
        try:
            num, denom = value.split('/')
            return float(num), float(denom)
        except:
            return np.nan, np.nan
    return np.nan, np.nan

# ============ USER INPUTS ============
user_inputs = []

for feature in feature_columns:
    if feature in encoders:
        options = encoders[feature].classes_.tolist()
        selected = st.sidebar.selectbox(f'{feature}', options)
        encoded = encoders[feature].transform([selected])[0]
        user_inputs.append(encoded)

    elif feature == 'High_BP':
        sample_bp = df['Blood_Pressure'].dropna().iloc[0] if 'Blood_Pressure' in df else '120/80'
        high_bp, _ = fraction_to_float(sample_bp)
        high_bp = int(high_bp) if not np.isnan(high_bp) else 120
        high_val = st.sidebar.slider('High_BP (Systolic)', 90, 180, high_bp)
        user_inputs.append(high_val)

    elif feature == 'Low_BP':
        sample_bp = df['Blood_Pressure'].dropna().iloc[0] if 'Blood_Pressure' in df else '120/80'
        _, low_bp = fraction_to_float(sample_bp)
        low_bp = int(low_bp) if not np.isnan(low_bp) else 80
        low_val = st.sidebar.slider('Low_BP (Diastolic)', 60, 120, low_bp)
        user_inputs.append(low_val)

    else:
        numeric_col = pd.to_numeric(
            df[feature].apply(
                lambda x: fraction_to_float(x)[0] if isinstance(x, str) and '/' in x else x
            ), errors='coerce'
        ).dropna()
        min_val = int(numeric_col.min()) if not numeric_col.empty else 0
        max_val = int(numeric_col.max()) if not numeric_col.empty else 10
        if min_val == max_val:
            max_val = min_val + 1
        default_val = int((min_val + max_val) / 2)
        selected = st.sidebar.slider(f'{feature}', min_val, max_val, default_val)
        user_inputs.append(selected)

input_array = np.array([user_inputs])
scaled_input = scaler.transform(input_array)

# ============ PREDICTION ============
if st.sidebar.button('🔍 Predict Brain Tumor'):
    with st.spinner('🧬 Analyzing MRI and Patient Data...'):
        time.sleep(2)
        prediction = model.predict(scaled_input)[0][0]
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        if prediction < 0.5:
            st.success(f'✅ No Brain Tumor Detected (Confidence: {(1 - prediction) * 100:.2f}%)')
        else:
            st.error(f'⚠️ Brain Tumor Detected (Confidence: {prediction * 100:.2f}%)')
        st.markdown("</div>", unsafe_allow_html=True)
