import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from PIL import Image



st.set_page_config(page_title="Brain Tumor Prediction", layout="centered")



st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #ffffff;
    text-align: center;
    text-shadow: 3px 3px 8px #000000;
    font-weight: 700;
}

/* PARAGRAPHS */
p {
    color: #e6e6e6;
    font-size: 18px;
    line-height: 1.6;
    text-align: justify;
    text-shadow: 1px 1px 3px black;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(8px);
    color: white;
}

/* BLACK BUTTONS */
.stButton > button {
    background-color: #000000 !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.7);
    transition: all 0.3s ease;
    font-size: 16px;
    padding: 8px 16px;
}
.stButton > button:hover {
    background-color: #333333 !important;
    transform: scale(1.07);
}

/* FIX BROWSE BUTTON */
[data-testid="stFileUploader"] > div > div > button {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
    border: 2px solid #000000 !important;
    font-size: 16px !important;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.7) !important;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] > div > div > button:hover {
    background-color: #303030 !important;
    transform: scale(1.05);
}

/* RESULT BOX */
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



st.header('🧠 Brain Tumor Prediction Using Machine Learning')

st.markdown("""
Brain tumor prediction is essential for early diagnosis and treatment.  
Machine learning helps analyze patient data and MRI findings to detect potential tumors accurately.  
This app uses a trained ML model to predict brain tumor presence based on clinical features and optional MRI scan.
""")



st.image(
    'https://www.yashodahospitals.com/wp-content/uploads/2018/03/Understanding-brain-tumours-1.jpg',
    use_container_width=True
)



with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = load_model('brain_tumor_pred_model.keras')



df = pd.read_csv('Brain_Tumor_Prediction_Dataset.csv.gz', compression='gzip')



feature_columns = [
    'Age', 'Gender', 'Country', 'Tumor_Size', 'Tumor_Location', 'MRI_Findings',
    'Genetic_Risk', 'Smoking_History', 'Alcohol_Consumption', 'Radiation_Exposure',
    'Head_Injury_History', 'Chronic_Illness', 'Diabetes', 'Tumor_Type',
    'Treatment_Received', 'Survival_Rate(%)', 'Tumor_Growth_Rate',
    'Family_History', 'Symptom_Severity', 'High_BP', 'Low_BP'
]



st.sidebar.header('🩺 Enter Patient Details')
st.sidebar.image('https://mdwestone.com/wp-content/uploads/2024/07/brain-tumor.jpg')


# MRI Upload
st.sidebar.header("🧲 Upload MRI Image (Optional)")
uploaded_image = st.sidebar.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])



def fraction_to_float(value):
    if isinstance(value, str) and '/' in value:
        try:
            num, denom = value.split('/')
            return float(num), float(denom)
        except:
            return np.nan, np.nan
    return np.nan, np.nan


def preprocess_mri_image(image, target_size=(128, 128)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



user_inputs = []

for feature in feature_columns:

    if feature in encoders:  # Categorical features
        options = encoders[feature].classes_.tolist()
        selected = st.sidebar.selectbox(f'{feature}', options)
        encoded = encoders[feature].transform([selected])[0]
        user_inputs.append(encoded)

    elif feature == 'High_BP':
        high_val = st.sidebar.slider('High_BP (Systolic)', 90, 180, 120)
        user_inputs.append(high_val)

    elif feature == 'Low_BP':
        low_val = st.sidebar.slider('Low_BP (Diastolic)', 60, 120, 80)
        user_inputs.append(low_val)

    else:  # Numeric
        numeric_col = pd.to_numeric(
            df[feature].apply(lambda x: fraction_to_float(x)[0] if isinstance(x, str) and '/' in x else x),
            errors='coerce'
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



if uploaded_image is not None:
    st.subheader("📌 Uploaded MRI Image")
    st.image(uploaded_image, caption="MRI Scan", use_container_width=True)



if st.sidebar.button('🔍 Predict Brain Tumor'):

    with st.spinner('🧬 Analyzing MRI and Patient Data...'):
        time.sleep(2)

        prediction_tabular = model.predict(scaled_input)[0][0]

        if uploaded_image is not None:
            try:
                image_model = load_model("mri_cnn_model.h5")
                mri_input = preprocess_mri_image(uploaded_image)
                prediction_image = image_model.predict(mri_input)[0][0]
                final_prediction = (prediction_tabular + prediction_image) / 2
            except:
                final_prediction = prediction_tabular
        else:
            final_prediction = prediction_tabular

        # Result UI
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        if final_prediction < 0.5:
            st.success(f'✅ No Brain Tumor Detected  
                        Confidence: {(1 - final_prediction) * 100:.2f}%')
        else:
            st.error(f'⚠️ Brain Tumor Detected  
                      Confidence: {final_prediction * 100:.2f}%')

        st.markdown("</div>", unsafe_allow_html=True)
