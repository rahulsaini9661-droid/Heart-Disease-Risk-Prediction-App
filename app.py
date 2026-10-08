import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# Saved Model Load Karein
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("❤️ Heart Disease Risk Prediction App")
st.write("Apni health details sidebar mein enter karein yeh pata lagane ke liye ki heart disease ka risk hai ya nahi.")

# Sidebar Inputs for User Parameters
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=52)
    sex = st.sidebar.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
    cp = st.sidebar.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', 100, 600, 230)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', [1, 0])
    restecg = st.sidebar.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [1, 0])
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-3)', [0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', [1, 2, 3])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main Panel par entered values dikhayein
st.subheader("Aapke Dale Gaye Parameters Summary:")
st.dataframe(input_df)

# Prediction Button
if st.button('Predict Heart Disease Risk', type="primary"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Analysis Result:")
    if prediction[0] == 1:
        st.error("⚠️ Warning: Model ke anusaar heart disease ka **High Risk** ho sakta hai. Kripya doctor se consult karein!")
    else:
        st.success("✅ Good News: Model ke anusaar heart disease ka **Low Risk** hai.")
    
    st.info(f"Prediction Confidence Probability: **{np.max(prediction_proba)*100:.2f}%**")