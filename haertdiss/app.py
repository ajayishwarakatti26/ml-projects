
import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration & Custom CSS
st.set_page_config(
    page_title="CardioPredict - Heart Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
        color: white;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Models Safely
@st.cache_resource
def load_assets():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("sclaer.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure `.pkl` files are in the directory.")
    st.stop()

# 3. Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/experimental-heart-outline-.png", width=70)
    st.markdown("### About CardioPredict")
    st.info(
        "This tool uses a Machine Learning (**K-Nearest Neighbors**) model "
        "to assess potential heart disease risk based on clinical parameters."
    )
    st.markdown("---")
    st.markdown("**Developer:** Akarsh")
    st.markdown("⚠️ *Note: This tool is for screening purposes and does not replace professional medical diagnosis.*")

# 4. Main Title Header
st.title("❤️ CardioPredict: Heart Risk Assessment")
st.markdown("Please fill out the patient evaluation metrics below to get an instant cardiovascular risk screening.")
st.markdown("---")

# 5. Form Grouping using Tabs for a Cleaner UI
tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🩺 Vitals & Blood", "⚡ Cardiac Metrics"])

with tab1:
    st.markdown("#### Patient Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age (Years)", 18, 100, 45, help="Patient's age in years.")
    with col2:
        sex = st.selectbox("Sex", ["M", "F"], help="Biological sex of the patient.")

with tab2:
    st.markdown("#### Vitals & Laboratory Results")
    col1, col2 = st.columns(2)
    with col1:
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Resting blood pressure upon admission.")
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol in mg/dl.")
    with col2:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", 
            [0, 1], 
            format_func=lambda x: "Yes (> 120 mg/dL)" if x == 1 else "No (<= 120 mg/dL)",
            help="Is fasting blood sugar greater than 120 mg/dl?"
        )

with tab3:
    st.markdown("#### Cardiac & ECG Metrics")
    col1, col2 = st.columns(2)
    with col1:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            format_func=lambda x: {
                "ATA": "Atypical Angina (ATA)",
                "NAP": "Non-Anginal Pain (NAP)",
                "TA": "Typical Angina (TA)",
                "ASY": "Asymptomatic (ASY)"
            }[x],
            help="Type of chest pain experienced."
        )
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            format_func=lambda x: {
                "Normal": "Normal",
                "ST": "ST-T wave abnormality",
                "LVH": "Left ventricular hypertrophy"
            }[x],
            help="Resting electrocardiogram result."
        )
    with col2:
        max_hr = st.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved during exercise.")
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
            help="Whether angina was induced by exercise."
        )
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, help="ST depression induced by exercise relative to rest.")
        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"],
            format_func=lambda x: {
                "Up": "Upsloping",
                "Flat": "Flat",
                "Down": "Downsloping"
            }[x],
            help="Slope of the peak exercise ST segment."
        )

# 6. Prediction Button & Results
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if st.button("Predict Heart Disease Risk", use_container_width=True):
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1,
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
    st.markdown("</div>", unsafe_allow_html=True)
