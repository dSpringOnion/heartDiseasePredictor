import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Real trained model loader
@st.cache_resource
def load_model():
    """Load the trained heart disease prediction model"""
    try:
        # Try to load the actual trained model
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        
        # Load feature names
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Load metadata
        import json
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ Loaded trained model from Cleveland Heart Disease dataset")
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'metadata': metadata
        }
        
    except FileNotFoundError:
        print("⚠️ Trained model not found. Please run 'python train_model.py' first.")
        # Fallback to demo model
        return load_demo_model()

def load_demo_model():
    """Fallback demo model if trained model is not available"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Create a simple demo model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    
    # Generate demo training data
    np.random.seed(42)
    X = np.random.randn(1000, 17)  # 17 features including engineered ones
    y = (X.sum(axis=1) > 0).astype(int)
    
    # Train demo model
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                    'age_group', 'chol_risk', 'bp_risk', 'hr_concern']
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metadata': {'model_type': 'Demo Model', 'dataset': 'Synthetic Data'}
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <h3>🏥 About This Demo</h3>
        <p>This interactive demo simulates a heart disease risk prediction model using machine learning. 
        Enter your health parameters below to get a risk assessment. This is for demonstration purposes only 
        and should not be used for actual medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    metadata = model_data['metadata']
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Basic Information")
        age = st.slider("Age", min_value=20, max_value=100, value=50, help="Your age in years")
        sex = st.selectbox("Sex", options=["Male", "Female"], help="Biological sex")
        cp = st.selectbox("Chest Pain Type", 
                         options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
                         help="Type of chest pain experienced")
        
        st.subheader("🩺 Vital Signs")
        trestbps = st.slider("Resting Blood Pressure", min_value=80, max_value=200, value=120, 
                            help="Resting blood pressure in mm Hg")
        chol = st.slider("Cholesterol Level", min_value=100, max_value=400, value=200,
                        help="Serum cholesterol in mg/dl")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"],
                          help="Is fasting blood sugar greater than 120 mg/dl?")
    
    with col2:
        st.subheader("🔬 Test Results")
        restecg = st.selectbox("Resting ECG Results",
                              options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                              help="Resting electrocardiographic results")
        thalach = st.slider("Maximum Heart Rate", min_value=60, max_value=220, value=150,
                           help="Maximum heart rate achieved during exercise")
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"],
                            help="Exercise induced angina")
        
        st.subheader("📈 Exercise Test")
        oldpeak = st.slider("ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1,
                           help="ST depression induced by exercise relative to rest")
        slope = st.selectbox("Slope of Peak Exercise ST Segment",
                            options=["Upsloping", "Flat", "Downsloping"],
                            help="The slope of the peak exercise ST segment")
        ca = st.slider("Number of Major Vessels", min_value=0, max_value=4, value=0,
                      help="Number of major vessels colored by fluoroscopy")
        thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"],
                           help="Thalassemia type")
    
    # Convert inputs to numerical values
    sex_val = 1 if sex == "Male" else 0
    cp_val = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
    fbs_val = 1 if fbs == "Yes" else 0
    restecg_val = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    exang_val = 1 if exang == "Yes" else 0
    slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
    
    # Create feature vector with feature engineering
    base_features = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                    thalach, exang_val, oldpeak, slope_val, ca, thal_val]
    
    # Feature engineering (same as in training)
    age_group = 0 if age <= 40 else (1 if age <= 55 else (2 if age <= 70 else 3))
    chol_risk = 1 if chol > 240 else 0
    bp_risk = 1 if trestbps > 140 else 0
    hr_concern = 1 if thalach < 100 else 0
    
    engineered_features = [age_group, chol_risk, bp_risk, hr_concern]
    
    # Combine all features
    all_features = base_features + engineered_features
    
    # Ensure we have the right number of features
    if len(all_features) != len(feature_names):
        # Pad or truncate to match expected features
        while len(all_features) < len(feature_names):
            all_features.append(0)
        all_features = all_features[:len(feature_names)]
    
    # Prediction button
    if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing your health data..."):
            try:
                # Scale features and get prediction
                features_scaled = scaler.transform([all_features])
                probabilities = model.predict_proba(features_scaled)[0]
                risk_probability = probabilities[1]  # Probability of heart disease
            except Exception as e:
                st.error(f"Prediction error: {e}")
                risk_probability = 0.5  # Default fallback
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Prediction Results")
            
            if risk_probability < 0.5:
                st.markdown(f"""
                <div class="prediction-box low-risk">
                    <h2>✅ Low Risk</h2>
                    <p>Probability of Heart Disease: <strong>{risk_probability:.1%}</strong></p>
                    <p>Your health parameters suggest a lower risk of heart disease. Keep maintaining a healthy lifestyle!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box high-risk">
                    <h2>⚠️ Higher Risk</h2>
                    <p>Probability of Heart Disease: <strong>{risk_probability:.1%}</strong></p>
                    <p>Your health parameters suggest a higher risk. Please consult with a healthcare professional for proper evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk factors analysis
            st.subheader("📊 Risk Factor Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Age Factor", f"{age} years", 
                         "Higher risk" if age > 55 else "Normal range")
            
            with col2:
                st.metric("Cholesterol", f"{chol} mg/dl",
                         "Elevated" if chol > 240 else "Normal range")
            
            with col3:
                st.metric("Max Heart Rate", f"{thalach} bpm",
                         "Lower capacity" if thalach < 120 else "Good capacity")
    
    # Model information section
    st.markdown("---")
    
    # Display model metadata
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ Important Disclaimer</h4>
            <p>This is a demonstration model and should NOT be used for actual medical diagnosis. 
            Always consult with qualified healthcare professionals for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>🛠️ Technical Details</h4>
            <p><strong>Model:</strong> {metadata.get('model_type', 'Machine Learning Model')}<br>
            <strong>Dataset:</strong> {metadata.get('dataset', 'Heart Disease Dataset')}<br>
            <strong>Features:</strong> {len(feature_names)} clinical parameters<br>
            <strong>Framework:</strong> Scikit-learn, Streamlit, Railway</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add dataset citation if using real data
    if 'Cleveland' in metadata.get('dataset', ''):
        st.markdown("""
        <div class="info-box">
            <h4>📚 Dataset Citation</h4>
            <p><strong>Source:</strong> UCI Machine Learning Repository<br>
            <strong>Citation:</strong> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). 
            Heart Disease. UCI Machine Learning Repository. 
            <a href="https://doi.org/10.24432/C52P4X" target="_blank">https://doi.org/10.24432/C52P4X</a></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()