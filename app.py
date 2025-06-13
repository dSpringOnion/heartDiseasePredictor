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
    
    # Generate realistic demo training data based on Cleveland dataset patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Generate base features with realistic distributions
    age = np.random.normal(54, 9, n_samples).clip(29, 77)
    sex = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.random.normal(131, 17, n_samples).clip(94, 200)
    chol = np.random.normal(246, 51, n_samples).clip(126, 564)
    
    # Create realistic risk-based target
    risk_score = (
        (age > 55) * 0.3 +
        (sex == 1) * 0.2 +
        (cp < 2) * 0.4 +
        (trestbps > 140) * 0.2 +
        (chol > 240) * 0.3
    )
    risk_score += np.random.normal(0, 0.2, n_samples)
    y = (risk_score > 0.8).astype(int)
    
    # Create feature matrix with all 24 features
    X = np.column_stack([
        age, sex, cp, trestbps, chol,
        np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),  # fbs
        np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.48, 0.04]),  # restecg
        np.random.normal(149, 22, n_samples).clip(71, 202),  # thalach
        np.random.choice([0, 1], n_samples, p=[0.68, 0.32]),  # exang
        np.random.exponential(1.04, n_samples).clip(0, 6.2),  # oldpeak
        np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.40, 0.39]),  # slope
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.59, 0.22, 0.13, 0.06]),  # ca
        np.random.choice([0, 1, 2], n_samples, p=[0.55, 0.38, 0.07]),  # thal
        # Engineered features (11 additional)
        np.random.choice([0, 1, 2, 3], n_samples),  # age_group
        (chol > 240).astype(int),  # chol_risk
        (trestbps > 140).astype(int),  # bp_risk
        np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),  # hr_concern
        np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),  # metabolic_syndrome
        np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  # cardiac_stress
        np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # electrical_abnormality
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),  # vessel_disease
        cp,  # chest_pain_severity
        np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),  # exercise_tolerance
        np.random.choice([0, 1], n_samples, p=[0.8, 0.2])   # perfusion_defect
    ])
    
    # Train demo model
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                    'age_group', 'chol_risk', 'bp_risk', 'hr_concern',
                    'metabolic_syndrome', 'cardiac_stress', 'electrical_abnormality', 
                    'vessel_disease', 'chest_pain_severity', 'exercise_tolerance', 
                    'perfusion_defect']
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metadata': {'model_type': 'Enhanced Demo Model', 'dataset': 'Realistic Synthetic Data (Cleveland-based)'}
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
    
    # Comprehensive clinical risk factor calculations
    age_group = 0 if age <= 40 else (1 if age <= 55 else (2 if age <= 70 else 3))
    chol_risk = 1 if chol > 240 else 0
    bp_risk = 1 if trestbps > 140 else 0
    hr_concern = 1 if thalach < 100 else 0
    
    # Advanced cardiovascular risk indicators
    metabolic_syndrome = (chol > 200 and trestbps > 130 and fbs_val == 1)
    cardiac_stress = (thalach < 150 and exang_val == 1 and oldpeak > 1.0)
    electrical_abnormality = (restecg_val > 0 or slope_val == 2)
    vessel_disease = (ca > 0 or thal_val == 2)
    
    # Clinical severity scores
    chest_pain_severity = cp_val  # 0=typical angina (most severe), 3=asymptomatic (least severe)
    exercise_tolerance = 1 if (thalach < 120 or exang_val == 1) else 0
    perfusion_defect = 1 if (thal_val == 2 or oldpeak > 2.0) else 0
    
    # Comprehensive feature engineering
    engineered_features = [
        age_group, chol_risk, bp_risk, hr_concern,
        int(metabolic_syndrome), int(cardiac_stress), 
        int(electrical_abnormality), int(vessel_disease),
        chest_pain_severity, exercise_tolerance, perfusion_defect
    ]
    
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
            
            # Comprehensive Risk Factor Analysis
            st.subheader("📊 Comprehensive Risk Factor Analysis")
            
            # Primary Clinical Indicators
            st.markdown("#### 🩺 Primary Clinical Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                age_status = "Higher risk" if age > 55 else ("Moderate risk" if age > 45 else "Lower risk")
                st.metric("Age Factor", f"{age} years", age_status)
            
            with col2:
                chol_status = "High risk" if chol > 240 else ("Borderline" if chol > 200 else "Optimal")
                st.metric("Cholesterol", f"{chol} mg/dl", chol_status)
            
            with col3:
                bp_status = "Hypertension" if trestbps > 140 else ("Pre-hypertension" if trestbps > 120 else "Normal")
                st.metric("Blood Pressure", f"{trestbps} mmHg", bp_status)
            
            with col4:
                hr_status = "Reduced capacity" if thalach < 120 else ("Good" if thalach > 150 else "Fair")
                st.metric("Max Heart Rate", f"{thalach} bpm", hr_status)
            
            # Advanced Cardiovascular Risk Indicators
            st.markdown("#### 🫀 Advanced Cardiovascular Risk Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metabolic_status = "⚠️ Present" if metabolic_syndrome else "✅ Absent"
                st.metric("Metabolic Syndrome", metabolic_status, 
                         "High cholesterol + hypertension + diabetes")
            
            with col2:
                cardiac_status = "⚠️ Present" if cardiac_stress else "✅ Normal"
                st.metric("Cardiac Stress", cardiac_status,
                         "Low exercise capacity + angina")
            
            with col3:
                electrical_status = "⚠️ Abnormal" if electrical_abnormality else "✅ Normal"
                st.metric("ECG/Exercise Response", electrical_status,
                         "Electrical conduction issues")
            
            with col4:
                vessel_status = "⚠️ Disease Present" if vessel_disease else "✅ Normal"
                st.metric("Vessel/Perfusion", vessel_status,
                         "Coronary vessel or perfusion defects")
            
            # Clinical Risk Score Breakdown
            st.markdown("#### 📈 Clinical Risk Score Breakdown")
            
            risk_factors = {
                "Age Risk": age > 55,
                "Male Gender": sex_val == 1,
                "Chest Pain (Symptomatic)": cp_val < 3,
                "Hypertension": trestbps > 140,
                "High Cholesterol": chol > 240,
                "Diabetes": fbs_val == 1,
                "ECG Abnormalities": restecg_val > 0,
                "Reduced Exercise Capacity": thalach < 150,
                "Exercise-Induced Angina": exang_val == 1,
                "ST Depression": oldpeak > 1.0,
                "Abnormal ST Slope": slope_val == 2,
                "Vessel Disease": ca > 0,
                "Perfusion Defect": thal_val == 2
            }
            
            present_risks = [factor for factor, present in risk_factors.items() if present]
            absent_risks = [factor for factor, present in risk_factors.items() if not present]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⚠️ Risk Factors Present:**")
                if present_risks:
                    for risk in present_risks:
                        st.markdown(f"• {risk}")
                else:
                    st.markdown("• No major risk factors identified")
            
            with col2:
                st.markdown("**✅ Protective Factors:**")
                if absent_risks:
                    for risk in absent_risks[:6]:  # Show first 6 to avoid clutter
                        st.markdown(f"• No {risk}")
                    if len(absent_risks) > 6:
                        st.markdown(f"• ...and {len(absent_risks) - 6} more protective factors")
                
            # Risk Score Summary
            risk_score = sum(risk_factors.values())
            total_factors = len(risk_factors)
            risk_percentage = (risk_score / total_factors) * 100
            
            st.markdown("#### 📊 Overall Risk Assessment")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Factors Present", f"{risk_score}/{total_factors}")
            
            with col2:
                st.metric("Risk Factor Score", f"{risk_percentage:.1f}%")
            
            with col3:
                risk_category = ("High" if risk_score >= 7 else 
                               "Moderate" if risk_score >= 4 else "Low")
                st.metric("Risk Category", risk_category)
    
    # Clinical Education Section
    st.markdown("---")
    st.subheader("📚 Understanding Your Results")
    
    # Create expandable sections for education
    with st.expander("🫀 **Cardiovascular Risk Factors Explained**"):
        st.markdown("""
        **Age & Gender:** Men over 45 and women over 55 have increased risk due to hormonal changes and arterial aging.
        
        **Chest Pain Types:**
        - *Typical Angina:* Classic heart pain with exertion, highest concern
        - *Atypical Angina:* Unusual chest pain patterns  
        - *Non-Anginal:* Chest pain likely not heart-related
        - *Asymptomatic:* No chest pain symptoms
        
        **Blood Pressure:** Hypertension (>140/90) damages arteries over time, increasing heart disease risk.
        
        **Cholesterol:** High levels (>240 mg/dl) lead to plaque buildup in arteries.
        
        **Blood Sugar:** Diabetes damages blood vessels and accelerates atherosclerosis.
        """)
    
    with st.expander("🔬 **Advanced Testing Explained**"):
        st.markdown("""
        **ECG (Electrocardiogram):** Measures heart's electrical activity
        - *Normal:* Regular heart rhythm
        - *ST-T Wave Abnormality:* Possible ischemia or old damage
        - *Left Ventricular Hypertrophy:* Heart muscle thickening from high blood pressure
        
        **Exercise Stress Test:**
        - *Maximum Heart Rate:* Higher is generally better (220 - age = target)
        - *Exercise Angina:* Chest pain during exercise indicates insufficient blood flow
        - *ST Depression:* Electrical changes suggesting blocked arteries
        
        **Coronary Angiography:**
        - *Vessel Count:* Number of major arteries with significant blockage (0-4)
        - *Thalassemia:* Blood flow patterns showing perfusion defects
        """)
    
    with st.expander("⚕️ **Clinical Interpretation Guidelines**"):
        st.markdown("""
        **Risk Stratification:**
        - *Low Risk (0-3 factors):* Standard preventive care
        - *Moderate Risk (4-6 factors):* Enhanced monitoring and lifestyle changes
        - *High Risk (7+ factors):* Aggressive treatment and specialist referral
        
        **Metabolic Syndrome:** Combination of diabetes, hypertension, and high cholesterol significantly increases risk.
        
        **Cardiac Stress Indicators:** Poor exercise tolerance combined with symptoms suggests significant coronary disease.
        
        **Important Note:** This analysis is for educational purposes only. Always consult with qualified cardiologists for proper medical evaluation and treatment decisions.
        """)
    
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