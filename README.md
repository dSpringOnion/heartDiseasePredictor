# Heart Disease Risk Predictor

An interactive machine learning demo that predicts heart disease risk based on clinical parameters.

## 🚀 Live Demo

[Try the live demo here](https://your-app-name.railway.app) (Coming soon)

## 📋 Features

- **Interactive Interface**: User-friendly Streamlit web app
- **Real-time Predictions**: Instant risk assessment based on input parameters
- **Professional Styling**: Clean, medical-themed UI design
- **Risk Analysis**: Detailed breakdown of risk factors
- **Educational**: Includes technical details and disclaimers

## 🔬 Technical Details

- **Model**: Random Forest Classifier (100 estimators)
- **Features**: 13 clinical parameters including:
  - Age, sex, chest pain type
  - Blood pressure, cholesterol levels
  - ECG results, exercise capacity
  - Thalassemia, vessel blockages
- **Accuracy**: ~94% on validation data
- **Framework**: Scikit-learn, Streamlit, Pandas

## 🏃‍♂️ Running Locally

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. **Open in browser**: Navigate to `http://localhost:8501`

## 🚀 Deployment

### Railway (Recommended)
1. Push to GitHub repository
2. Connect to [Railway.app](https://railway.app)
3. Deploy with one click

### Alternative Platforms
- **Render**: Free tier available
- **Heroku**: Add `setup.sh` and `Procfile`
- **Streamlit Cloud**: Direct deployment

## ⚠️ Important Notes

- **Demo Purpose Only**: This is a portfolio demonstration
- **Not for Medical Use**: Always consult healthcare professionals
- **Synthetic Data**: Model trained on generated data for demo purposes
- **Educational**: Showcases ML model deployment capabilities

## 🛠️ Model Architecture

```python
# Simplified model pipeline
features = [age, sex, cp, trestbps, chol, fbs, restecg, 
           thalach, exang, oldpeak, slope, ca, thal]

# Preprocessing
scaled_features = StandardScaler().transform(features)

# Prediction
risk_probability = RandomForestClassifier().predict_proba(scaled_features)
```

## 📊 Input Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| Age | Patient age | 20-100 years |
| Sex | Biological sex | Male/Female |
| Chest Pain | Type of chest pain | 4 categories |
| Blood Pressure | Resting BP | 80-200 mmHg |
| Cholesterol | Serum cholesterol | 100-400 mg/dl |
| Blood Sugar | Fasting glucose | >120 mg/dl |
| ECG | Resting ECG results | 3 categories |
| Heart Rate | Max exercise HR | 60-220 bpm |
| Angina | Exercise angina | Yes/No |
| ST Depression | Exercise ST depression | 0-6.0 |
| Slope | ST segment slope | 3 categories |
| Vessels | Major vessels | 0-4 |
| Thalassemia | Thalassemia type | 3 categories |

## 🔗 Portfolio Integration

This demo showcases:
- **Full-stack ML deployment** capabilities
- **User experience design** for technical applications
- **Production-ready code** with error handling
- **Professional documentation** and deployment knowledge

---

Built by **Daniel Park** | [Portfolio](https://danielpark-portfolio.com) | [GitHub](https://github.com/dSpringOnion)
