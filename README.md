# Heart Disease Risk Predictor

A production-grade machine learning application that predicts heart disease risk using the renowned Cleveland Heart Disease dataset from UCI ML Repository.

## 🚀 Live Demo

[Try the live demo here](https://heartdiseasepredictor-production.up.railway.app) (Deploying soon)

## 📋 Features

- **Research-Grade Model**: Trained on real clinical data from Cleveland Clinic Foundation
- **Interactive Interface**: Professional medical-themed web application  
- **Real-time Predictions**: Instant risk assessment with confidence scores
- **Feature Engineering**: Advanced preprocessing with risk factor analysis
- **Model Comparison**: Ensemble of Random Forest, XGBoost, and Logistic Regression
- **Production Ready**: Scalable deployment with comprehensive error handling

## 🔬 Technical Details

- **Dataset**: Cleveland Heart Disease Dataset (UCI ML Repository)
- **Model**: Best performing ensemble (Random Forest/XGBoost/Logistic Regression)
- **Features**: 17 clinical parameters including:
  - **Primary**: Age, sex, chest pain type, blood pressure, cholesterol
  - **Cardiac**: ECG results, max heart rate, exercise angina
  - **Advanced**: ST depression, vessel blockages, thalassemia
  - **Engineered**: Age groups, risk categories, composite indicators
- **Performance**: 85-90% accuracy on real clinical validation data
- **Framework**: Scikit-learn, XGBoost, Streamlit, Railway
- **Deployment**: Production containers with auto-scaling

## 🏃‍♂️ Running Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dSpringOnion/heartDiseasePredictor.git
   cd heartDiseasePredictor
   ```

2. **Train the model** (downloads real Cleveland dataset):
   ```bash
   ./run_training.sh
   # OR manually:
   pip install -r requirements.txt
   python train_model.py
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**: Navigate to `http://localhost:8501`

## 🚀 Deployment

### Railway (Production Platform)
1. Push to GitHub repository
2. Connect to [Railway.app](https://railway.app)
3. Automatic deployment with container orchestration
4. Custom domain and SSL certificate support

### Alternative Platforms
- **Render**: Good alternative with free tier
- **Heroku**: Classic choice with buildpacks
- **Streamlit Cloud**: Quick prototyping option

## ⚠️ Important Notes

- **Research Demonstration**: This showcases ML capabilities using real clinical data
- **Not for Medical Use**: Always consult qualified healthcare professionals for medical decisions
- **Educational Purpose**: Demonstrates production-grade ML pipeline development
- **Real Dataset**: Trained on legitimate Cleveland Heart Disease dataset from UCI

## 🛠️ Model Architecture

```python
# Production ML Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 1. Data Loading & Preprocessing
data = load_cleveland_dataset()  # Real UCI data
X, y = feature_engineering(data)  # 17 engineered features

# 2. Model Training & Selection
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'XGBoost': xgb.XGBClassifier(),
    'LogisticRegression': LogisticRegression()
}

# 3. Cross-validation & Best Model Selection
best_model = select_best_model(models, X, y)

# 4. Production Inference
risk_probability = best_model.predict_proba(scaled_features)[0][1]
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

## 📚 Dataset Information

**Source**: UCI Machine Learning Repository  
**Dataset**: Heart Disease Dataset (Cleveland database)  
**Citation**: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

**Key Statistics**:
- 303 patient records from Cleveland Clinic Foundation
- 14 attributes (13 features + target)
- Widely cited in ML research (1000+ citations)
- Benchmark dataset for heart disease prediction

## 🔗 Portfolio Integration

This project demonstrates:
- **Research-grade ML development** with real clinical data
- **Production deployment pipeline** from data to web application
- **Advanced feature engineering** and model selection
- **Professional software development** practices
- **Healthcare ML expertise** for medical applications

**Perfect for**: Data Science, ML Engineering, and Healthcare Technology roles

---

Built by **Daniel Park** | [Portfolio](https://danielpark-portfolio.com) | [GitHub](https://github.com/dSpringOnion)
