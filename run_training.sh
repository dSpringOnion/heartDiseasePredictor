#!/bin/bash

echo "🏥 Training Heart Disease Prediction Model with Real Data"
echo "========================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Install requirements if not already installed
echo "📦 Installing required packages..."
pip install -r requirements.txt

# Run the training script
echo "🚀 Starting model training..."
python3 train_model.py

# Check if model files were created
if [ -f "heart_disease_model.pkl" ] && [ -f "feature_scaler.pkl" ]; then
    echo "✅ Model training completed successfully!"
    echo "📁 Generated files:"
    echo "   - heart_disease_model.pkl"
    echo "   - feature_scaler.pkl"
    echo "   - feature_names.txt"
    echo "   - model_metadata.json"
    echo ""
    echo "🚀 Now you can run the Streamlit app:"
    echo "   streamlit run app.py"
else
    echo "❌ Model training failed. Check the output above for errors."
    exit 1
fi