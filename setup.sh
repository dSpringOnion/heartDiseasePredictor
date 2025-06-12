#!/bin/bash

echo "🏥 Setting up Heart Disease Predictor Demo..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv heart-disease-env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source heart-disease-env/bin/activate

# Install requirements
echo "⬇️ Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "1. source heart-disease-env/bin/activate"
echo "2. streamlit run app.py"
echo ""
echo "🚀 The app will open at http://localhost:8501"