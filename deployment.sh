#!/bin/bash

echo "🚀 Deploying Sales Forecasting System..."

# Creating virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Installing dependencies
pip install -r requirements.txt

# Running training
echo "📊 Training model..."
python train_model.py

# Starting MLflow UI (in background)
echo "📈 Starting MLflow UI..."
mlflow ui --backend-store-uri file:///mlflow --host 0.0.0.0 --port 5001 &

# Starting Flask app
echo "🌐 Starting Flask application..."
python app/app.py