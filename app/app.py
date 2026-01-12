from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import plotly
import plotly.graph_objs as go
import json

app = Flask(__name__)

# Load ingmodels
MODEL_PATH = "models/xgboost_model.pkl"
SCALER_PATH = "models/scaler.pkl"

class SalesForecastAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
    
    def prepare_input_features(self, input_data):
        """Prepare input features for prediction"""
        
    
        return input_data
    
    def predict(self, features):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        prediction = self.model.predict(features)
        return prediction[0]

# Initializing API
forecast_api = SalesForecastAPI()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': forecast_api.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        
        # Preparing features from input data
        features = forecast_api.prepare_input_features(data)
        
        # Making prediction
        prediction = forecast_api.predict(features)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'currency': 'USD',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Generate future forecasts"""
    try:
        data = request.json
        days = data.get('days', 30)
        product_id = data.get('product_id', None)
        
        # Generating forecast (simplified)
        
        base_value = 5000  # Example base value
        forecast_data = []
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i+1)
            # Simple trend + seasonality + noise
            value = base_value * (1 + 0.01 * i) + 1000 * np.sin(2 * np.pi * i / 7)
            forecast_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_sales': float(value),
                'lower_bound': float(value * 0.9),
                'upper_bound': float(value * 1.1)
            })
        
        # Creating visualization
        fig = go.Figure()
        
        dates = [item['date'] for item in forecast_data]
        preds = [item['predicted_sales'] for item in forecast_data]
        lower = [item['lower_bound'] for item in forecast_data]
        upper = [item['upper_bound'] for item in forecast_data]
        
        fig.add_trace(go.Scatter(
            x=dates, y=preds,
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='royalblue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(65, 105, 225, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title='Sales Forecast (Next 30 Days)',
            xaxis_title='Date',
            yaxis_title='Sales Amount ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'forecast': forecast_data,
            'plot': graphJSON,
            'summary': {
                'avg_prediction': np.mean(preds),
                'total_prediction': np.sum(preds),
                'growth_rate': ((preds[-1] - preds[0]) / preds[0]) * 100
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if forecast_api.model is None:
        return jsonify({'error': 'Model not loaded'}), 404
    
    return jsonify({
        'model_type': type(forecast_api.model).__name__,
        'features_used': forecast_api.model.n_features_in_ if hasattr(forecast_api.model, 'n_features_in_') else 'Unknown',
        'training_date': datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat() if os.path.exists(MODEL_PATH) else 'Unknown'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)