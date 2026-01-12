import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os

class SalesForecastModel:
    def __init__(self, experiment_name="sales_forecast"):
        self.model = None
        self.scaler = None
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_experiment(self.experiment_name)
        
        # Creating local MLflow tracking directory
        if not os.path.exists("mlflow"):
            os.makedirs("mlflow")
        
        mlflow.set_tracking_uri("file:///mlflow")
    
    def prepare_data(self, df, features, target='Sales_Amount', test_size=0.2):
        """Prepare train-test split for time series"""
        # Time-based split (important for time series)
        split_idx = int(len(df) * (1 - test_size))
        
        X = df[features].values
        y = df[target].values
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, split_idx
    
    def train_model(self, X_train, y_train, X_test, y_test, params=None):
        """Train XGBoost model with MLflow tracking"""
        if params is None:
            params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            
            # Training model
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)],
                          verbose=False)
            
            # Making predictions
            y_pred = self.model.predict(X_test)
            
            # Calculating metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            
            # Log model
            mlflow.xgboost.log_model(self.model, "xgboost_model")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': range(X_train.shape[1]),
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(feature_importance.to_csv(), "feature_importance.csv")
            
            print(f"Model trained successfully!")
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            return self.model, {"MAE": mae, "RMSE": rmse, "R2": r2}
    
    def cross_validate(self, X, y, params=None, n_splits=5):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**(params or {}))
            model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_scores.append(rmse)
            
            print(f"Validation RMSE: {rmse:.2f}")
        
        print(f"\nCross-validation RMSE: {np.mean(cv_scores):.2f} (±{np.std(cv_scores):.2f})")
        return cv_scores
    
    def save_model(self, path="models/xgboost_model.pkl"):
        """Save trained model"""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="models/xgboost_model.pkl"):
        """Load trained model"""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model
    
    def predict_future(self, df, features, future_days=30):
        """Generate future forecasts"""
        # Use last known data point
        last_data = df[features].iloc[-1:].values
        
        predictions = []
        for i in range(future_days):
            # Making prediction
            pred = self.model.predict(last_data)[0]
            predictions.append(pred)
            
            # Updating features for next prediction (simplified)
            
            if i < len(predictions) - 1:
                last_data[0, 0] = predictions[-2] if i > 0 else pred
        
        return predictions
    
    def feature_importance_analysis(self, feature_names):
        """Analyze and display feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feat_imp.head(10).to_string(index=False))
        
        return feat_imp