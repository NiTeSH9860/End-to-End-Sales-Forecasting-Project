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
import shutil

class SalesForecastModel:
    def __init__(self, experiment_name="sales_forecast", reset_experiment=False):
        self.model = None
        self.scaler = None
        self.experiment_name = experiment_name
        
        if reset_experiment:
            self.reset_mlflow()
        
        self.setup_mlflow()
        
    def reset_mlflow(self):
        """Reset MLflow tracking directory"""
        if os.path.exists("mlflow"):
            print("⚠️ Resetting MLflow directory...")
            shutil.rmtree("mlflow")
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        # Creating local MLflow tracking directory
        os.makedirs("mlflow", exist_ok=True)
        
        # Setting tracking URI first
        mlflow.set_tracking_uri("file:///mlflow")
        
        # Checking if experiment exists, create if not
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                print(f"📁 Creating new experiment: {self.experiment_name}")
                mlflow.create_experiment(self.experiment_name, artifact_location="mlflow")
        except:
            print(f"📁 Creating new experiment: {self.experiment_name}")
            mlflow.create_experiment(self.experiment_name, artifact_location="mlflow")
        
        # Setting experiment
        mlflow.set_experiment(self.experiment_name)
        
        print("✅ MLflow setup completed")
        print(f"📊 Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"🔬 Experiment: {self.experiment_name}")
    
    def prepare_data(self, df, features, target='Sales_Amount', test_size=0.2):
        """Prepare train-test split for time series"""
        # Time-based split (important for time series)
        split_idx = int(len(df) * (1 - test_size))
        
        X = df[features].values
        y = df[target].values
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📈 Training samples: {len(X_train)}")
        print(f"📊 Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, split_idx
    
    def train_model(self, X_train, y_train, X_test, y_test, params=None, run_name=None):
        """Train XGBoost model with MLflow tracking"""
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'  # This goes in model params, not fit()
            }
        
        if run_name is None:
            run_name = f"xgboost_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            print(f"🚀 Starting run: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_params(params)
            
            # Log tags
            mlflow.set_tag("model_type", "xgboost")
            mlflow.set_tag("problem_type", "time_series_forecast")
            mlflow.set_tag("framework", "xgboost")
            
            # Training model 
            print("🤖 Training XGBoost model...")
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=100 
            )
            
        
            # Making predictions
            y_pred = self.model.predict(X_test)
            
            # Calculating metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1e-10, None))) * 100
            
            # Log metrics
            metrics = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape
            }
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.xgboost.log_model(self.model, "model")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': range(X_train.shape[1]),
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Saving feature importance as artifact
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            os.remove("feature_importance.csv")
            
            # Log eval metrics from training
            try:
                eval_results = self.model.evals_result()
                for epoch, rmse_val in enumerate(eval_results['validation_0']['rmse']):
                    mlflow.log_metric("train_rmse", rmse_val, step=epoch)
            except:
                pass
            
            print("\n✅ Model training completed!")
            print(f"📊 Run ID: {run.info.run_id}")
            print(f"📈 MAE: {mae:.2f}")
            print(f"📈 RMSE: {rmse:.2f}")
            print(f"📈 R²: {r2:.4f}")
            print(f"📈 MAPE: {mape:.2f}%")
            
            return self.model, metrics
    
    def cross_validate(self, X, y, params=None, n_splits=5):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'random_state': 42
            }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_scores.append(rmse)
            
            print(f"Validation RMSE: {rmse:.2f}")
        
        print(f"\nCross-validation RMSE: {np.mean(cv_scores):.2f} (±{np.std(cv_scores):.2f})")
        return cv_scores
    
    def save_model(self, path="models/xgboost_model.pkl"):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="models/xgboost_model.pkl"):
        """Load trained model"""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model
    
    def predict_future(self, df, features, future_days=30):
        """Generate future forecasts"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Getting last known data point
        if len(df) == 0:
            raise ValueError("No data provided")
        
        # Creating a copy of the last row's features
        last_features = df[features].iloc[-1:].copy()
        
        predictions = []
        for i in range(future_days):
            # Making prediction
            pred = self.model.predict(last_features)[0]
            predictions.append(pred)
            
            # Updating the lag features for next prediction
           
            if i < future_days - 1:
                # Shift lag features
                for col in features:
                    if 'lag_1' in col:
                        last_features[col] = pred
        
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
        
        print("\n" + "="*50)
        print("TOP 10 MOST IMPORTANT FEATURES:")
        print("="*50)
        print(feat_imp.head(10).to_string(index=False))
        print("="*50)
        
        return feat_imp
    
    def get_model_summary(self):
        """Get model configuration summary"""
        if self.model is None:
            return "Model not trained"
        
        summary = {
            'n_features': self.model.n_features_in_,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'booster': self.model.booster
        }
        
        return summary