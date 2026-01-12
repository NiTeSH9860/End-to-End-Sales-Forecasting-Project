import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
import joblib
import json

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.metrics = {}
        
    def prepare_data(self, df):
        """Preparing data for time series forecasting"""
        target_col = self.config['model']['target_column']
        date_col = self.config['model']['date_column']
        
        # Separating features and target
        X = df.drop(columns=[target_col, date_col], errors='ignore')
        y = df[target_col]
        
        # For time series, we need to maintain temporal order
        split_idx = int(len(df) * (1 - self.config['model']['test_size']))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, experiment_name="xgboost"):
        """Train XGBoost model with MLflow tracking"""
        
        with mlflow.start_run(run_name=f"xgboost_{experiment_name}"):
            # Log parameters
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.config['model']['random_state']
            }
            
            mlflow.log_params(params)
            
            # Training model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Making predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculating metrics
            metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            # Saving feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log feature importance as artifact
            temp_path = "temp_feature_importance.csv"
            feature_importance.to_csv(temp_path, index=False)
            mlflow.log_artifact(temp_path)
            
            self.models['xgboost'] = model
            self.metrics['xgboost'] = metrics
            
            print(f"XGBoost training complete. Test RMSE: {metrics['test_rmse']:.2f}")
            
            return model, metrics
    
    def train_prophet(self, df, experiment_name="prophet"):
        """Train Facebook Prophet model"""
        
        with mlflow.start_run(run_name=f"prophet_{experiment_name}"):
            # Preparing data for Prophet
            prophet_df = df[[self.config['model']['date_column'], 
                           self.config['model']['target_column']]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Splitting data
            split_idx = int(len(prophet_df) * (1 - self.config['model']['test_size']))
            train_df = prophet_df.iloc[:split_idx]
            test_df = prophet_df.iloc[split_idx:]
            
            # Creating and training model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            # Adding additional regressors if available
            if 'Region' in df.columns:
                model.add_regressor('Region')
            if 'Product_Category' in df.columns:
                model.add_regressor('Product_Category')
            
            model.fit(train_df)
            
            # Making predictions
            future = model.make_future_dataframe(periods=len(test_df))
            forecast = model.predict(future)
            
            # Calculating metrics
            y_true = test_df['y'].values
            y_pred = forecast.iloc[split_idx:]['yhat'].values
            
            metrics = {
                'test_mae': mean_absolute_error(y_true, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'test_r2': r2_score(y_true, y_pred)
            }
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.prophet.log_model(model, "model")
            
            self.models['prophet'] = model
            self.metrics['prophet'] = metrics
            
            print(f"Prophet training complete. Test RMSE: {metrics['test_rmse']:.2f}")
            
            return model, metrics
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train ensemble of models"""
        # Training multiple models
        xgb_model, xgb_metrics = self.train_xgboost(X_train, y_train, X_test, y_test)
        
       
        
        return xgb_model, xgb_metrics
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """Calculating performance metrics"""
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        return metrics
    
    def save_model(self, model, model_name, path=None):
        """Saving trained model"""
        if path is None:
            path = f"data/models/{model_name}.pkl"
        
        joblib.dump(model, path)
        print(f"Model saved to: {path}")
        
        # Also saving metrics
        metrics_path = f"data/models/{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics.get(model_name, {}), f)
        
        return path
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning with MLflow tracking"""
        import optuna
        
        def objective(trial):
            with mlflow.start_run(nested=True):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
                }
                
                mlflow.log_params(params)
                
                model = xgb.XGBRegressor(**params, random_state=self.config['model']['random_state'])
                
                # Use time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_train_cv, y_train_cv)
                    y_pred = model.predict(X_val_cv)
                    cv_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred)))
                
                avg_rmse = np.mean(cv_scores)
                mlflow.log_metric('cv_rmse', avg_rmse)
                
                return avg_rmse
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        print(f"Best trial: {study.best_trial.params}")
        print(f"Best CV RMSE: {study.best_trial.value}")
        
        return study.best_params