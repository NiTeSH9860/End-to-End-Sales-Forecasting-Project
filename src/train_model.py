import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.model_training import SalesForecastModel
import pandas as pd
import joblib

def main():
    print("🚀 Starting Sales Forecasting Pipeline...")
    
    # Initializing components
    preprocessor = DataPreprocessor()
    model_trainer = SalesForecastModel()
    
    # Loading and preprocessing data
    print("\n📊 Loading and preprocessing data...")
    df_clean, features = preprocessor.full_pipeline(
        "sales_data.csv",
        target_product=1055 # Focusing on a specific product
    )
    
    print(f"Processed data shape: {df_clean.shape}")
    print(f"Number of features: {len(features)}")
    
    # Preparing data for training
    X_train, X_test, y_train, y_test, split_idx = model_trainer.prepare_data(
        df_clean, features, test_size=0.2
    )
    
    print(f"\n📈 Training samples: {X_train.shape[0]}")
    print(f"📈 Testing samples: {X_test.shape[0]}")
    
    # Training model
    print("\n🤖 Training XGBoost model...")
    model, metrics = model_trainer.train_model(X_train, y_train, X_test, y_test)
    
    # Performing cross-validation
    print("\n🔍 Performing cross-validation...")
    X_full = df_clean[features].values
    y_full = df_clean['Sales_Amount'].values
    cv_scores = model_trainer.cross_validate(X_full, y_full, n_splits=3)
    
    # Feature importance analysis
    print("\n🎯 Analyzing feature importance...")
    feat_imp = model_trainer.feature_importance_analysis(features)
    
    # Saving model
    print("\n💾 Saving model...")
    model_trainer.save_model()
    
    # Generating sample predictions
    print("\n🔮 Generating sample predictions...")
    sample_predictions = model_trainer.predict_future(df_clean, features, future_days=7)
    
    print("\n📅 Next 7 days predictions:")
    for i, pred in enumerate(sample_predictions, 1):
        print(f"  Day {i}: ${pred:.2f}")
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📊 Final RMSE: {metrics['RMSE']:.2f}")
    print(f"📊 Final R²: {metrics['R2']:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()