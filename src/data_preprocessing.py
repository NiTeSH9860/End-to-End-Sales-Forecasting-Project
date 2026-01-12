import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.date_format = "%m/%d/%Y"
        
    def load_data(self, file_path):
        """Loading sales data from CSV"""
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def preprocess_dates(self, df):
        """Converting and extracting date features"""
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], format=self.date_format)
        
        # Extracting temporal features
        df['Year'] = df['Sale_Date'].dt.year
        df['Month'] = df['Sale_Date'].dt.month
        df['Quarter'] = df['Sale_Date'].dt.quarter
        df['Week'] = df['Sale_Date'].dt.isocalendar().week
        df['Day'] = df['Sale_Date'].dt.day
        df['DayOfWeek'] = df['Sale_Date'].dt.dayofweek
        df['DayOfYear'] = df['Sale_Date'].dt.dayofyear
        
        return df
    
    def create_lag_features(self, df, product_id, lags=[1, 7, 30, 90]):
        """Create lag features for time series forecasting"""
        df = df.sort_values(['Product_ID', 'Sale_Date'])
        
        for lag in lags:
            df[f'Sales_Amount_lag_{lag}'] = df.groupby('Product_ID')['Sales_Amount'].shift(lag)
            df[f'Quantity_Sold_lag_{lag}'] = df.groupby('Product_ID')['Quantity_Sold'].shift(lag)
        
        # Rolling statistics
        df['Sales_Amount_rolling_mean_7'] = df.groupby('Product_ID')['Sales_Amount'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['Sales_Amount_rolling_std_7'] = df.groupby('Product_ID')['Sales_Amount'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        categorical_cols = ['Sales_Rep', 'Region', 'Product_Category', 
                          'Customer_Type', 'Payment_Method', 'Sales_Channel']
        
        # One-hot encoding for low cardinality features
        for col in ['Region', 'Customer_Type', 'Payment_Method', 'Sales_Channel']:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
        
        # Target encoding for high cardinality features
        rep_sales = df.groupby('Sales_Rep')['Sales_Amount'].mean().to_dict()
        df['Sales_Rep_encoded'] = df['Sales_Rep'].map(rep_sales)
        
        cat_sales = df.groupby('Product_Category')['Sales_Amount'].mean().to_dict()
        df['Product_Category_encoded'] = df['Product_Category'].map(cat_sales)
        
        return df
    
    def engineer_features(self, df):
        """Create additional business features"""
        # Financial features
        df['Revenue'] = df['Sales_Amount'] * df['Quantity_Sold']
        df['Profit_Margin'] = (df['Unit_Price'] - df['Unit_Cost']) / df['Unit_Price']
        df['Discount_Amount'] = df['Sales_Amount'] * df['Discount']
        df['Net_Revenue'] = df['Sales_Amount'] - df['Discount_Amount']
        
        # Interaction features
        df['High_Value_Sale'] = (df['Sales_Amount'] > df['Sales_Amount'].median()).astype(int)
        df['Bulk_Purchase'] = (df['Quantity_Sold'] > df['Quantity_Sold'].median()).astype(int)
        
        # Seasonality features
        df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['Is_Month_End'] = (df['Day'] >= 25).astype(int)
        df['Is_Quarter_End'] = ((df['Month'] % 3 == 0) & (df['Day'] >= 25)).astype(int)
        
        return df
    
    def prepare_time_series(self, df, target_product=None, forecast_days=30):
        """Prepare data for time series forecasting"""
        if target_product:
            df = df[df['Product_ID'] == target_product]
        
        # Sorting by date
        df = df.sort_values('Sale_Date')
        
        # Creating time index
        df['Time_Index'] = range(len(df))
        
        # Preparing features and target
        features = [col for col in df.columns if col not in 
                   ['Product_ID', 'Sale_Date', 'Sales_Amount', 'Region_and_Sales_Rep', 
                    'Sales_Rep', 'Product_Category']]
        
        # Removing rows with NaN values from lag features
        df_clean = df.dropna(subset=features + ['Sales_Amount'])
        
        return df_clean, features
    
    def full_pipeline(self, file_path, target_product=None):
        """Complete preprocessing pipeline"""
        df = self.load_data(file_path)
        df = self.preprocess_dates(df)
        df = self.create_lag_features(df, target_product)
        df = self.encode_categorical(df)
        df = self.engineer_features(df)
        df_clean, features = self.prepare_time_series(df, target_product)
        
        return df_clean, features