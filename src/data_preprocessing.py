#importing necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.df = None
        
    def load_data(self, filepath=None):
        """Load raw sales data"""
        if filepath is None:
            filepath = self.config['data']['raw_path']
        
        self.df = pd.read_csv(filepath)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def preprocess_sales_data(self):
        """Clean and preprocess the sales data"""
        # Converting date column
        self.df['Sale_Date'] = pd.to_datetime(self.df['Sale_Date'])
        
        # Sorting by date
        self.df = self.df.sort_values('Sale_Date').reset_index(drop=True)
        
        # Handling missing values
        self.df = self._handle_missing_values()
        
        # Calculating derived metrics
        self.df['Profit'] = self.df['Sales_Amount'] - (self.df['Unit_Cost'] * self.df['Quantity_Sold'])
        self.df['Profit_Margin'] = self.df['Profit'] / self.df['Sales_Amount']
        
        # Creating time-based features
        self.df = self._create_time_features()
        
        # Encoding categorical variables
        self.df = self._encode_categorical()
        
        return self.df
    
    def _handle_missing_values(self):
        """Handl missing values in the dataset"""
        # Forward fill for time series
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(method='ffill')
        
        # Filling remaining with median
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        return self.df
    
    def _create_time_features(self):
        """Create time-based features"""
        df = self.df.copy()
        
        # Basic time features
        df['Year'] = df['Sale_Date'].dt.year
        df['Month'] = df['Sale_Date'].dt.month
        df['Week'] = df['Sale_Date'].dt.isocalendar().week
        df['Day'] = df['Sale_Date'].dt.day
        df['DayOfWeek'] = df['Sale_Date'].dt.dayofweek
        df['DayOfYear'] = df['Sale_Date'].dt.dayofyear
        df['Quarter'] = df['Sale_Date'].dt.quarter
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsMonthStart'] = df['Sale_Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Sale_Date'].dt.is_month_end.astype(int)
        
        # Cyclical encoding for periodic features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        
        return df
    
    def _encode_categorical(self):
        """Encode categorical variables"""
        df = self.df.copy()
        
        # One-hot encode nominal categorical variables
        categorical_cols = ['Region', 'Customer_Type', 'Payment_Method', 'Sales_Channel']
        df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        
        return df
    
    def save_processed_data(self, path=None):
        """Save processed data to file"""
        if path is None:
            path = self.config['data']['processed_path']
        
        self.df.to_csv(path, index=False)
        print(f"Processed data saved to: {path}")