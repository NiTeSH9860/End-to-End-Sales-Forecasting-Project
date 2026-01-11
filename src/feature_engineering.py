import pandas as pd
import numpy as np
from typing import List, Dict

class TimeSeriesFeatureEngineer:
    def __init__(self, config):
        self.config = config
        
    def create_lag_features(self, df: pd.DataFrame, group_cols: List[str] = None) -> pd.DataFrame:
        """
        Create lag features for time series forecasting
        """
        df = df.copy()
        target_col = self.config['model']['target_column']
        date_col = self.config['model']['date_column']
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # If grouping by certain columns (e.g., region, product category)
        if group_cols:
            for col in group_cols:
                if col in df.columns:
                    for lag in self.config['features']['lag_features']:
                        df[f'{target_col}_lag_{lag}_{col}'] = df.groupby(col)[target_col].shift(lag)
        else:
            # Global lag features
            for lag in self.config['features']['lag_features']:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, group_cols: List[str] = None) -> pd.DataFrame:
        """
        Create rolling window statistics
        """
        df = df.copy()
        target_col = self.config['model']['target_column']
        date_col = self.config['model']['date_column']
        
        df = df.sort_values(date_col)
        
        # Defining aggregation functions
        agg_funcs = ['mean', 'std', 'min', 'max', 'median']
        
        if group_cols:
            for col in group_cols:
                if col in df.columns:
                    for window in self.config['features']['rolling_windows']:
                        for agg_func in agg_funcs:
                            df[f'{target_col}_rolling_{window}_{agg_func}_{col}'] = df.groupby(col)[target_col]\
                                .transform(lambda x: x.rolling(window=window, min_periods=1).agg(agg_func))
        else:
            for window in self.config['features']['rolling_windows']:
                for agg_func in agg_funcs:
                    df[f'{target_col}_rolling_{window}_{agg_func}'] = df[target_col]\
                        .rolling(window=window, min_periods=1).agg(agg_func)
        
        return df
    
    def create_expanding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create expanding window features
        """
        df = df.copy()
        target_col = self.config['model']['target_column']
        date_col = self.config['model']['date_column']
        
        df = df.sort_values(date_col)
        
        # Expanding window statistics
        df[f'{target_col}_expanding_mean'] = df[target_col].expanding().mean()
        df[f'{target_col}_expanding_std'] = df[target_col].expanding().std()
        df[f'{target_col}_expanding_min'] = df[target_col].expanding().min()
        df[f'{target_col}_expanding_max'] = df[target_col].expanding().max()
        
        return df
    
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creating advanced date-based features
        """
        df = df.copy()
        date_col = self.config['model']['date_column']
        
        if date_col in df.columns:
            # Business days in month
            df['BusinessDaysInMonth'] = df[date_col].apply(
                lambda x: np.busday_count(x.replace(day=1), 
                                         (x.replace(day=1) + pd.offsets.MonthEnd(1)).replace(day=1))
            )
            
            # Days until next holiday (simplified)
            df['DaysUntilMonthEnd'] = (df[date_col] + pd.offsets.MonthEnd(0)) - df[date_col]
            df['DaysUntilMonthEnd'] = df['DaysUntilMonthEnd'].dt.days
            
            # Seasonality features
            df['Season'] = df['Month'].apply(self._get_season)
            
        return df
    
    def _get_season(self, month):
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creating interaction features between important variables
        """
        df = df.copy()
        
        # Example interaction features
        if 'Quantity_Sold' in df.columns and 'Unit_Price' in df.columns:
            df['Quantity_Price_Interaction'] = df['Quantity_Sold'] * df['Unit_Price']
        
        if 'Discount' in df.columns and 'Sales_Amount' in df.columns:
            df['Discount_Impact'] = df['Discount'] * df['Sales_Amount']
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appling all feature engineering steps
        """
        print("Starting feature engineering...")
        
        # Original shape
        original_shape = df.shape
        print(f"Original shape: {original_shape}")
        
        # Creating lag features
        print("Creating lag features...")
        df = self.create_lag_features(df, group_cols=['Region', 'Product_Category'])
        
        # Creating rolling features
        print("Creating rolling features...")
        df = self.create_rolling_features(df, group_cols=['Region', 'Product_Category'])
        
        # Creating expanding features
        print("Creating expanding features...")
        df = self.create_expanding_features(df)
        
        # Creating date features
        print("Creating date features...")
        df = self.create_date_features(df)
        
        # Creating interaction features
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Handling NaN values from lag/rolling features
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        print(f"Feature engineering complete. New shape: {df.shape}")
        print(f"Added {df.shape[1] - original_shape[1]} new features")
        
        return df