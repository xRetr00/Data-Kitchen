"""
Data validation module for Data Kitchen
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .logger import setup_logger
from sklearn.preprocessing import RobustScaler
import warnings

logger = setup_logger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DataValidator:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataValidator
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {
            'nan_threshold': 0.1,  # Maximum allowed proportion of NaN values
            'duplicate_threshold': 0.01,  # Maximum allowed proportion of duplicates
            'constant_threshold': 0.95,  # Minimum variance for non-constant columns
            'outlier_threshold': 3.0,  # Number of standard deviations for outlier detection
            'future_window': 0,  # Number of future periods to check for leakage
            'min_periods': 20,  # Minimum number of periods required
        }
        self.scaler = RobustScaler()
        
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate entire dataset and return cleaned data with validation report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple containing:
            - Cleaned DataFrame
            - Validation report dictionary
        """
        report = {
            'original_shape': df.shape,
            'issues': [],
            'dropped_rows': 0,
            'dropped_columns': [],
            'outliers_detected': {},
            'future_leaks': [],
        }
        
        try:
            # Check minimum size
            if len(df) < self.config['min_periods']:
                raise DataValidationError(f"Dataset too small: {len(df)} rows < {self.config['min_periods']} required")
            
            # Handle missing values
            df, nan_report = self._handle_missing_values(df)
            report['issues'].extend(nan_report)
            
            # Remove duplicates
            df, dup_report = self._handle_duplicates(df)
            report['issues'].extend(dup_report)
            
            # Check for constant columns
            df, const_report = self._handle_constant_columns(df)
            report['issues'].extend(const_report)
            
            # Detect and handle outliers
            df, outlier_report = self._handle_outliers(df)
            report.update(outlier_report)
            
            # Check for future data leaks
            future_leaks = self._check_future_leaks(df)
            report['future_leaks'] = future_leaks
            
            # Update final statistics
            report['final_shape'] = df.shape
            report['dropped_rows'] = report['original_shape'][0] - report['final_shape'][0]
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise DataValidationError(f"Validation failed: {str(e)}")
        
        return df, report
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values in the dataset"""
        issues = []
        
        # Calculate missing value proportions
        missing_props = df.isnull().mean()
        high_missing = missing_props[missing_props > self.config['nan_threshold']]
        
        if not high_missing.empty:
            # Drop columns with too many missing values
            df = df.drop(columns=high_missing.index)
            issues.extend([f"Dropped column {col} ({prop:.2%} missing)" 
                         for col, prop in high_missing.items()])
        
        # Forward fill remaining missing values
        df = df.fillna(method='ffill')
        # Backward fill any remaining NaNs at the start
        df = df.fillna(method='bfill')
        
        return df, issues
    
    def _handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle duplicate rows in the dataset"""
        issues = []
        
        # Check for duplicates
        dup_prop = df.duplicated().mean()
        if dup_prop > self.config['duplicate_threshold']:
            original_len = len(df)
            df = df.drop_duplicates()
            dropped = original_len - len(df)
            issues.append(f"Removed {dropped} duplicate rows ({dup_prop:.2%} of data)")
        
        return df, issues
    
    def _handle_constant_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Identify and handle constant or near-constant columns"""
        issues = []
        
        # Calculate normalized variance for each column
        variances = df.var() / df.mean()
        low_var_cols = variances[variances < (1 - self.config['constant_threshold'])]
        
        if not low_var_cols.empty:
            df = df.drop(columns=low_var_cols.index)
            issues.extend([f"Dropped near-constant column {col}" for col in low_var_cols.index])
        
        return df, issues
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers using robust scaling"""
        report = {'outliers_detected': {}}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_data = self.scaler.fit_transform(df[[col]])
            
            # Detect outliers
            outliers = np.abs(scaled_data) > self.config['outlier_threshold']
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                report['outliers_detected'][col] = int(n_outliers)
                # Cap outliers at threshold value
                df.loc[outliers.ravel(), col] = df[col].mean() + (
                    df[col].std() * self.config['outlier_threshold'] * 
                    np.sign(scaled_data[outliers])
                )
        
        return df, report
    
    def _check_future_leaks(self, df: pd.DataFrame) -> List[str]:
        """Check for potential future data leaks"""
        future_leaks = []
        
        if self.config['future_window'] > 0:
            # Check for shifted values
            for col in df.columns:
                if any(col.startswith(prefix) for prefix in ['future_', 'next_', 'forward_']):
                    future_leaks.append(f"Column name suggests future data: {col}")
                
                # Check for forward-looking calculations
                if df[col].shift(-self.config['future_window']).corr(df[col]) > 0.95:
                    future_leaks.append(f"Possible future leak in column: {col}")
        
        return future_leaks
