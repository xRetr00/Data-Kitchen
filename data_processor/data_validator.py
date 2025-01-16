"""
Data validation module for Data Kitchen
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
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
            'constant_threshold': 0.05,  # Minimum unique ratio for non-constant columns
            'outlier_threshold': 3.0,  # Number of standard deviations for outlier detection
            'future_window': 0,  # Number of future periods to check for leakage
            'min_periods': 20,  # Minimum number of periods required
        }
        self.scaler = RobustScaler()
        
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate entire dataset and return cleaned data with validation report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple containing:
            - Cleaned DataFrame
            - Validation report dictionary
        """
        if len(df) < self.config['min_periods']:
            raise DataValidationError(f"Dataset too small: {len(df)} rows < {self.config['min_periods']} required")
        
        report = {
            'original_shape': df.shape,
            'dropped_columns': [],
            'outliers_detected': False
        }
        
        try:
            # Handle missing values
            df, missing_issues = self._handle_missing_values(df)
            report['dropped_columns'].extend([col for col in missing_issues if 'Removed column' in col])
            
            # Handle duplicates
            df, duplicate_issues = self._handle_duplicates(df)
            
            # Handle constant columns
            df, constant_issues = self._handle_constant_columns(df)
            report['dropped_columns'].extend([col.split(': ')[1].split(' ')[0] for col in constant_issues])
            
            # Detect and handle outliers
            df, outlier_report = self._handle_outliers(df)
            report['outliers_detected'] = bool(outlier_report['outliers_detected'])
            
            # Check for future data leaks
            future_leaks = self._check_future_leaks(df)
            
            report.update({
                'final_shape': df.shape,
                'missing_data_issues': missing_issues,
                'duplicate_issues': duplicate_issues,
                'constant_column_issues': constant_issues,
                'future_data_leaks': future_leaks
            })
        
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
        df = df.ffill()
        # Backward fill any remaining NaNs at the start
        df = df.bfill()
        
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
        """Handle constant columns"""
        issues = []
        constant_cols = []
        
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < self.config['constant_threshold']:
                constant_cols.append(col)
                issues.append(f"Removed constant column: {col} (unique ratio: {unique_ratio:.2f})")
        
        if constant_cols:
            df = df.drop(columns=constant_cols)
        
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

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """معالجة البيانات المفقودة"""
        if df.empty:
            logger.error("Error handling missing data: Cannot handle missing data on empty DataFrame")
            return df

        try:
            # حساب نسبة القيم المفقودة لكل عمود
            missing_ratio = df.isnull().sum() / len(df)
            
            # حذف الأعمدة التي تحتوي على نسبة عالية من القيم المفقودة
            columns_to_drop = missing_ratio[missing_ratio > 0.5].index
            if not columns_to_drop.empty:
                logger.warning(f"Dropping columns with high missing value proportion: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop)
            
            # ملء القيم المفقودة المتبقية
            df = df.ffill()  # Forward fill
            df = df.bfill()  # Backward fill
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing data: {str(e)}")
            return df
