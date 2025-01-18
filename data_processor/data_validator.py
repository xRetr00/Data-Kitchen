"""
Data Validator Module
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .logger import setup_logger

logger = setup_logger(__name__)

class DataValidator:
    """Class for validating and cleaning data"""
    
    def __init__(self, missing_threshold: float = 0.35, min_data_points: int = 50):
        """
        Initialize DataValidator
        
        Args:
            missing_threshold: Maximum allowed ratio of missing data (default: 0.35)
            min_data_points: Minimum number of data points required (default: 50)
        """
        self.missing_threshold = missing_threshold
        self.min_data_points = min_data_points
        logger.info(f"Data Validator initialized with missing_threshold={missing_threshold}, min_data_points={min_data_points}")
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            # Check for minimum sequence length
            if len(df) < self.min_data_points:
                return False, "Sequence too short"
                
            # Check for missing values
            missing_ratio = df.isnull().sum() / len(df)
            if missing_ratio.max() > self.missing_threshold:
                return False, f"Too many missing values: {missing_ratio.max():.2%}"
                
            # Check for variance
            for col in df.columns:
                if col != 'timestamp':
                    variance = df[col].var()
                    if variance < 1e-10:
                        return False, f"Low variance in {col}: {variance}"
                        
            # Check for data consistency
            if not self._check_data_consistency(df):
                return False, "Data consistency check failed"
                
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
            
    def _check_data_consistency(self, df: pd.DataFrame) -> bool:
        """
        Check data consistency
        
        Args:
            df: DataFrame to check
            
        Returns:
            bool: True if data is consistent
        """
        try:
            # Price consistency checks with tolerance
            tolerance = 1e-6
            
            price_issues = (
                (df['high'] < df['low'] - tolerance) |
                (df['close'] < df['low'] - tolerance) |
                (df['close'] > df['high'] + tolerance) |
                (df['open'] < df['low'] - tolerance) |
                (df['open'] > df['high'] + tolerance)
            )
            
            # Allow a small percentage of inconsistencies
            max_inconsistencies = len(df) * 0.02  # 2% tolerance
            if price_issues.sum() > max_inconsistencies:
                return False
                
            return True
            
        except Exception:
            return False
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        if df is None or df.empty:
            return df
            
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining missing values at the start
        df = df.fillna(method='bfill')
        
        # Drop rows if still have missing values
        df = df.dropna()
        
        logger.info(f"Handled missing data. Rows remaining: {len(df)}")
        return df
    
    def check_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check and improve data quality
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        if df is None or df.empty:
            return df
            
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Remove rows with negative values in price and volume
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df = df[df[numeric_columns] >= 0].copy()
        
        # Ensure high >= low
        df = df[df['high'] >= df['low']].copy()
        
        # Ensure high >= open and close
        df = df[
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close'])
        ].copy()
        
        # Ensure low <= open and close
        df = df[
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])
        ].copy()
        
        logger.info(f"Checked data quality. Rows remaining: {len(df)}")
        return df

    def validate_data_quality(self, df: pd.DataFrame, config: dict) -> Tuple[bool, str]:
        """التحقق من جودة البيانات"""
        try:
            # التحقق من البيانات المفقودة
            missing_threshold = config.get('missing_data_threshold', 0.15)  # تخفيض النسبة من 35% إلى 15%
            missing_percentages = df.isnull().mean()
            
            columns_over_threshold = missing_percentages[missing_percentages > missing_threshold]
            if not columns_over_threshold.empty:
                return False, f"Columns exceeding missing data threshold ({missing_threshold*100}%): {columns_over_threshold.to_dict()}"
                
            # التحقق من التباين
            variance_threshold = config.get('variance_threshold', 1e-6)
            variances = df.select_dtypes(include=[np.number]).var()
            low_variance_cols = variances[variances < variance_threshold]
            
            if not low_variance_cols.empty:
                return False, f"Columns with low variance (< {variance_threshold}): {low_variance_cols.to_dict()}"
                
            # التحقق من القيم المتطرفة
            z_score_threshold = config.get('z_score_threshold', 3.0)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            outliers_info = {}
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > z_score_threshold).sum()
                if outliers > 0:
                    outliers_info[col] = outliers
                    
            if outliers_info:
                logger.warning(f"Found outliers: {outliers_info}")
                
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Error in data validation: {str(e)}"

    def validate_data_consistency(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """التحقق من اتساق البيانات"""
        try:
            # التحقق من القيم السالبة في الأعمدة التي يجب أن تكون موجبة
            positive_columns = ['volume', 'open', 'high', 'low', 'close']
            for col in positive_columns:
                if col in df.columns and (df[col] <= 0).any():
                    return False, f"Found non-positive values in {col}"
                    
            # التحقق من العلاقات المنطقية
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                if not (df['high'] >= df['low']).all():
                    return False, "High values must be greater than or equal to Low values"
                    
                if not ((df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
                    return False, "High values must be the highest"
                    
                if not ((df['low'] <= df['open']) & (df['low'] <= df['close'])).all():
                    return False, "Low values must be the lowest"
                    
            return True, "Data consistency validation passed"
            
        except Exception as e:
            return False, f"Error in consistency validation: {str(e)}"

    def validate_timestamps(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, str]:
        """التحقق من اتساق التواريخ"""
        try:
            if 'timestamp' not in df.columns:
                return False, "Timestamp column not found"
                
            # التأكد من أن التواريخ مرتبة تصاعدياً
            if not df['timestamp'].is_monotonic_increasing:
                return False, "Timestamps are not monotonically increasing"
                
            # التحقق من الفجوات في التواريخ
            time_diff = pd.Timedelta(timeframe)
            expected_diff = pd.Timedelta(timeframe)
            
            actual_diffs = df['timestamp'].diff()
            unexpected_gaps = actual_diffs[actual_diffs > expected_diff * 1.5]
            
            if not unexpected_gaps.empty:
                gap_info = {
                    str(idx): f"{diff.total_seconds() / 60} minutes"
                    for idx, diff in unexpected_gaps.items()
                }
                logger.warning(f"Found unexpected time gaps: {gap_info}")
                
            # التحقق من التواريخ المكررة
            duplicates = df[df['timestamp'].duplicated()]
            if not duplicates.empty:
                return False, f"Found {len(duplicates)} duplicate timestamps"
                
            # التحقق من المستقبل
            current_time = pd.Timestamp.now()
            future_data = df[df['timestamp'] > current_time]
            if not future_data.empty:
                return False, f"Found {len(future_data)} timestamps in the future"
                
            return True, "Timestamp validation passed"
            
        except Exception as e:
            return False, f"Error in timestamp validation: {str(e)}"

    def validate_timeframe_consistency(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, str]:
        """التحقق من اتساق الإطار الزمني"""
        try:
            if 'timestamp' not in df.columns:
                return False, "Timestamp column not found"
                
            # تحويل الإطار الزمني إلى timedelta
            timeframe_map = {
                '1m': pd.Timedelta(minutes=1),
                '5m': pd.Timedelta(minutes=5),
                '15m': pd.Timedelta(minutes=15),
                '30m': pd.Timedelta(minutes=30),
                '1h': pd.Timedelta(hours=1),
                '4h': pd.Timedelta(hours=4),
                '1d': pd.Timedelta(days=1)
            }
            
            if timeframe not in timeframe_map:
                return False, f"Invalid timeframe: {timeframe}"
                
            expected_diff = timeframe_map[timeframe]
            actual_diffs = df['timestamp'].diff()
            
            # السماح بهامش خطأ صغير (1%)
            tolerance = expected_diff * 0.01
            invalid_intervals = actual_diffs[
                (actual_diffs > expected_diff + tolerance) |
                (actual_diffs < expected_diff - tolerance)
            ]
            
            if not invalid_intervals.empty:
                return False, f"Found {len(invalid_intervals)} invalid time intervals"
                
            return True, "Timeframe consistency validation passed"
            
        except Exception as e:
            return False, f"Error in timeframe validation: {str(e)}"
