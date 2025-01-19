"""
Data Validator Module
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from .logger import setup_logger
import logging

logger = setup_logger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DataValidator:
    """Class for validating and cleaning data"""
    
    def __init__(self, 
                 missing_threshold: float = 0.15,  # Reduced from 0.35
                 min_data_points: int = 50,
                 variance_threshold: float = 1e-6,
                 z_score_threshold: float = 3.0):
        """
        Initialize DataValidator with improved thresholds
        
        Args:
            missing_threshold: Maximum allowed ratio of missing data
            min_data_points: Minimum number of data points required
            variance_threshold: Minimum variance threshold
            z_score_threshold: Z-score threshold for outlier detection
        """
        self.missing_threshold = missing_threshold
        self.min_data_points = min_data_points
        self.variance_threshold = variance_threshold
        self.z_score_threshold = z_score_threshold
        
        logger.info(
            f"Data Validator initialized with:"
            f"\n- missing_threshold={missing_threshold}"
            f"\n- min_data_points={min_data_points}"
            f"\n- variance_threshold={variance_threshold}"
            f"\n- z_score_threshold={z_score_threshold}"
        )
    
    def validate_data_structure(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data structure and required columns
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            # Check if DataFrame is None or empty
            if df is None or df.empty:
                return False, "DataFrame is None or empty"
            
            # Required columns for price data
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            # Check data types
            expected_types = {
                'timestamp': ['datetime64[ns]', 'datetime64'],
                'open': ['float64', 'float32', 'int64', 'int32'],
                'high': ['float64', 'float32', 'int64', 'int32'],
                'low': ['float64', 'float32', 'int64', 'int32'],
                'close': ['float64', 'float32', 'int64', 'int32'],
                'volume': ['float64', 'float32', 'int64', 'int32']
            }
            
            for col, expected_type in expected_types.items():
                if str(df[col].dtype) not in expected_type:
                    return False, f"Invalid data type for {col}: expected {expected_type}, got {df[col].dtype}"
            
            return True, "Data structure validation passed"
            
        except Exception as e:
            return False, f"Error in structure validation: {str(e)}"
    
    def validate_sequence_data(self, sequences: np.ndarray, targets: np.ndarray) -> Tuple[bool, str]:
        """
        Validate sequence data for model training
        
        Args:
            sequences: Input sequences array
            targets: Target values array
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            # Check if arrays are None
            if sequences is None or targets is None:
                return False, "Sequences or targets array is None"
            
            # Check if arrays are empty
            if sequences.size == 0 or targets.size == 0:
                return False, "Empty sequences or targets array"
            
            # Check dimensions
            if len(sequences.shape) != 3:
                return False, f"Invalid sequence shape: expected 3D array, got {len(sequences.shape)}D"
            
            if len(targets.shape) != 2:
                return False, f"Invalid targets shape: expected 2D array, got {len(targets.shape)}D"
            
            # Check sequence length
            if sequences.shape[1] < self.min_data_points:
                return False, f"Sequence length too short: {sequences.shape[1]} < {self.min_data_points}"
            
            # Check matching dimensions
            if sequences.shape[0] != targets.shape[0]:
                return False, f"Mismatched dimensions: sequences={sequences.shape[0]}, targets={targets.shape[0]}"
            
            # Check for NaN and infinite values
            if np.isnan(sequences).any() or np.isnan(targets).any():
                return False, "Found NaN values in sequences or targets"
            
            if np.isinf(sequences).any() or np.isinf(targets).any():
                return False, "Found infinite values in sequences or targets"
            
            return True, "Sequence validation passed"
            
        except Exception as e:
            return False, f"Error in sequence validation: {str(e)}"
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
        """
        Validate data quality with detailed statistics
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple[bool, Dict]: (is_valid, statistics)
        """
        try:
            stats = {}
            
            # Missing data analysis
            missing_stats = df.isnull().sum() / len(df)
            stats['missing_data'] = missing_stats.to_dict()
            
            if missing_stats.max() > self.missing_threshold:
                return False, {'error': 'Too many missing values', 'stats': stats}
            
            # Variance analysis
            variance_stats = df.select_dtypes(include=[np.number]).var()
            stats['variance'] = variance_stats.to_dict()
            
            if (variance_stats < self.variance_threshold).any():
                return False, {'error': 'Low variance detected', 'stats': stats}
            
            # Outlier analysis
            outliers = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = (z_scores > self.z_score_threshold).sum()
            
            stats['outliers'] = outliers
            
            # Data consistency checks
            consistency_valid, consistency_msg = self.validate_data_consistency(df)
            stats['consistency'] = {'valid': consistency_valid, 'message': consistency_msg}
            
            if not consistency_valid:
                return False, {'error': consistency_msg, 'stats': stats}
            
            return True, stats
            
        except Exception as e:
            return False, {'error': f"Error in quality validation: {str(e)}"}
    
    def validate_data_consistency(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data consistency with improved checks
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            # Price relationship checks
            price_issues = []
            
            # High >= Low
            high_low_issues = df[df['high'] < df['low']].index.tolist()
            if high_low_issues:
                price_issues.append(f"High < Low at indices: {high_low_issues[:5]}...")
            
            # High >= Open, Close
            high_issues = df[
                (df['high'] < df['open']) | 
                (df['high'] < df['close'])
            ].index.tolist()
            if high_issues:
                price_issues.append(f"High not highest at indices: {high_issues[:5]}...")
            
            # Low <= Open, Close
            low_issues = df[
                (df['low'] > df['open']) | 
                (df['low'] > df['close'])
            ].index.tolist()
            if low_issues:
                price_issues.append(f"Low not lowest at indices: {low_issues[:5]}...")
            
            # Volume checks
            volume_issues = df[df['volume'] <= 0].index.tolist()
            if volume_issues:
                price_issues.append(f"Invalid volume at indices: {volume_issues[:5]}...")
            
            if price_issues:
                return False, "Data consistency issues found:\n" + "\n".join(price_issues)
            
            return True, "Data consistency validation passed"
            
        except Exception as e:
            return False, f"Error in consistency validation: {str(e)}"

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
