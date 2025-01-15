"""
Helper functions for data processing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import talib
from sklearn.preprocessing import MinMaxScaler
from .logger import setup_logger

logger = setup_logger(__name__)

class IndicatorValidationError(Exception):
    """Custom exception for indicator validation errors"""
    pass

def validate_input_data(data: np.ndarray, indicator_name: str) -> None:
    """
    Validate input data
    
    Args:
        data: Input data array
        indicator_name: Indicator name
    
    Raises:
        IndicatorValidationError: If input data is invalid
    """
    if data is None or len(data) == 0:
        raise IndicatorValidationError(f"Input data is empty for indicator {indicator_name}")
    
    if np.any(np.isnan(data)):
        raise IndicatorValidationError(f"Input data contains NaN values for indicator {indicator_name}")
    
    if np.any(np.isinf(data)):
        raise IndicatorValidationError(f"Input data contains infinite values for indicator {indicator_name}")

def validate_rsi(rsi_values: np.ndarray) -> Tuple[bool, str]:
    """
    Validate RSI values
    
    Args:
        rsi_values: RSI values array
        
    Returns:
        Tuple[bool, str]: Validation result and error message
    """
    if np.any(np.isnan(rsi_values)):
        return False, "RSI values contain NaN"
    
    if np.any((rsi_values < 0) | (rsi_values > 100)):
        return False, "RSI values are out of range (0-100)"
    
    return True, ""

def validate_macd(macd_values: np.ndarray, signal_values: np.ndarray) -> Tuple[bool, str]:
    """
    Validate MACD values
    
    Args:
        macd_values: MACD values array
        signal_values: MACD signal values array
        
    Returns:
        Tuple[bool, str]: Validation result and error message
    """
    if np.any(np.isnan(macd_values)) or np.any(np.isnan(signal_values)):
        return False, "MACD values contain NaN"
    
    if np.any(np.isinf(macd_values)) or np.any(np.isinf(signal_values)):
        return False, "MACD values contain infinite values"
    
    return True, ""

def validate_sma(sma_values: np.ndarray, window: int, data_length: int) -> Tuple[bool, str]:
    """
    Validate SMA values
    
    Args:
        sma_values: SMA values array
        window: SMA window size
        data_length: Input data length
        
    Returns:
        Tuple[bool, str]: Validation result and error message
    """
    expected_nan_count = min(window - 1, data_length)
    actual_nan_count = np.isnan(sma_values).sum()
    
    if actual_nan_count > expected_nan_count:
        return False, f"Number of NaN values in SMA ({actual_nan_count}) is greater than expected ({expected_nan_count})"
    
    if np.any(np.isinf(sma_values)):
        return False, "SMA values contain infinite values"
    
    return True, ""

def calculate_technical_indicators(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    حساب المؤشرات الفنية مع التحقق من صحتها
    """
    try:
        df = df.copy()
        
        # تحويل البيانات إلى النوع المناسب
        close_prices = df['close'].astype(float).values
        
        # معالجة القيم المفقودة قبل حساب المؤشرات
        close_prices = pd.Series(close_prices).fillna(method='ffill').fillna(method='bfill').values
        
        validate_input_data(close_prices, "close prices")
        
        # حساب RSI
        if 'RSI' in settings:
            period = settings['RSI']['period']
            # التأكد من وجود بيانات كافية لحساب RSI
            if len(close_prices) > period:
                rsi_values = talib.RSI(close_prices, timeperiod=period)
                is_valid, error_msg = validate_rsi(rsi_values)
                if not is_valid:
                    logger.warning(f"RSI validation warning: {error_msg}")
                    # معالجة القيم غير الصالحة
                    rsi_values = pd.Series(rsi_values).fillna(method='ffill').fillna(50).values
                df[f'RSI_{period}'] = rsi_values
            else:
                logger.warning(f"Not enough data to calculate RSI (need > {period} points)")
        
        # حساب MACD
        if 'MACD' in settings:
            macd_settings = settings['MACD']
            macd, signal, _ = talib.MACD(
                close_prices,
                fastperiod=macd_settings['fast_period'],
                slowperiod=macd_settings['slow_period'],
                signalperiod=macd_settings['signal_period']
            )
            is_valid, error_msg = validate_macd(macd, signal)
            if not is_valid:
                logger.warning(f"MACD validation warning: {error_msg}")
                # معالجة القيم غير الصالحة
                macd = pd.Series(macd).fillna(method='ffill').fillna(0).values
                signal = pd.Series(signal).fillna(method='ffill').fillna(0).values
            df['MACD'] = macd
            df['MACD_Signal'] = signal
        
        # حساب SMA
        if 'SMA' in settings:
            for period in settings['SMA']['periods']:
                sma_values = talib.SMA(close_prices, timeperiod=period)
                is_valid, error_msg = validate_sma(sma_values, period, len(close_prices))
                if not is_valid:
                    logger.warning(f"SMA validation warning: {error_msg}")
                    # معالجة القيم غير الصالحة
                    sma_values = pd.Series(sma_values).fillna(method='ffill').fillna(close_prices.mean()).values
                df[f'SMA_{period}'] = sma_values
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise

def normalize_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Normalize feature values
    
    Args:
        df: Input data frame
        feature_columns: List of feature columns to normalize
        
    Returns:
        pd.DataFrame: Data frame with normalized feature values
    """
    try:
        df = df.copy()
        scaler = MinMaxScaler()
        
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        return df
    
    except Exception as e:
        logger.error(f"Error normalizing feature values: {str(e)}")
        raise

def handle_missing_data(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Handle missing data
    
    Args:
        df: Input data frame
        threshold: Maximum allowed proportion of missing values
        
    Returns:
        pd.DataFrame: Data frame with handled missing data
    """
    try:
        # Calculate proportion of missing values for each column
        missing_ratio = df.isnull().sum() / len(df)
        
        # Drop columns with missing values above threshold
        columns_to_drop = missing_ratio[missing_ratio > threshold].index
        if len(columns_to_drop) > 0:
            logger.warning(f"Dropping columns with high missing value proportion: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # Fill remaining missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    except Exception as e:
        logger.error(f"Error handling missing data: {str(e)}")
        raise
