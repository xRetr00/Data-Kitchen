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
        IndicatorValidationError: If the input data is invalid
    """
    if data is None or len(data) == 0:
        raise IndicatorValidationError("The data is empty")

    try:
        # Convert data to numeric if it's not already
        numeric_data = data.astype(float)
        
        if np.any(np.isnan(numeric_data)):
            raise IndicatorValidationError("The data contains NaN values")
            
    except (ValueError, TypeError):
        raise IndicatorValidationError("The data must be numeric")

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

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate RSI
    
    Args:
        df: DataFrame with price data
        period: RSI period
        
    Returns:
        pd.DataFrame: DataFrame with calculated RSI values
    """
    try:
        delta = df['close'].diff().dropna()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = up.ewm(com=period - 1, adjust=False).mean()
        roll_down1 = down.ewm(com=period - 1, adjust=False).mean().abs()
        RS = roll_up1 / roll_down1
        RSI = 100.0 - (100.0 / (1.0 + RS))
        df[f'RSI_{period}'] = RSI
        return df
    
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return df

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Calculate MACD"""
    try:
        exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        df[f'MACD_{fast_period}_{slow_period}_{signal_period}'] = macd
        df[f'MACD_Signal_{fast_period}_{slow_period}_{signal_period}'] = signal
        
        # Validate MACD values
        if macd.isna().any():
            logger.warning("MACD validation warning: MACD values contain NaN")
        
        return df
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return df

def calculate_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculate SMA
    
    Args:
        df: DataFrame with price data
        window: SMA window size
        
    Returns:
        pd.DataFrame: DataFrame with calculated SMA values
    """
    try:
        df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
        return df
    
    except Exception as e:
        logger.error(f"Error calculating SMA: {str(e)}")
        return df

def calculate_ema(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculate EMA
    
    Args:
        df: DataFrame with price data
        window: EMA window size
        
    Returns:
        pd.DataFrame: DataFrame with calculated EMA values
    """
    try:
        df[f'EMA_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        return df
    
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        return df

def calculate_technical_indicators(df: pd.DataFrame, indicators_config: dict) -> pd.DataFrame:
    """
    Calculate technical indicators
    
    Args:
        df: DataFrame with price data
        indicators_config: Technical indicators configuration
        
    Returns:
        DataFrame with calculated technical indicators
    """
    try:
        logger.debug(f"Starting technical indicators calculation with config: {indicators_config}")
        result_df = df.copy()
        
        for indicator, params in indicators_config.items():
            logger.debug(f"Calculating {indicator} with parameters: {params}")
            
            try:
                if indicator == 'RSI':
                    validate_input_data(df['close'].values, 'RSI')
                    result_df[f'RSI_{params["period"]}'] = talib.RSI(df['close'].values, timeperiod=params['period'])
                    
                elif indicator == 'MACD':
                    validate_input_data(df['close'].values, 'MACD')
                    macd, signal, _ = talib.MACD(
                        df['close'].values,
                        fastperiod=params['fast_period'],
                        slowperiod=params['slow_period'],
                        signalperiod=params['signal_period']
                    )
                    result_df[f'MACD_{params["fast_period"]}_{params["slow_period"]}_{params["signal_period"]}'] = macd
                    result_df[f'MACD_Signal_{params["fast_period"]}_{params["slow_period"]}_{params["signal_period"]}'] = signal
                    
                elif indicator == 'SMA':
                    for period in params['periods']:
                        validate_input_data(df['close'].values, f'SMA_{period}')
                        result_df[f'SMA_{period}'] = talib.SMA(df['close'].values, timeperiod=period)
                
                logger.debug(f"Successfully calculated {indicator}")
                
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {str(e)}")
                raise
        
        logger.debug("Technical indicators calculation completed successfully")
        return result_df
        
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
        logger.debug(f"Starting normalization for columns: {feature_columns}")
        df = df.copy()
        scaler = MinMaxScaler()
        
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Clip values to ensure they are exactly between 0 and 1
        df[feature_columns] = df[feature_columns].clip(0, 1)
        
        logger.debug(f"Normalization completed successfully. Data range: {df[feature_columns].min().to_dict()} to {df[feature_columns].max().to_dict()}")
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
        logger.debug(f"Starting missing data handling. Input shape: {df.shape}")
        
        # Calculate missing value proportions
        missing_proportions = df.isnull().mean()
        missing_columns = missing_proportions[missing_proportions > 0].index.tolist()
        
        if missing_columns:
            logger.debug(f"Detected missing values in columns: {missing_columns}")
            logger.debug(f"Missing proportions: {missing_proportions[missing_columns].to_dict()}")
        
        # Drop columns with too many missing values
        columns_to_drop = missing_proportions[missing_proportions > threshold].index
        if len(columns_to_drop) > 0:
            logger.warning(f"Dropping columns with high missing value proportion: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # Forward fill then backward fill missing values
        df = df.ffill().bfill()
        
        logger.debug(f"Missing data handling completed. Output shape: {df.shape}")
        logger.debug(f"Remaining missing values: {df.isnull().sum().to_dict()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error handling missing data: {str(e)}")
        raise
