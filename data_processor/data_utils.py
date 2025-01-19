"""
Helper functions for data processing
"""
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import talib
from sklearn.preprocessing import MinMaxScaler
from .logger import setup_logger
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, Console

logger = setup_logger(__name__)

class IndicatorValidationError(Exception):
    """Custom exception for indicator validation errors"""
    pass

def validate_input_data(data: np.ndarray, indicator: str) -> Tuple[bool, str]:
    """Validate input data with improved NaN handling"""
    if data is None:
        return False, f"No data provided for {indicator}"
        
    # Check for empty data
    if len(data) == 0:
        return False, f"Empty data provided for {indicator}"
        
    # Count NaN values
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        nan_ratio = nan_count / len(data)
        if nan_ratio > 0.5:  # If more than 50% are NaN
            return False, f"Too many NaN values in {indicator} data ({nan_ratio:.2%})"
        logger.warning(f"{indicator}: {nan_count} NaN values found ({nan_ratio:.2%})")
        
    # Check for infinite values
    inf_count = np.isinf(data).sum()
    if inf_count > 0:
        return False, f"{inf_count} infinite values found in {indicator} data"
        
    # Check data length
    if len(data) < 2:
        return False, f"Not enough data points for {indicator} (minimum 2 required)"
        
    return True, ""



def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate RSI with improved NaN handling and validation
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int): RSI period
        
    Returns:
        pd.DataFrame: DataFrame with calculated RSI values
    """
    try:
        df_copy = df.copy()
        
        # Validate input data
        if len(df_copy) < period + 1:
            logger.warning(f"Not enough data points for RSI calculation. Need at least {period + 1}, got {len(df_copy)}")
            return df
            
        # Handle NaN in close prices
        close_prices = df_copy['close']
        if close_prices.isnull().any():
            logger.warning("Found NaN values in close prices")
            # Fill small gaps with interpolation
            close_prices = close_prices.interpolate(method='linear', limit=5)
            # Fill remaining gaps with forward fill
            close_prices = close_prices.fillna(method='ffill')
            df_copy['close'] = close_prices
            
        # Calculate price changes
        delta = df_copy['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50)  # Fill initial NaN with neutral RSI
        rsi = rsi.clip(0, 100)  # Ensure RSI stays within bounds
        
        # Validate final RSI values
        if rsi.isnull().any():
            nan_count = rsi.isnull().sum()
            logger.warning(f"RSI still contains {nan_count} NaN values after calculation")
            rsi = rsi.fillna(method='ffill').fillna(50)
            
        df[f'RSI_{period}'] = rsi
        return df
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return df

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate MACD with improved NaN handling
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        pd.DataFrame: DataFrame with calculated MACD values
    """
    try:
        # Handle NaN in close prices
        close_prices = df['close']
        if close_prices.isnull().any():
            logger.warning("Found NaN values in close prices for MACD calculation")
            close_prices = close_prices.interpolate(method='linear', limit=5)
            close_prices = close_prices.fillna(method='ffill')
            
        # Calculate EMAs
        exp1 = close_prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = close_prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Handle any remaining NaN values
        if macd.isna().any():
            logger.warning("MACD contains NaN values, applying forward fill")
            macd = macd.fillna(method='ffill')
            
        if signal.isna().any():
            logger.warning("MACD signal contains NaN values, applying forward fill")
            signal = signal.fillna(method='ffill')
            
        # Add to DataFrame with consistent naming
        macd_col = f'MACD_{fast_period}_{slow_period}_{signal_period}'
        signal_col = f'MACD_Signal_{fast_period}_{slow_period}_{signal_period}'
        
        df[macd_col] = macd
        df[signal_col] = signal
        
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
    """Calculate technical indicators with error handling"""
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df

    console = Console(file=sys.stderr)
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    )
        
    try:
        result_df = df.copy()
        with progress:
            task = progress.add_task("[cyan]Calculating indicators...", total=len(indicators_config))
        
        for indicator, params in indicators_config.items():
            logger.info(f"Calculating {indicator}...")
            try:
                # Validate input data
                validate_input_data(df['close'].values, indicator)
                
                if indicator == 'RSI':
                    result_df = calculate_rsi(result_df, params.get('period', 14))
                    # Validate RSI
                    is_valid, error_msg = validate_rsi(result_df[f'RSI_{params.get("period", 14)}'].values)
                    if not is_valid:
                        logger.error(f"RSI validation failed: {error_msg}")
                        result_df.drop(columns=[f'RSI_{params.get("period", 14)}'], inplace=True)
                        
                elif indicator == 'MACD':
                    result_df = calculate_macd(result_df, 
                        params.get('fast_period', 12),
                        params.get('slow_period', 26),
                        params.get('signal_period', 9)
                    )
                    # Validate MACD
                    macd_col = f'MACD_{params.get("fast_period", 12)}_{params.get("slow_period", 26)}_{params.get("signal_period", 9)}'
                    signal_col = f'MACD_Signal_{params.get("fast_period", 12)}_{params.get("slow_period", 26)}_{params.get("signal_period", 9)}'
                    is_valid, error_msg = validate_macd(result_df[macd_col].values, result_df[signal_col].values)
                    if not is_valid:
                        logger.error(f"MACD validation failed: {error_msg}")
                        result_df.drop(columns=[macd_col, signal_col], inplace=True)
                        
                elif indicator == 'SMA':
                    result_df = calculate_sma(result_df, params.get('window', 20))
                    # Validate SMA
                    is_valid, error_msg = validate_sma(
                        result_df[f'SMA_{params.get("window", 20)}'].values,
                        params.get('window', 20),
                        len(df)
                    )
                    if not is_valid:
                        logger.error(f"SMA validation failed: {error_msg}")
                        result_df.drop(columns=[f'SMA_{params.get("window", 20)}'], inplace=True)
                        
                elif indicator == 'EMA':
                    result_df = calculate_ema(result_df, params.get('window', 20))
                    # Validate EMA
                    ema_values = result_df[f'EMA_{params.get("window", 20)}'].values
                    if np.any(np.isnan(ema_values)) or np.any(np.isinf(ema_values)):
                        logger.error("EMA validation failed: contains NaN or infinite values")
                        result_df.drop(columns=[f'EMA_{params.get("window", 20)}'], inplace=True)
                    
                progress.update(task, advance=1)
                
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {str(e)}")
                continue
                
        return result_df
        
    except Exception as e:
        logger.error(f"Error in calculate_technical_indicators: {str(e)}")
        return df

def handle_missing_data(df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """
    Handle missing data in DataFrame with improved gap handling
    
    Args:
        df: DataFrame with possibly missing values
        max_gap: Maximum number of consecutive NaN values to fill
        
    Returns:
        DataFrame with handled missing values
    """
    try:
        logger.info("Starting missing data handling")
        result_df = df.copy()
        
        # Get initial NaN statistics
        initial_nans = result_df.isna().sum()
        logger.info(f"Initial NaN count per column: {initial_nans}")
        
        # Handle different columns differently
        for column in result_df.columns:
            nan_mask = result_df[column].isna()
            nan_count = nan_mask.sum()
            
            if nan_count == 0:
                continue
                
            # Get consecutive NaN sequences
            nan_groups = result_df[column].isna().astype(int).groupby(result_df.index).sum()
            max_consecutive_nans = nan_groups.max()
            
            logger.info(f"Column {column}: {nan_count} NaNs, max consecutive: {max_consecutive_nans}")
            
            if max_consecutive_nans <= max_gap:
                # Small gaps: Use linear interpolation
                result_df[column] = result_df[column].interpolate(method='linear', limit=max_gap)
            else:
                # Larger gaps: Use forward fill then backward fill
                result_df[column] = result_df[column].fillna(method='ffill', limit=max_gap)
                result_df[column] = result_df[column].fillna(method='bfill', limit=max_gap)
        
        # Check remaining NaNs
        final_nans = result_df.isna().sum()
        logger.info(f"Final NaN count per column: {final_nans}")
        
        # Drop rows where all indicator columns are NaN
        indicator_columns = [col for col in result_df.columns if any(ind in col for ind in ['RSI', 'MACD', 'SMA', 'EMA'])]
        if indicator_columns:
            result_df = result_df.dropna(subset=indicator_columns, how='all')
            logger.info(f"Dropped {len(df) - len(result_df)} rows where all indicators were NaN")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in handle_missing_data: {str(e)}")
        return df


def normalize_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Normalize features using MinMaxScaler
    
    Args:
        df: Input DataFrame
        feature_columns: List of columns to normalize
        
    Returns:
        DataFrame with normalized features
    """
    try:
        # Create a copy to avoid modifying the original
        df_norm = df.copy()
        
        # Initialize scaler
        scaler = MinMaxScaler()
        
        # Normalize each feature
        for col in feature_columns:
            # Skip if column has all NaN values
            if df_norm[col].isnull().all():
                logger.warning(f"Skipping normalization for {col}: all values are NaN")
                continue
                
            # Fill NaN with median for normalization
            temp_col = df_norm[col].fillna(df_norm[col].median())
            
            # Reshape for scaler
            values = temp_col.values.reshape(-1, 1)
            
            # Normalize
            try:
                normalized = scaler.fit_transform(values)
                df_norm[col] = normalized.flatten()
            except Exception as e:
                logger.error(f"Error normalizing {col}: {str(e)}")
                continue
                
        # Log success
        logger.info(f"Successfully normalized {len(feature_columns)} features")
        
        return df_norm
        
    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        raise


def validate_rsi(rsi_values: np.ndarray) -> Tuple[bool, str]:
    """
    Validate RSI values
    
    Args:
        rsi_values: numpy array of RSI values
        
    Returns:
        (is_valid, error_message)
    """
    try:
        # Allow up to 10% NaN values at the beginning
        if np.isnan(rsi_values).sum() > len(rsi_values) * 0.1:
            return False, "RSI contains too many NaN values"
            
        # Check non-NaN values only
        valid_values = rsi_values[~np.isnan(rsi_values)]
        if np.any((valid_values < 0) | (valid_values > 100)):
            return False, "RSI values outside valid range [0, 100]"
            
        return True, ""
        
    except Exception as e:
        return False, str(e)


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
        return False, "MACD or Signal contains NaN values"
    
    if np.any(np.isinf(macd_values)) or np.any(np.isinf(signal_values)):
        return False, "MACD or Signal contains infinite values"
    
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
        return False, f"SMA contains NaN values"
    
    if np.any(np.isinf(sma_values)):
        return False, "SMA contains infinite values"

    if window > data_length:
        return False, f"Window size ({window}) larger than data length ({data_length})"

    return True, ""
