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
    """Validate input data"""
    if np.any(np.isnan(data)):
        logger.warning(f"Input data for {indicator} contains NaN values")
        return False, f"Input data for {indicator} contains NaN values"
    if np.any(np.isinf(data)):
        logger.warning(f"Input data for {indicator} contains infinite values")
        return False, f"Input data for {indicator} contains infinite values"
    if len(data) < 2:
        logger.warning(f"Not enough data points for {indicator}")
        return False, f"Not enough data points for {indicator}"
    return True, ""



def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate RSI
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int): RSI period
        
    Returns:
        pd.DataFrame: DataFrame with calculated RSI values
    """
    try:
        # Copy the data to avoid modifying the original
        df_copy = df.copy()
        
        # Reorder the index to avoid repetition
        df_copy = df_copy.reset_index(drop=True)
        
        # Calculate the price change
        delta = df_copy['close'].diff()
        
        # Separate the gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate the exponential moving average
        avg_gain = gain.ewm(com=period-1, adjust=True).mean()  # Change adjust to True
        avg_loss = loss.ewm(com=period-1, adjust=True).mean()
        
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.nan)
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Fill the missing values at the beginning
        rsi[:period] = np.nan

        # Add RSI to the original data
        df[f'RSI_{period}'] = rsi
        
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

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data in DataFrame
    
    Args:
        df: DataFrame with possibly missing values
        
    Returns:
        DataFrame with handled missing values
    """
    try:
        # Make a copy to avoid modifying original
        logger.info("Starting missing data handling")
        result_df = df.copy()
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=Console()
        ) as progress:
            total_steps = 4
            task = progress.add_task("[cyan]Handling missing data...", total=total_steps)
            
            # Forward fill missing values
            result_df = result_df.ffill()
            logger.debug("Completed forward fill")
            progress.update(task, advance=1, description="[cyan]Forward filling...")
            
            # If still have missing values at the start, backward fill
            result_df = result_df.bfill()
            logger.debug("Completed backward fill")
            progress.update(task, advance=1, description="[cyan]Backward filling...")
            
            # Drop rows where all indicator columns are NaN
            indicator_cols = [col for col in result_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            if indicator_cols:
                result_df = result_df.dropna(subset=indicator_cols, how='all')
                logger.debug("Completed dropping empty rows")
                progress.update(task, advance=1, description="[cyan]Dropping empty rows...")
            else:
                logger.info("Successfully handled all missing values")
            
            # Log missing data info
            missing_count = result_df.isnull().sum()
            if missing_count.any():
                logger.warning(f"Missing values after handling:\n{missing_count[missing_count > 0]}")
            progress.update(task, advance=1, description="[cyan]Checking remaining missing values...")
            
            return result_df
            
    except Exception as e:
        logger.error(f"Error handling missing data: {str(e)}")
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
