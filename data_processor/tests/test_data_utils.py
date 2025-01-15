"""
Unit tests for data_utils.py
"""

import pytest
import pandas as pd
import numpy as np
from ..data_utils import (
    calculate_technical_indicators,
    normalize_features,
    handle_missing_data,
    validate_rsi,
    validate_macd,
    validate_sma,
    validate_input_data,
    IndicatorValidationError
)
from .test_logger import setup_test_logger

logger = setup_test_logger()

@pytest.fixture
def sample_data():
    """Create test data"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(len(dates)) + 100,
        'high': np.random.randn(len(dates)) + 101,
        'low': np.random.randn(len(dates)) + 99,
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    return df

def test_calculate_technical_indicators(sample_data):
    """Test calculating technical indicators"""
    logger.info("Starting test for calculate_technical_indicators")
    
    try:
        settings = {
            'RSI': {'period': 14},
            'MACD': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'SMA': {'periods': [20, 50]}
        }
        
        result = calculate_technical_indicators(sample_data, settings)
        
        # Check for expected columns
        assert 'RSI_14' in result.columns
        assert 'MACD_12_26_9' in result.columns
        assert 'MACD_Signal_12_26_9' in result.columns
        assert 'SMA_20' in result.columns
        assert 'SMA_50' in result.columns
        
        # Check for no NaN values at the end of the data
        assert not result.iloc[-1:].isna().any().any()
        
        logger.info("Test for calculate_technical_indicators succeeded")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for calculate_technical_indicators")
        raise
    
    logger.info("Test for calculate_technical_indicators completed successfully")

def test_normalize_features(sample_data):
    """Test normalizing features"""
    logger.info("Starting test for normalize_features")
    
    try:
        feature_columns = ['close', 'volume']
        result = normalize_features(sample_data, feature_columns)
        
        # Check that values are between 0 and 1
        for col in feature_columns:
            assert result[col].min() >= 0
            assert result[col].max() <= 1
        
        # Check that other columns are unchanged
        assert (result['open'] == sample_data['open']).all()
        
        logger.info("Test for normalize_features succeeded")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for normalize_features")
        raise
    
    logger.info("Test for normalize_features completed successfully")

def test_handle_missing_data():
    """Test handling missing data"""
    logger.info("Starting test for handle_missing_data")
    
    try:
        # Create data with missing values
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [np.nan, np.nan, np.nan, 4, 5]
        })
        
        result = handle_missing_data(df, threshold=0.5)
        
        # Check column C is dropped (more than 50% missing values)
        assert 'C' not in result.columns
        
        # Check missing values are filled in remaining columns
        assert not result.isna().any().any()
        
        logger.info("Test for handle_missing_data succeeded")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for handle_missing_data")
        raise
    
    logger.info("Test for handle_missing_data completed successfully")

def test_handle_missing_data_no_missing():
    """Test handling missing data without missing values"""
    logger.info("Starting test for handle_missing_data_no_missing")
    
    try:
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = handle_missing_data(df)
        
        # Check that data is not changed
        assert (result == df).all().all()
        
        logger.info("Test for handle_missing_data_no_missing succeeded")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for handle_missing_data_no_missing")
        raise
    
    logger.info("Test for handle_missing_data_no_missing completed successfully")

def test_validate_input_data():
    """Test validating input data"""
    logger.info("Starting test for validate_input_data")

    try:
        # Valid data
        valid_data = np.array([1.0, 2.0, 3.0])
        logger.debug(f"Testing valid data: {valid_data}")
        validate_input_data(valid_data, "test")
        logger.info("Test for valid data succeeded")

        # Empty data
        logger.debug("Testing empty data")
        with pytest.raises(IndicatorValidationError, match="The data is empty"):
            validate_input_data(np.array([]), "test")
        logger.info("Test for empty data succeeded, expected exception was raised")

        # Data with NaN values
        invalid_data = np.array([1.0, np.nan, 3.0])
        logger.debug(f"Testing data with NaN values: {invalid_data}")
        with pytest.raises(IndicatorValidationError, match="The data contains NaN values"):
            validate_input_data(invalid_data, "test")
        logger.info("Test for data with NaN values succeeded")

        # Non-numeric data
        non_numeric_data = np.array(['a', 'b', 'c'])
        logger.debug(f"Testing non-numeric data: {non_numeric_data}")
        with pytest.raises(IndicatorValidationError, match="The data must be numeric"):
            validate_input_data(non_numeric_data, "test")
        logger.info("Test for non-numeric data succeeded")

    except Exception as e:
        logger.error(f"An unexpected error occurred in test for validate_input_data: {str(e)}")
        raise

def test_validate_rsi():
    """Test validating RSI"""
    logger.info("Starting test for validate_rsi")
    
    try:
        # Valid values
        valid_rsi = np.array([30.0, 50.0, 70.0])
        logger.debug(f"Testing valid RSI values: {valid_rsi}")
        is_valid, _ = validate_rsi(valid_rsi)
        assert is_valid
        logger.info("Test for valid RSI values succeeded")
        
        # Out of range values
        invalid_rsi = np.array([-10.0, 50.0, 110.0])
        logger.debug(f"Testing out of range RSI values: {invalid_rsi}")
        is_valid, error_msg = validate_rsi(invalid_rsi)
        assert not is_valid
        logger.info(f"Test for out of range RSI values succeeded, expected error message: {error_msg}")
        
        # Values with NaN
        invalid_rsi = np.array([30.0, np.nan, 70.0])
        logger.debug(f"Testing RSI values with NaN: {invalid_rsi}")
        is_valid, error_msg = validate_rsi(invalid_rsi)
        assert not is_valid
        logger.info(f"Test for RSI values with NaN succeeded, expected error message: {error_msg}")
        
        logger.info("Test for validate_rsi succeeded")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for validate_rsi")
        raise
    
    logger.info("Test for validate_rsi completed successfully")

def test_validate_macd():
    """Test validating MACD"""
    logger.info("Starting test for validate_macd")
    
    try:
        # Valid values
        valid_macd = np.array([1.0, -1.0, 0.5])
        valid_signal = np.array([0.8, -0.8, 0.3])
        logger.debug(f"Testing valid MACD values: {valid_macd}, signal values: {valid_signal}")
        is_valid, _ = validate_macd(valid_macd, valid_signal)
        assert is_valid
        logger.info("Test for valid MACD values succeeded")
        
        # Values with NaN
        invalid_macd = np.array([1.0, np.nan, 0.5])
        invalid_signal = np.array([0.8, -0.8, 0.3])
        logger.debug(f"Testing MACD values with NaN: {invalid_macd}, signal values: {invalid_signal}")
        is_valid, error_msg = validate_macd(invalid_macd, invalid_signal)
        assert not is_valid
        logger.info(f"Test for MACD values with NaN succeeded, expected error message: {error_msg}")
        
        # Values with infinite values
        invalid_macd = np.array([1.0, np.inf, 0.5])
        invalid_signal = np.array([0.8, -0.8, 0.3])
        logger.debug(f"Testing MACD values with infinite values: {invalid_macd}, signal values: {invalid_signal}")
        is_valid, error_msg = validate_macd(invalid_macd, invalid_signal)
        assert not is_valid
        logger.info(f"Test for MACD values with infinite values succeeded, expected error message: {error_msg}")
        
        logger.info("Test for validate_macd succeeded")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for validate_macd")
        raise
    
    logger.info("Test for validate_macd completed successfully")

def test_validate_sma():
    """Test validating SMA"""
    logger.info("Starting test for validate_sma")
    
    try:
        # Valid values
        valid_sma = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        logger.debug(f"Testing valid SMA values: {valid_sma}")
        is_valid, _ = validate_sma(valid_sma, window=3, data_length=5)
        assert is_valid
        logger.info("Test for valid SMA values succeeded")
        
        # Number of NaN values greater than expected
        invalid_sma = np.array([np.nan, np.nan, np.nan, np.nan, 5.0])
        logger.debug(f"Testing SMA values with number of NaN values greater than expected: {invalid_sma}")
        is_valid, error_msg = validate_sma(invalid_sma, window=3, data_length=5)
        assert not is_valid
        logger.info(f"Test for SMA values with number of NaN values greater than expected succeeded, expected error message: {error_msg}")
        
        # Infinite values
        invalid_sma = np.array([np.nan, np.nan, np.inf, 4.0, 5.0])
        logger.debug(f"Testing SMA values with infinite values: {invalid_sma}")
        is_valid, error_msg = validate_sma(invalid_sma, window=3, data_length=5)
        assert not is_valid
        logger.info(f"Test for SMA values with infinite values succeeded, expected error message: {error_msg}")
        
        logger.info("Test for validate_sma succeeded")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for validate_sma")
        raise
    
    logger.info("Test for validate_sma completed successfully")

def test_calculate_technical_indicators_with_invalid_data():
    """Test calculating technical indicators with invalid data"""
    logger.info("Starting test for calculate_technical_indicators_with_invalid_data")
    
    try:
        # Create data with invalid values
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
        df = pd.DataFrame({
            'open': range(len(dates)),
            'high': range(len(dates)),
            'low': range(len(dates)),
            'close': [np.nan if i == 5 else float(i) for i in range(len(dates))],
            'volume': range(len(dates))
        }, index=dates)

        settings = {
            'RSI': {'period': 14},
            'MACD': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'SMA': {'periods': [20]}
        }

        # should raise an exception due to NaN values in the data
        with pytest.raises(IndicatorValidationError):
            calculate_technical_indicators(df, settings)
        
        logger.info("Test for calculate_technical_indicators_with_invalid_data succeeded")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred in test for calculate_technical_indicators_with_invalid_data: {str(e)}")
        raise

def test_empty_dataframe():
    """Test handling empty DataFrame"""
    logger.info("Starting test for empty DataFrame")
    
    try:
        # Create empty DataFrame
        df = pd.DataFrame()
        
        # Test normalize_features
        with pytest.raises(Exception):
            normalize_features(df, ['close'])
        
        # Test handle_missing_data
        with pytest.raises(Exception):
            handle_missing_data(df)
        
        # Test calculate_technical_indicators
        with pytest.raises(Exception):
            calculate_technical_indicators(df, {'RSI': {'period': 14}})
            
        logger.info("Test for empty DataFrame completed successfully")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for empty DataFrame")
        raise

def test_invalid_timeframe():
    """Test handling invalid timeframe data"""
    logger.info("Starting test for invalid timeframe")
    
    try:
        # Create DataFrame with invalid timeframe
        df = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1.1, 2.1, 3.1],
            'low': [0.9, 1.9, 2.9],
            'close': [1, 2, 3],
            'volume': [100, 200, 300]
        })
        
        # Test technical indicators with insufficient data
        with pytest.raises(Exception):
            calculate_technical_indicators(df, {
                'RSI': {'period': 14},  # Requires more data points than available
                'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            })
        
        logger.info("Test for invalid timeframe completed successfully")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for invalid timeframe")
        raise

def test_memory_usage():
    """Test memory usage with large dataset"""
    logger.info("Starting test for memory usage")
    
    try:
        # Create large dataset (100K rows)
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='5min')
        df = pd.DataFrame({
            'open': np.random.randn(len(dates)) + 100,
            'high': np.random.randn(len(dates)) + 101,
            'low': np.random.randn(len(dates)) + 99,
            'close': np.random.randn(len(dates)) + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Get initial memory usage
        initial_memory = df.memory_usage().sum() / 1024 / 1024  # MB
        logger.debug(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Process data
        result = calculate_technical_indicators(df, {
            'RSI': {'period': 14},
            'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'SMA': {'periods': [20, 50]}
        })
        
        # Get final memory usage
        final_memory = result.memory_usage().sum() / 1024 / 1024  # MB
        logger.debug(f"Final memory usage: {final_memory:.2f} MB")
        
        # Check memory usage increase is reasonable (less than 3x)
        assert final_memory < initial_memory * 3, "Memory usage increased too much"
        
        logger.info("Test for memory usage completed successfully")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for memory usage")
        raise

def test_unexpected_data_format():
    """Test handling unexpected data formats"""
    logger.info("Starting test for unexpected data format")
    
    try:
        # Test with string data
        df = pd.DataFrame({
            'open': ['1', '2', '3'],
            'high': ['1.1', '2.1', '3.1'],
            'low': ['0.9', '1.9', '2.9'],
            'close': ['1', '2', '3'],
            'volume': ['100', '200', '300']
        })
        
        # Should raise exception for string data
        with pytest.raises(Exception):
            calculate_technical_indicators(df, {'RSI': {'period': 14}})
        
        # Test with missing required columns
        df = pd.DataFrame({
            'price': [1, 2, 3],
            'amount': [100, 200, 300]
        })
        
        # Should raise exception for missing columns
        with pytest.raises(Exception):
            calculate_technical_indicators(df, {'RSI': {'period': 14}})
        
        logger.info("Test for unexpected data format completed successfully")
    
    except Exception as e:
        logger.exception("An unexpected error occurred in test for unexpected data format")
        raise
