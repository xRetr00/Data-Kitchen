"""
Data Processor Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..processor import DataProcessor
from ..config import TECHNICAL_INDICATORS

@pytest.fixture
def processor(tmp_path):
    """Create a DataProcessor instance for testing"""
    return DataProcessor(str(tmp_path))

@pytest.fixture
def sample_data():
    """Generate sample data"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
    df = pd.DataFrame({
        'open': np.random.rand(len(dates)),
        'high': np.random.rand(len(dates)),
        'low': np.random.rand(len(dates)),
        'close': np.random.rand(len(dates)),
        'volume': np.random.rand(len(dates))
    }, index=dates)
    df.index.name = 'timestamp'
    return df

def test_process_chunk(processor, sample_data):
    """Test processing a chunk of data"""
    processed_data = processor.process_chunk(sample_data)
    
    # Check that technical indicators are present
    for indicator in TECHNICAL_INDICATORS:
        if indicator == 'RSI':
            assert f'RSI_{TECHNICAL_INDICATORS[indicator]["period"]}' in processed_data.columns
        elif indicator == 'MACD':
            macd_config = TECHNICAL_INDICATORS[indicator]
            macd_name = f'MACD_{macd_config["fast_period"]}_{macd_config["slow_period"]}_{macd_config["signal_period"]}'
            assert macd_name in processed_data.columns
            assert f'MACD_Signal_{macd_config["fast_period"]}_{macd_config["slow_period"]}_{macd_config["signal_period"]}' in processed_data.columns
        elif indicator in ['SMA', 'EMA']:
            for period in TECHNICAL_INDICATORS[indicator]['periods']:
                assert f'{indicator}_{period}' in processed_data.columns
    
    # Check that data is normalized
    for col in processed_data.columns:
        if col != 'timestamp':
            non_null_values = processed_data[col].dropna()
            if len(non_null_values) > 0:
                assert non_null_values.min() >= -1
                assert non_null_values.max() <= 1

def test_process_pair(processor, sample_data, mocker):
    """Test processing a trading pair"""
    # Mock fetch_data_generator to return sample data
    mocker.patch.object(
        processor,
        'fetch_data_generator',
        return_value=[sample_data]
    )
    
    pair = 'BTC/USDT'
    timeframe = '1h'
    result = processor.process_pair(pair, timeframe)
    
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert result.index.freq == '1h'
    
    # Check that data is saved
    loaded_data = processor.data_storage.load_data(pair, timeframe)
    pd.testing.assert_frame_equal(result, loaded_data)

def test_handle_missing_data(processor):
    """Test handling missing data"""
    # Create a DataFrame with missing values
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
    df = pd.DataFrame({
        'open': np.random.rand(len(dates)),
        'high': np.random.rand(len(dates)),
        'low': np.random.rand(len(dates)),
        'close': np.random.rand(len(dates)),
        'volume': np.random.rand(len(dates))
    }, index=dates)
    df.iloc[5:10] = np.nan
    
    processed_data = processor.process_chunk(df)
    
    # Check that basic columns have no missing values
    basic_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in basic_columns:
        assert not processed_data[col].isna().any()
