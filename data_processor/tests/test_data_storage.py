"""
Tests for data_storage.py module
"""

import os
import shutil
import pandas as pd
import numpy as np
import pytest
from ..data_storage import DataStorage, PARQUET_DIR

@pytest.fixture
def storage(tmp_path):
    """Create DataStorage object for testing"""
    return DataStorage(str(tmp_path))

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='h')
    df = pd.DataFrame({
        'open': np.random.rand(len(dates)),
        'high': np.random.rand(len(dates)),
        'low': np.random.rand(len(dates)),
        'close': np.random.rand(len(dates)),
        'volume': np.random.rand(len(dates))
    }, index=dates)
    df.index.name = 'timestamp'
    return df

def test_save_and_load_data(storage, sample_data):
    """Test saving and loading data"""
    pair = "BTC/USDT"
    timeframe = "1h"
    
    # save data
    storage.save_data(sample_data, pair, timeframe)
    
    # load data
    loaded_df = storage.load_data(pair, timeframe)
    
    # check that data is the same
    pd.testing.assert_frame_equal(sample_data, loaded_df)
    
    # check that frequency is saved correctly
    assert loaded_df.index.freq == sample_data.index.freq

def test_load_nonexistent_data(storage):
    """Test loading nonexistent data"""
    with pytest.raises(FileNotFoundError):
        storage.load_data('NONEXISTENT', '1h')

def test_save_data_creates_directory(tmp_path):
    """Test that directory is created when saving data"""
    storage = DataStorage(str(tmp_path))
    pair = 'BTC/USDT'
    timeframe = '1h'
    
    # make sure directory does not exist
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    
    # save data
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='h')
    df = pd.DataFrame({
        'open': np.random.rand(len(dates)),
        'high': np.random.rand(len(dates)),
        'low': np.random.rand(len(dates)),
        'close': np.random.rand(len(dates)),
        'volume': np.random.rand(len(dates))
    }, index=dates)
    df.index.name = 'timestamp'
    
    storage.save_data(df, pair, timeframe)
    
    # check that directory is created
    assert os.path.exists(tmp_path)
    assert os.path.exists(os.path.join(tmp_path, f"{pair.replace('/', '_')}_{timeframe}.parquet"))

def test_file_path_generation(tmp_path):
    """Test file path generation"""
    storage = DataStorage(str(tmp_path))
    pair = 'BTC/USDT'
    timeframe = '1h'
    expected_path = os.path.join(tmp_path, f"{pair.replace('/', '_')}_{timeframe}.parquet")
    assert storage._get_file_path(pair, timeframe) == expected_path
