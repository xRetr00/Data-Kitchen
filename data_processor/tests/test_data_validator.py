"""
Tests for data validation module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..data_validator import DataValidator, DataValidationError

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')
    df = pd.DataFrame({
        'close': np.random.randn(len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates)),
        'constant': 1.0,
        'mostly_missing': [np.nan] * (len(dates) - 2) + [1.0, 2.0],
        'outliers': np.random.randn(len(dates))
    }, index=dates)
    
    # Add some outliers
    df.loc[df.index[5], 'outliers'] = 1000.0
    
    # Add some duplicates
    df.iloc[10:15] = df.iloc[5:10].values
    
    return df

def test_validate_dataset(sample_data):
    """Test complete dataset validation"""
    validator = DataValidator()
    cleaned_data, report = validator.validate_dataset(sample_data)
    
    assert isinstance(cleaned_data, pd.DataFrame)
    assert isinstance(report, dict)
    assert 'constant' not in cleaned_data.columns
    assert 'mostly_missing' not in cleaned_data.columns
    assert report['dropped_columns']
    assert report['outliers_detected']
    assert cleaned_data.shape[0] < sample_data.shape[0]

def test_handle_missing_values():
    """Test missing value handling"""
    validator = DataValidator({'nan_threshold': 0.3})
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [np.nan] * 3 + [1],
        'c': [1, 2, 3, 4]
    })
    
    cleaned, issues = validator._handle_missing_values(df)
    assert 'b' not in cleaned.columns
    assert not cleaned.isnull().any().any()
    assert len(issues) > 0

def test_handle_duplicates():
    """Test duplicate handling"""
    validator = DataValidator({'duplicate_threshold': 0.1})
    df = pd.DataFrame({
        'a': [1, 2, 1, 2],
        'b': [3, 4, 3, 4]
    })
    
    cleaned, issues = validator._handle_duplicates(df)
    assert len(cleaned) < len(df)
    assert len(issues) > 0

def test_handle_constant_columns():
    """Test constant column detection"""
    validator = DataValidator({'constant_threshold': 0.95})
    df = pd.DataFrame({
        'constant': [1] * 10,
        'almost_constant': [1] * 9 + [1.01],
        'varying': np.random.randn(10)
    })
    
    cleaned, issues = validator._handle_constant_columns(df)
    assert 'constant' not in cleaned.columns
    assert 'almost_constant' not in cleaned.columns
    assert 'varying' in cleaned.columns

def test_handle_outliers():
    """Test outlier detection and handling"""
    validator = DataValidator({'outlier_threshold': 3.0})
    df = pd.DataFrame({
        'normal': np.random.randn(100),
        'with_outliers': np.random.randn(100)
    })
    df.loc[0, 'with_outliers'] = 1000.0
    
    cleaned, report = validator._handle_outliers(df)
    assert 'with_outliers' in report['outliers_detected']
    assert cleaned.loc[0, 'with_outliers'] < 1000.0

def test_check_future_leaks():
    """Test future data leak detection"""
    validator = DataValidator({'future_window': 1})
    df = pd.DataFrame({
        'price': np.random.randn(100),
        'future_price': np.random.randn(100),
        'next_value': np.random.randn(100)
    })
    
    leaks = validator._check_future_leaks(df)
    assert len(leaks) > 0
    assert any('future_price' in leak for leak in leaks)
    assert any('next_value' in leak for leak in leaks)

def test_validation_error():
    """Test validation error handling"""
    validator = DataValidator({'min_periods': 1000})
    df = pd.DataFrame({'a': range(10)})
    
    with pytest.raises(DataValidationError):
        validator.validate_dataset(df)
