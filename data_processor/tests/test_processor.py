"""
اختبارات وحدة processor.py
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from ..processor import DataProcessor

@pytest.fixture
def processor():
    """إنشاء معالج للاختبار"""
    return DataProcessor()

@pytest.fixture
def sample_ohlcv():
    """إنشاء بيانات OHLCV للاختبار"""
    return [
        [1609459200000, 100, 101, 99, 100, 1000],  # 2021-01-01
        [1609462800000, 101, 102, 100, 101, 1100],  # 2021-01-01
        [1609466400000, 101, 103, 100, 102, 1200],  # 2021-01-01
    ]

def test_fetch_historical_data(processor, sample_ohlcv):
    """اختبار جلب البيانات التاريخية"""
    # تجهيز mock للـ exchange
    processor.exchange.fetch_ohlcv = Mock(return_value=sample_ohlcv)
    
    df = processor.fetch_historical_data('BTC/USDT', '1h')
    
    # التحقق من استدعاء الدالة بشكل صحيح
    processor.exchange.fetch_ohlcv.assert_called_once_with('BTC/USDT', '1h', limit=1000)
    
    # التحقق من تنسيق البيانات
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'timestamp'
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_process_data(processor):
    """اختبار معالجة البيانات"""
    # إنشاء بيانات اختبار
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
    df = pd.DataFrame({
        'open': range(len(dates)),
        'high': range(len(dates)),
        'low': range(len(dates)),
        'close': range(len(dates)),
        'volume': range(len(dates))
    }, index=dates)
    
    result = processor.process_data(df)
    
    # التحقق من وجود المؤشرات الفنية
    assert 'RSI_14' in result.columns
    assert 'MACD' in result.columns
    
    # التحقق من تطبيع البيانات
    feature_cols = [col for col in result.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in feature_cols:
        assert result[col].min() >= -1
        assert result[col].max() <= 1

@patch('concurrent.futures.ThreadPoolExecutor')
def test_process_all_pairs(mock_executor, processor):
    """اختبار معالجة جميع الأزواج"""
    mock_executor.return_value.__enter__.return_value = Mock()
    
    processor.process_all_pairs()
    
    # التحقق من استخدام ThreadPoolExecutor
    mock_executor.assert_called_once()

def test_process_pair_error_handling(processor):
    """اختبار معالجة الأخطاء"""
    # تجهيز mock للـ exchange يرجع خطأ
    processor.exchange.fetch_ohlcv = Mock(side_effect=Exception("API Error"))
    
    result = processor.process_pair('BTC/USDT', '1h')
    
    # التحقق من إرجاع None في حالة الخطأ
    assert result is None
