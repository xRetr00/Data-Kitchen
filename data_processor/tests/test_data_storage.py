"""
اختبارات وحدة data_storage.py
"""

import pytest
import pandas as pd
import os
from ..data_storage import DataStorage
from ..config import PARQUET_DIR

@pytest.fixture
def storage():
    """إنشاء كائن DataStorage للاختبار"""
    return DataStorage()

@pytest.fixture
def sample_data():
    """إنشاء بيانات اختبار"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
    return pd.DataFrame({
        'close': range(len(dates)),
        'volume': range(len(dates))
    }, index=dates)

def test_save_and_load_data(storage, sample_data):
    """اختبار حفظ وتحميل البيانات"""
    pair = 'BTC_USDT'
    timeframe = '1h'
    
    # حفظ البيانات
    storage.save_data(sample_data, pair, timeframe)
    
    # التحقق من وجود الملف
    file_path = storage._get_file_path(pair, timeframe)
    assert os.path.exists(file_path)
    
    # تحميل البيانات
    loaded_data = storage.load_data(pair, timeframe)
    
    # التحقق من تطابق البيانات
    pd.testing.assert_frame_equal(sample_data, loaded_data)

def test_load_nonexistent_data(storage):
    """اختبار تحميل بيانات غير موجودة"""
    result = storage.load_data('NONEXISTENT', '1h')
    assert result is None

def test_save_data_creates_directory(storage, sample_data):
    """اختبار إنشاء المجلد عند الحفظ"""
    # حذف المجلد إذا كان موجوداً
    if os.path.exists(PARQUET_DIR):
        os.rmdir(PARQUET_DIR)
    
    # حفظ البيانات
    storage.save_data(sample_data, 'TEST', '1h')
    
    # التحقق من إنشاء المجلد
    assert os.path.exists(PARQUET_DIR)

def test_file_path_generation(storage):
    """اختبار إنشاء مسار الملف"""
    pair = 'BTC/USDT'
    timeframe = '1h'
    
    file_path = storage._get_file_path(pair, timeframe)
    
    # التحقق من تنسيق المسار
    assert 'BTC_USDT_1h.parquet' in file_path
    assert PARQUET_DIR in file_path
