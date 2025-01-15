"""
اختبارات وحدة data_utils.py
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

# إعداد التسجيل للاختبارات
logger = setup_test_logger()

@pytest.fixture
def sample_data():
    """إنشاء بيانات اختبار"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(len(dates)) + 100,
        'high': np.random.randn(len(dates)) + 101,
        'low': np.random.randn(len(dates)) + 99,
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    return df

def test_calculate_technical_indicators(sample_data):
    """اختبار حساب المؤشرات الفنية"""
    logger.info("بدء اختبار calculate_technical_indicators")
    
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
        
        # التحقق من وجود الأعمدة المتوقعة
        assert f'RSI_14' in result.columns
        assert 'MACD' in result.columns
        assert 'MACD_Signal' in result.columns
        assert 'SMA_20' in result.columns
        assert 'SMA_50' in result.columns
        
        # التحقق من عدم وجود قيم NaN في نهاية البيانات
        assert not result.iloc[-1:].isna().any().any()
        
        logger.info("نجح اختبار calculate_technical_indicators")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار calculate_technical_indicators")
        raise
    
    logger.info("اكتمل اختبار calculate_technical_indicators بنجاح")

def test_normalize_features(sample_data):
    """اختبار تطبيع البيانات"""
    logger.info("بدء اختبار normalize_features")
    
    try:
        feature_columns = ['close', 'volume']
        result = normalize_features(sample_data, feature_columns)
        
        # التحقق من أن القيم بين 0 و 1
        for col in feature_columns:
            assert result[col].min() >= 0
            assert result[col].max() <= 1
        
        # التحقق من عدم تغيير الأعمدة الأخرى
        assert (result['open'] == sample_data['open']).all()
        
        logger.info("نجح اختبار normalize_features")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار normalize_features")
        raise
    
    logger.info("اكتمل اختبار normalize_features بنجاح")

def test_handle_missing_data():
    """اختبار معالجة البيانات المفقودة"""
    logger.info("بدء اختبار handle_missing_data")
    
    try:
        # إنشاء بيانات مع قيم مفقودة
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [np.nan, np.nan, np.nan, 4, 5]
        })
        
        result = handle_missing_data(df, threshold=0.5)
        
        # التحقق من حذف العمود C (أكثر من 50% قيم مفقودة)
        assert 'C' not in result.columns
        
        # التحقق من ملء القيم المفقودة في الأعمدة المتبقية
        assert not result.isna().any().any()
        
        logger.info("نجح اختبار handle_missing_data")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار handle_missing_data")
        raise
    
    logger.info("اكتمل اختبار handle_missing_data بنجاح")

def test_handle_missing_data_no_missing():
    """اختبار معالجة البيانات بدون قيم مفقودة"""
    logger.info("بدء اختبار handle_missing_data_no_missing")
    
    try:
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        result = handle_missing_data(df)
        
        # التحقق من عدم تغيير البيانات
        assert (result == df).all().all()
        
        logger.info("نجح اختبار handle_missing_data_no_missing")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار handle_missing_data_no_missing")
        raise
    
    logger.info("اكتمل اختبار handle_missing_data_no_missing بنجاح")

def test_validate_input_data():
    """اختبار التحقق من صحة البيانات المدخلة"""
    logger.info("بدء اختبار validate_input_data")
    
    try:
        # بيانات صحيحة
        valid_data = np.array([1.0, 2.0, 3.0])
        logger.debug(f"اختبار البيانات الصالحة: {valid_data}")
        validate_input_data(valid_data, "test")
        logger.info("نجح اختبار البيانات الصالحة")
        
        # بيانات فارغة
        logger.debug("اختبار البيانات الفارغة")
        with pytest.raises(IndicatorValidationError, match="The data is empty"):
            validate_input_data(np.array([]), "test")
        logger.info("نجح اختبار البيانات الفارغة، تم رفع الاستثناء المتوقع")
        
        # Data containing NaN
        invalid_data = np.array([1.0, np.nan, 3.0])
        logger.debug(f"اختبار البيانات مع NaN: {invalid_data}")
        with pytest.raises(IndicatorValidationError, match="Data contains NaN values"):
            validate_input_data(invalid_data, "test")
        logger.info("نجح اختبار البيانات مع NaN، تم رفع الاستثناء المتوقع")
        
        # Data containing infinite values
        invalid_data = np.array([1.0, np.inf, 3.0])
        logger.debug(f"اختبار البيانات مع قيم لا نهائية: {invalid_data}")
        with pytest.raises(IndicatorValidationError, match="Data contains infinite values"):
            validate_input_data(invalid_data, "test")
        logger.info("نجح اختبار البيانات مع قيم لا نهائية، تم رفع الاستثناء المتوقع")
        
        logger.info("نجح اختبار validate_input_data")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار validate_input_data")
        raise
    
    logger.info("اكتمل اختبار validate_input_data بنجاح")

def test_validate_rsi():
    """اختبار التحقق من صحة RSI"""
    logger.info("بدء اختبار validate_rsi")
    
    try:
        # قيم صحيحة
        valid_rsi = np.array([30.0, 50.0, 70.0])
        logger.debug(f"اختبار قيم RSI صالحة: {valid_rsi}")
        is_valid, _ = validate_rsi(valid_rsi)
        assert is_valid
        logger.info("نجح اختبار قيم RSI الصالحة")
        
        # قيم خارج النطاق
        invalid_rsi = np.array([-10.0, 50.0, 110.0])
        logger.debug(f"اختبار قيم RSI خارج النطاق: {invalid_rsi}")
        is_valid, error_msg = validate_rsi(invalid_rsi)
        assert not is_valid
        logger.info(f"نجح اختبار قيم RSI خارج النطاق، الرسالة: {error_msg}")
        
        # قيم NaN
        invalid_rsi = np.array([30.0, np.nan, 70.0])
        logger.debug(f"اختبار قيم RSI مع NaN: {invalid_rsi}")
        is_valid, error_msg = validate_rsi(invalid_rsi)
        assert not is_valid
        logger.info(f"نجح اختبار قيم RSI مع NaN، الرسالة: {error_msg}")
        
        logger.info("نجح اختبار validate_rsi")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار validate_rsi")
        raise
    
    logger.info("اكتمل اختبار validate_rsi بنجاح")

def test_validate_macd():
    """اختبار التحقق من صحة MACD"""
    logger.info("بدء اختبار validate_macd")
    
    try:
        # قيم صحيحة
        valid_macd = np.array([1.0, -1.0, 0.5])
        valid_signal = np.array([0.8, -0.8, 0.3])
        logger.debug(f"اختبار قيم MACD صالحة: {valid_macd}, قيم الإشارة: {valid_signal}")
        is_valid, _ = validate_macd(valid_macd, valid_signal)
        assert is_valid
        logger.info("نجح اختبار قيم MACD الصالحة")
        
        # قيم NaN
        invalid_macd = np.array([1.0, np.nan, 0.5])
        invalid_signal = np.array([0.8, -0.8, 0.3])
        logger.debug(f"اختبار قيم MACD مع NaN: {invalid_macd}, قيم الإشارة: {invalid_signal}")
        is_valid, error_msg = validate_macd(invalid_macd, invalid_signal)
        assert not is_valid
        logger.info(f"نجح اختبار قيم MACD مع NaN، الرسالة: {error_msg}")
        
        # قيم لا نهائية
        invalid_macd = np.array([1.0, np.inf, 0.5])
        invalid_signal = np.array([0.8, -0.8, 0.3])
        logger.debug(f"اختبار قيم MACD مع قيم لا نهائية: {invalid_macd}, قيم الإشارة: {invalid_signal}")
        is_valid, error_msg = validate_macd(invalid_macd, invalid_signal)
        assert not is_valid
        logger.info(f"نجح اختبار قيم MACD مع قيم لا نهائية، الرسالة: {error_msg}")
        
        logger.info("نجح اختبار validate_macd")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار validate_macd")
        raise
    
    logger.info("اكتمل اختبار validate_macd بنجاح")

def test_validate_sma():
    """اختبار التحقق من صحة SMA"""
    logger.info("بدء اختبار validate_sma")
    
    try:
        # قيم صحيحة
        valid_sma = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        logger.debug(f"اختبار قيم SMA صالحة: {valid_sma}")
        is_valid, _ = validate_sma(valid_sma, window=3, data_length=5)
        assert is_valid
        logger.info("نجح اختبار قيم SMA الصالحة")
        
        # عدد قيم NaN أكبر من المتوقع
        invalid_sma = np.array([np.nan, np.nan, np.nan, np.nan, 5.0])
        logger.debug(f"اختبار قيم SMA مع عدد قيم NaN أكبر من المتوقع: {invalid_sma}")
        is_valid, error_msg = validate_sma(invalid_sma, window=3, data_length=5)
        assert not is_valid
        logger.info(f"نجح اختبار قيم SMA مع عدد قيم NaN أكبر من المتوقع، الرسالة: {error_msg}")
        
        # قيم لا نهائية
        invalid_sma = np.array([np.nan, np.nan, np.inf, 4.0, 5.0])
        logger.debug(f"اختبار قيم SMA مع قيم لا نهائية: {invalid_sma}")
        is_valid, error_msg = validate_sma(invalid_sma, window=3, data_length=5)
        assert not is_valid
        logger.info(f"نجح اختبار قيم SMA مع قيم لا نهائية، الرسالة: {error_msg}")
        
        logger.info("نجح اختبار validate_sma")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار validate_sma")
        raise
    
    logger.info("اكتمل اختبار validate_sma بنجاح")

def test_calculate_technical_indicators_with_invalid_data():
    """اختبار حساب المؤشرات الفنية مع بيانات غير صالحة"""
    logger.info("بدء اختبار calculate_technical_indicators_with_invalid_data")
    
    try:
        # إنشاء بيانات مع قيم غير صالحة
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
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
        
        # يجب أن يرفع استثناء بسبب وجود NaN في البيانات
        with pytest.raises(IndicatorValidationError, match="contains NaN values"):
            calculate_technical_indicators(df, settings)
        logger.info("نجح اختبار calculate_technical_indicators_with_invalid_data")
    
    except Exception as e:
        logger.exception("حدث خطأ غير متوقع في اختبار calculate_technical_indicators_with_invalid_data")
        raise
    
    logger.info("اكتمل اختبار calculate_technical_indicators_with_invalid_data بنجاح")
