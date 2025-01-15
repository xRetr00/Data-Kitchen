"""
وحدة إدارة الذاكرة
"""

import os
import gc
import psutil
import logging
from functools import wraps
from .logger import setup_logger

logger = setup_logger(__name__)

def get_memory_usage():
    """
    الحصول على استخدام الذاكرة الحالي
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def log_memory_usage(func):
    """
    مزخرف لتسجيل استخدام الذاكرة قبل وبعد تنفيذ الدالة
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()  # تنظيف الذاكرة قبل التنفيذ
        
        # تسجيل استخدام الذاكرة قبل التنفيذ
        start_mem = get_memory_usage()
        logger.debug(f"بداية {func.__name__}: استخدام الذاكرة {start_mem:.2f} MB")
        
        try:
            result = func(*args, **kwargs)
            
            # تنظيف الذاكرة بعد التنفيذ
            gc.collect()
            
            # تسجيل استخدام الذاكرة بعد التنفيذ
            end_mem = get_memory_usage()
            logger.debug(
                f"نهاية {func.__name__}: استخدام الذاكرة {end_mem:.2f} MB "
                f"(تغيير: {end_mem - start_mem:.2f} MB)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"خطأ في {func.__name__}: {str(e)}")
            raise
            
    return wrapper

def check_memory_threshold(threshold_mb=1000):
    """
    التحقق من تجاوز حد الذاكرة
    """
    current_mem = get_memory_usage()
    if current_mem > threshold_mb:
        logger.warning(
            f"تحذير: استخدام الذاكرة ({current_mem:.2f} MB) "
            f"تجاوز الحد ({threshold_mb} MB)"
        )
        return True
    return False

def optimize_dataframe(df):
    """
    تحسين استخدام الذاكرة في DataFrame
    """
    memory_usage_start = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # تحويل الأنواع العددية إلى float32 بدلاً من float64
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        df[col] = df[col].astype('float32')
    
    memory_usage_end = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.debug(
        f"تم تحسين DataFrame: من {memory_usage_start:.2f} MB "
        f"إلى {memory_usage_end:.2f} MB"
    )
    
    return df
