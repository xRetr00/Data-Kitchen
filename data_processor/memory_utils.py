"""
Memory Management Module
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
    Get current memory usage
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def log_memory_usage(func):
    """
    Decorator to log memory usage before and after function execution
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()  # clean up memory before execution
        
        # log memory usage before execution
        start_mem = get_memory_usage()
        logger.debug(f"Start {func.__name__}: memory usage {start_mem:.2f} MB")
        
        try:
            result = func(*args, **kwargs)
            
            # clean up memory after execution
            gc.collect()
            
            # log memory usage after execution
            end_mem = get_memory_usage()
            logger.debug(
                f"End {func.__name__}: memory usage {end_mem:.2f} MB "
                f"(change: {end_mem - start_mem:.2f} MB)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
            
    return wrapper

def check_memory_threshold(threshold_mb=1000):
    """
    Check if memory usage exceeds threshold
    """
    current_mem = get_memory_usage()
    if current_mem > threshold_mb:
        logger.warning(
            f"Warning: memory usage ({current_mem:.2f} MB) "
            f"exceeds threshold ({threshold_mb} MB)"
        )
        return True
    return False

def optimize_dataframe(df):
    """
    Optimize memory usage in DataFrame
    """
    memory_usage_start = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # convert numeric columns to float32 instead of float64
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        df[col] = df[col].astype('float32')
    
    memory_usage_end = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.debug(
        f"Optimized DataFrame: from {memory_usage_start:.2f} MB "
        f"to {memory_usage_end:.2f} MB"
    )
    
    return df
