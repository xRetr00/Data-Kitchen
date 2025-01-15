"""
Data Storage module for Data Processor
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional
from .logger import setup_logger
from .config import PARQUET_DIR

logger = setup_logger(__name__)

class DataStorage:
    """Class for managing data storage and loading"""
    
    def __init__(self, storage_dir: str = PARQUET_DIR):
        """Initialize DataStorage class
        
        Args:
            storage_dir (str): Directory path for storing data. Default is PARQUET_DIR
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"Data storage initialized in: {storage_dir}")
        
    def _get_file_path(self, pair: str, timeframe: str) -> str:
        """Generate file path for given pair and timeframe"""
        filename = f"{pair.replace('/', '_')}_{timeframe}.parquet"
        return os.path.join(self.storage_dir, filename)
    
    def save_data(self, df: pd.DataFrame, pair: str, timeframe: str) -> None:
        """Save data to a parquet file"""
        file_path = self._get_file_path(pair, timeframe)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # copy DataFrame to avoid modifying original data
        df = df.copy()
        
        # ensure 'timestamp' column is the index
        if df.index.name != 'timestamp':
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                df.index.name = 'timestamp'
        
        # ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # save frequency as a string in the attributes
        freq_str = df.index.freqstr if df.index.freq is not None else None
        df.attrs['freq'] = freq_str
        
        df.to_parquet(file_path)
        logger.info(f"Data saved successfully: {file_path}")

    def load_data(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Load data from a parquet file"""
        file_path = self._get_file_path(pair, timeframe)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        
        # ensure 'timestamp' column is the index
        if df.index.name != 'timestamp':
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                df.index.name = 'timestamp'
        
        # ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # restore frequency from the attributes
        if 'freq' in df.attrs and df.attrs['freq'] is not None:
            df.index.freq = pd.tseries.frequencies.to_offset(df.attrs['freq'])
        
        logger.info(f"Data loaded successfully: {file_path}")
        return df
