"""
Data Storage module for Data Processor
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from .logger import setup_logger
from .config import PARQUET_DIR, COMPRESSION_TYPE, CHUNK_SIZE, COMPRESSION_LEVEL, ROW_GROUP_SIZE, PAGE_SIZE
from .memory_utils import optimize_dataframe, check_memory_threshold
import shutil
import gc
import fastparquet
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import threading

logger = setup_logger(__name__)

class DataStorage:
    """Class for managing data storage and loading with optimized performance"""
    
    def __init__(self, storage_dir: str = PARQUET_DIR):
        """Initialize DataStorage class with optimized settings
        
        Args:
            storage_dir (str): Directory path for storing data. Default is PARQUET_DIR
        """
        self.storage_dir = Path(storage_dir)
        self.data_dir = storage_dir
        self.backup_dir = storage_dir / 'backups'
        self.cache_dir = storage_dir / 'cache'
        self.max_backups = 3
        self.lock = threading.Lock()
        self.cache: Dict[str, pd.DataFrame] = {}
        
        # Create necessary directories
        for directory in [self.storage_dir, self.backup_dir, self.cache_dir]:
            os.makedirs(directory, exist_ok=True)
            
        logger.info(f"Data storage initialized in: {storage_dir}")
        
    def _get_cache_key(self, pair: str, timeframe: str) -> str:
        """Get cache key for a pair and timeframe"""
        return f"{pair}_{timeframe}"
        
    def _get_file_path(self, pair: str, timeframe: str) -> Path:
        """Get the file path for a pair and timeframe"""
        safe_pair = pair.replace('/', '_')
        return self.storage_dir / f"{safe_pair}_{timeframe}.parquet"
        
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        try:
            # Convert float64 to float32
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                df[col] = df[col].astype(np.float32)
            
            # Convert int64 to int32 where possible
            int_cols = df.select_dtypes(include=['int64']).columns
            for col in int_cols:
                if df[col].min() > np.iinfo(np.int32).min and df[col].max() < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            return df
            
        except Exception as e:
            logger.error(f"Error optimizing DataFrame: {str(e)}")
            return df
            
    def load_data(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data with caching and optimization
        
        Args:
            pair: Trading pair
            timeframe: Time frame
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        try:
            cache_key = self._get_cache_key(pair, timeframe)
            
            # Check cache first
            with self.lock:
                if cache_key in self.cache:
                    logger.info(f"Retrieved {pair} {timeframe} from cache")
                    return self.cache[cache_key].copy()
            
            file_path = self._get_file_path(pair, timeframe)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
                
            # Load data in chunks if file is large
            if file_path.stat().st_size > CHUNK_SIZE:
                chunks = []
                for chunk in pq.read_table(file_path).to_batches(CHUNK_SIZE):
                    df_chunk = chunk.to_pandas()
                    chunks.append(df_chunk)
                    
                    if check_memory_threshold():
                        gc.collect()
                
                df = pd.concat(chunks, axis=0)
                del chunks
                gc.collect()
            else:
                df = pq.read_table(file_path).to_pandas()
            
            # Optimize DataFrame
            df = self._optimize_dataframe(df)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Cache the data
            with self.lock:
                self.cache[cache_key] = df.copy()
            
            logger.info(f"Loaded {len(df)} data points from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def save_data(self, df: pd.DataFrame, pair: str, timeframe: str) -> bool:
        """Save data with optimized performance and validation
        
        Args:
            df: DataFrame to save
            pair: Trading pair
            timeframe: Time frame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if df is None or df.empty:
                logger.warning(f"No data to save for {pair} {timeframe}")
                return False
                
            file_path = self._get_file_path(pair, timeframe)
            
            # Optimize DataFrame before saving
            df = self._optimize_dataframe(df)
            
            # Check for duplicates
            has_existing, final_df = self.check_duplicates(df, file_path)
            
            if has_existing:
                # Create backup before updating
                self.create_backup(file_path)
            
            # Convert to PyArrow Table for efficient writing
            table = pa.Table.from_pandas(final_df)
            
            # Save with optimized compression settings
            pq.write_table(
                table,
                file_path,
                compression=COMPRESSION_TYPE,
                compression_level=COMPRESSION_LEVEL,
                row_group_size=ROW_GROUP_SIZE,
                data_page_size=PAGE_SIZE,
                use_dictionary=True,
                write_statistics=True
            )
            
            # Verify integrity
            is_valid, msg = self.check_data_integrity(file_path)
            if not is_valid:
                logger.error(f"Data integrity check failed after save: {msg}")
                return False
                
            # Update cache
            cache_key = self._get_cache_key(pair, timeframe)
            with self.lock:
                self.cache[cache_key] = final_df.copy()
            
            logger.info(f"Saved {len(final_df)} data points to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
            
    def check_data_integrity(self, file_path: str) -> Tuple[bool, str]:
        """Verify file integrity with enhanced checks"""
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
                
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "File is empty"
                
            # Verify Parquet format
            try:
                parquet_file = pq.ParquetFile(file_path)
                metadata = parquet_file.metadata
                
                if metadata.num_rows == 0:
                    return False, "File contains no rows"
                    
                # Check column statistics
                for i in range(metadata.num_columns):
                    if metadata.row_group(0).column(i).statistics is None:
                        return False, f"Missing statistics for column {i}"
                
            except Exception as e:
                return False, f"Invalid Parquet format: {str(e)}"
            
            return True, "File integrity check passed"
            
        except Exception as e:
            return False, f"File integrity check failed: {str(e)}"
            
    def check_duplicates(self, df: pd.DataFrame, file_path: str) -> Tuple[bool, pd.DataFrame]:
        """Check for duplicates with optimized memory usage"""
        try:
            if not os.path.exists(file_path):
                return False, df
                
            # Load existing data in chunks
            existing_chunks = []
            for chunk in pq.read_table(file_path).to_batches(CHUNK_SIZE):
                existing_chunks.append(chunk.to_pandas())
                
                if check_memory_threshold():
                    gc.collect()
            
            existing_df = pd.concat(existing_chunks, axis=0)
            del existing_chunks
            gc.collect()
            
            # Combine and check duplicates
            combined_df = pd.concat([existing_df, df])
            duplicates = combined_df.duplicated(subset=['timestamp'], keep='first')
            
            if duplicates.any():
                logger.warning(f"Found {duplicates.sum()} duplicate entries")
                combined_df = combined_df[~duplicates]
                
            return True, combined_df
            
        except Exception as e:
            logger.error(f"Error checking duplicates: {str(e)}")
            return False, df
            
    def clear_cache(self):
        """Clear the data cache"""
        with self.lock:
            self.cache.clear()
        gc.collect()
        logger.info("Cache cleared")
        
    def get_storage_stats(self) -> Dict[str, int]:
        """Get storage statistics"""
        try:
            stats = {
                'total_files': 0,
                'total_size': 0,
                'backup_files': 0,
                'backup_size': 0
            }
            
            # Main storage stats
            for file in self.storage_dir.glob('*.parquet'):
                stats['total_files'] += 1
                stats['total_size'] += file.stat().st_size
            
            # Backup stats
            for file in self.backup_dir.glob('*.bak'):
                stats['backup_files'] += 1
                stats['backup_size'] += file.stat().st_size
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {}
