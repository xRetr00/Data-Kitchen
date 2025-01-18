"""
Data Storage module for Data Processor
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path
from .logger import setup_logger
from .config import PARQUET_DIR
import shutil

logger = setup_logger(__name__)

class DataStorage:
    """Class for managing data storage and loading"""
    
    def __init__(self, storage_dir: str = PARQUET_DIR):
        """Initialize DataStorage class
        
        Args:
            storage_dir (str): Directory path for storing data. Default is PARQUET_DIR
        """
        self.storage_dir = Path(storage_dir)
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"Data storage initialized in: {storage_dir}")
        
        self.data_dir = storage_dir
        self.backup_dir = storage_dir / 'backups'
        self.max_backups = 3
        
        # إنشاء المجلدات إذا لم تكن موجودة
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def _get_file_path(self, pair: str, timeframe: str) -> Path:
        """Get the file path for a pair and timeframe"""
        # Replace '/' with '_' in pair name for file system compatibility
        safe_pair = pair.replace('/', '_')
        return self.storage_dir / f"{safe_pair}_{timeframe}.parquet"
        
    def load_data(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load data from parquet file
        
        Args:
            pair: Trading pair
            timeframe: Time frame
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        try:
            file_path = self._get_file_path(pair, timeframe)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
                
            df = pd.read_parquet(file_path)
            
            # تحويل timestamp إلى datetime إذا لم يكن كذلك
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                df.index = pd.to_datetime(df.index)
            
            logger.info(f"Loaded {len(df)} data points from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def check_data_integrity(self, file_path: str) -> Tuple[bool, str]:
        """التحقق من سلامة الملف"""
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
                
            # التحقق من حجم الملف
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "File is empty"
                
            # محاولة قراءة الملف
            pd.read_parquet(file_path, engine='fastparquet')
            
            return True, "File integrity check passed"
            
        except Exception as e:
            return False, f"File integrity check failed: {str(e)}"
            
    def check_duplicates(self, df: pd.DataFrame, file_path: str) -> Tuple[bool, pd.DataFrame]:
        """التحقق من البيانات المكررة"""
        try:
            # قراءة البيانات الموجودة
            if os.path.exists(file_path):
                existing_df = pd.read_parquet(file_path)
                
                # دمج البيانات والتحقق من التكرار
                combined_df = pd.concat([existing_df, df])
                duplicates = combined_df.duplicated(subset=['timestamp'], keep='first')
                
                if duplicates.any():
                    logger.warning(f"Found {duplicates.sum()} duplicate entries")
                    combined_df = combined_df[~duplicates]
                    
                return True, combined_df
                
            return False, df
            
        except Exception as e:
            logger.error(f"Error checking duplicates: {str(e)}")
            return False, df
            
    def create_backup(self, file_path: str) -> bool:
        """إنشاء نسخة احتياطية"""
        try:
            if not os.path.exists(file_path):
                return False
                
            # إنشاء اسم النسخة الاحتياطية
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{os.path.basename(file_path)}_{timestamp}.bak"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            # نسخ الملف
            shutil.copy2(file_path, backup_path)
            
            # حذف النسخ القديمة إذا تجاوز العدد الحد الأقصى
            self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return False
            
    def _cleanup_old_backups(self):
        """تنظيف النسخ الاحتياطية القديمة"""
        try:
            backups = sorted(
                [f for f in os.listdir(self.backup_dir) if f.endswith('.bak')],
                key=lambda x: os.path.getmtime(os.path.join(self.backup_dir, x))
            )
            
            while len(backups) > self.max_backups:
                oldest_backup = backups.pop(0)
                os.remove(os.path.join(self.backup_dir, oldest_backup))
                
        except Exception as e:
            logger.error(f"Error cleaning up backups: {str(e)}")
            
    def save_data(self, df: pd.DataFrame, pair: str, timeframe: str) -> bool:
        """
        Save data to parquet file
        
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
            
            # التحقق من التكرار
            has_existing, final_df = self.check_duplicates(df, file_path)
            
            if has_existing:
                # إنشاء نسخة احتياطية قبل التحديث
                self.create_backup(file_path)
                
            # حفظ البيانات
            final_df.to_parquet(file_path, index=False)
            
            # التحقق من سلامة الملف
            is_valid, msg = self.check_data_integrity(file_path)
            if not is_valid:
                logger.error(f"Data integrity check failed after save: {msg}")
                return False
                
            logger.info(f"Saved {len(final_df)} data points to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
            
    def delete_data(self, pair: str, timeframe: str) -> bool:
        """
        Delete data file
        
        Args:
            pair: Trading pair
            timeframe: Time frame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_file_path(pair, timeframe)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted {file_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting data: {str(e)}")
            return False
            
    def list_available_data(self) -> list:
        """
        List all available data files
        
        Returns:
            List of (pair, timeframe) tuples
        """
        try:
            available_data = []
            for file in os.listdir(self.storage_dir):
                if file.endswith('.parquet'):
                    # استخراج الزوج والإطار الزمني من اسم الملف
                    name = file.replace('.parquet', '')
                    parts = name.split('_')
                    if len(parts) >= 3:
                        pair = f"{parts[0]}/{parts[1]}"
                        timeframe = parts[2]
                        available_data.append((pair, timeframe))
            return available_data
            
        except Exception as e:
            logger.error(f"Error listing available data: {str(e)}")
            return []
