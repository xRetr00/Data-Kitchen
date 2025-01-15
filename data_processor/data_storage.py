"""
وحدة تخزين البيانات لمعالج البيانات
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional
from .logger import setup_logger
from .config import PARQUET_DIR

logger = setup_logger(__name__)

class DataStorage:
    def __init__(self):
        """تهيئة فئة تخزين البيانات"""
        os.makedirs(PARQUET_DIR, exist_ok=True)
        
    def _get_file_path(self, pair: str, timeframe: str) -> str:
        """
        إنشاء مسار الملف للزوج والإطار الزمني
        """
        return os.path.join(PARQUET_DIR, f"{pair.replace('/', '_')}_{timeframe}.parquet")
    
    def save_data(self, df: pd.DataFrame, pair: str, timeframe: str) -> None:
        """
        حفظ البيانات بتنسيق Parquet
        
        Args:
            df: إطار البيانات للحفظ
            pair: زوج العملات
            timeframe: الإطار الزمني
        """
        try:
            file_path = self._get_file_path(pair, timeframe)
            df.to_parquet(file_path, index=True)
            logger.info(f"تم حفظ البيانات بنجاح: {file_path}")
        except Exception as e:
            logger.error(f"خطأ في حفظ البيانات: {str(e)}")
            raise
    
    def load_data(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        تحميل البيانات من ملف Parquet
        
        Args:
            pair: زوج العملات
            timeframe: الإطار الزمني
            
        Returns:
            pd.DataFrame: إطار البيانات المحمل أو None في حالة الخطأ
        """
        try:
            file_path = self._get_file_path(pair, timeframe)
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                logger.info(f"تم تحميل البيانات بنجاح: {file_path}")
                return df
            logger.warning(f"ملف البيانات غير موجود: {file_path}")
            return None
        except Exception as e:
            logger.error(f"خطأ في تحميل البيانات: {str(e)}")
            return None
