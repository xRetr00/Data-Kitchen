"""
المعالج الرئيسي للبيانات
"""

import pandas as pd
from typing import Optional, Generator
from datetime import datetime
import ccxt
from concurrent.futures import ThreadPoolExecutor
from .data_storage import DataStorage
from .data_utils import calculate_technical_indicators, normalize_features, handle_missing_data
from .logger import setup_logger
from .memory_utils import log_memory_usage, optimize_dataframe, check_memory_threshold
from .config import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    TRADING_PAIRS,
    TIMEFRAMES,
    CHUNK_SIZE,
    MAX_WORKERS,
    TECHNICAL_INDICATORS,
    START_DATE,
    END_DATE,
    TIMEFRAME_CHUNKS
)

logger = setup_logger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize Data Processor"""
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True
        })
        self.storage = DataStorage()
    
    def fetch_data_generator(
        self,
        pair: str,
        timeframe: str,
        chunk_size: int = 1000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        مولد لجلب البيانات على دفعات
        """
        try:
            since = int(START_DATE.timestamp() * 1000)
            end_timestamp = int(END_DATE.timestamp() * 1000)
            
            while since < end_timestamp:
                ohlcv = self.exchange.fetch_ohlcv(
                    pair,
                    timeframe,
                    since=since,
                    limit=chunk_size
                )
                
                if not ohlcv:
                    break
                
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # تحسين استخدام الذاكرة
                df = optimize_dataframe(df)
                
                yield df
                
                since = ohlcv[-1][0] + 1
                
                if since >= end_timestamp:
                    break
                
                # التحقق من استخدام الذاكرة
                if check_memory_threshold():
                    logger.warning("تجاوز حد الذاكرة، إيقاف جلب البيانات مؤقتاً")
                    break
        
        except Exception as e:
            logger.error(f"خطأ في جلب البيانات: {str(e)}")
            raise
    
    @log_memory_usage
    def process_chunk(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        معالجة دفعة من البيانات
        """
        try:
            # معالجة القيم المفقودة
            df = handle_missing_data(df)
            
            # حساب المؤشرات الفنية
            df = calculate_technical_indicators(df, TECHNICAL_INDICATORS)
            
            # تطبيع البيانات إذا كان مطلوباً
            if normalize:
                feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                df = normalize_features(df, feature_columns)
            
            return optimize_dataframe(df)
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الدفعة: {str(e)}")
            raise
    
    @log_memory_usage
    def process_pair(
        self,
        pair: str,
        timeframe: str,
        is_live: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        معالجة زوج عملات محدد
        """
        try:
            all_data = []
            
            # معالجة البيانات على دفعات
            for chunk in self.fetch_data_generator(pair, timeframe):
                processed_chunk = self.process_chunk(chunk, normalize=not is_live)
                all_data.append(processed_chunk)
                
                # حفظ الذاكرة
                if len(all_data) > 10:  # تجميع كل 10 دفعات
                    combined_data = pd.concat(all_data)
                    self.storage.save_data(combined_data, pair, timeframe)
                    all_data = []
            
            # معالجة الدفعات المتبقية
            if all_data:
                final_data = pd.concat(all_data)
                self.storage.save_data(final_data, pair, timeframe)
                return final_data
            
            return None
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الزوج {pair}: {str(e)}")
            return None
    
    def process_all_pairs(self, is_live: bool = False) -> None:
        """
        معالجة جميع أزواج العملات
        """
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for pair in TRADING_PAIRS:
                    for timeframe in TIMEFRAMES:
                        executor.submit(self.process_pair, pair, timeframe, is_live)
            
            logger.info("تمت معالجة جميع الأزواج بنجاح")
        
        except Exception as e:
            logger.error(f"خطأ في معالجة جميع الأزواج: {str(e)}")
            raise
