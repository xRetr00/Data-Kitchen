"""
Data Processor
"""

import pandas as pd
from typing import Optional, Generator, List
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
    """Data Processor"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize Data Processor
        
        Args:
            storage_dir: Storage directory path. If not specified, default path will be used
        """
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True
        })
        self.data_storage = DataStorage(storage_dir)
        logger.info(f"Data Processor initialized")
    
    def fetch_data_generator(
        self,
        pair: str,
        timeframe: str,
        chunk_size: int = CHUNK_SIZE
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Data fetcher generator
        
        Args:
            pair: Trading pair
            timeframe: Time frame
            chunk_size: Chunk size
            
        Yields:
            DataFrame with OHLCV data
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
                
                # Set time frame
                df = df.asfreq(timeframe)
                
                # Optimize memory usage
                df = optimize_dataframe(df)
                
                yield df
                
                since = ohlcv[-1][0] + 1
                
                if since >= end_timestamp:
                    break
                
                # Check memory usage
                if check_memory_threshold():
                    logger.warning("Memory usage exceeded, temporarily stopping data fetch")
                    break
        
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    @log_memory_usage
    def process_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a chunk of data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            pd.DataFrame: Processed data with technical indicators
        """
        try:
            # Make sure data is numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)

            # Handle missing values
            df = handle_missing_data(df)
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df, TECHNICAL_INDICATORS)
            
            # Normalize features
            feature_columns = [col for col in df.columns if col not in ['timestamp']]
            df = normalize_features(df, feature_columns)
            
            return optimize_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            raise
    
    @log_memory_usage
    def process_pair(
        self,
        pair: str,
        timeframe: str,
        is_live: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Process a specific trading pair
        
        Args:
            pair: Trading pair
            timeframe: Time frame
            is_live: Whether this is a live process or not
            
        Returns:
            DataFrame with processed data, or None if error occurred
        """
        try:
            all_data = []
            chunk_count = 0
            
            # Process data in chunks
            for chunk in self.fetch_data_generator(pair, timeframe):
                processed_chunk = self.process_chunk(chunk)
                all_data.append(processed_chunk)
                chunk_count += 1
                
                # Save data every certain number of chunks
                if chunk_count >= TIMEFRAME_CHUNKS.get(timeframe, 10):
                    combined_data = pd.concat(all_data)
                    # Preserve time frame
                    combined_data = combined_data.asfreq(timeframe)
                    self.data_storage.save_data(combined_data, pair, timeframe)
                    all_data = []
                    chunk_count = 0
                    logger.info(f"Saved {len(combined_data)} data points for {pair} and {timeframe}")
            
            # Process remaining chunks
            if all_data:
                final_data = pd.concat(all_data)
                # Preserve time frame
                final_data = final_data.asfreq(timeframe)
                self.data_storage.save_data(final_data, pair, timeframe)
                logger.info(f"Saved {len(final_data)} remaining data points for {pair} and {timeframe}")
                return final_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing {pair}: {str(e)}")
            return None
    
    def configure(self, indicators: list, missing_threshold: float, chunk_size: int):
        """
        Configure the data processor

        Args:
            indicators: List of technical indicators to calculate
            missing_threshold: Threshold for handling missing data
            chunk_size: Size of data chunks to process
        """
        self.indicators_config = {}
        
        # Configure indicators
        if 'RSI' in indicators:
            self.indicators_config['RSI'] = {'period': 14}
        
        if 'MACD' in indicators:
            self.indicators_config['MACD'] = {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            }
        
        if 'SMA' in indicators:
            self.indicators_config['SMA'] = {'periods': [20, 50, 200]}
            
        if 'EMA' in indicators:
            self.indicators_config['EMA'] = {'periods': [20, 50, 200]}
        
        # Configure other parameters
        self.missing_threshold = missing_threshold
        self.chunk_size = chunk_size
        
        logger.info(f"Data processor configured with: indicators={indicators}, "
                   f"missing_threshold={missing_threshold}, chunk_size={chunk_size}")
    
    def process_all_pairs(self, pairs: List[str], timeframes: List[str]) -> None:
        """Process all trading pairs and time frames"""
        for pair in pairs:
            for timeframe in timeframes:
                self.process_pair(pair, timeframe)
