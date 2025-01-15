"""
Main Data Processor
"""

import pandas as pd
from typing import Optional, List
from datetime import datetime
import ccxt
from concurrent.futures import ThreadPoolExecutor
from .data_storage import DataStorage
from .data_utils import calculate_technical_indicators, normalize_features, handle_missing_data
from .logger import setup_logger
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
    
    def fetch_historical_data(
        self,
        pair: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch Historical Data
        
        Args:
            pair: Trading Pair
            timeframe: Timeframe
            
        Returns:
            pd.DataFrame: DataFrame with Historical Data
        """
        try:
            # Convert dates to timestamp in milliseconds
            since = int(START_DATE.timestamp() * 1000)
            end_timestamp = int(END_DATE.timestamp() * 1000)
            
            # Fetch data in chunks
            all_ohlcv = []
            while since < end_timestamp:
                ohlcv = self.exchange.fetch_ohlcv(
                    pair,
                    timeframe,
                    since=since,
                    limit=1000  # Maximum allowed by Binance
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # Next timestamp after the last chunk
                
                # Check if we have reached the end date
                if since >= end_timestamp:
                    break
            
            if not all_ohlcv:
                logger.warning(f"No data found for pair {pair} in timeframe {timeframe}")
                return None
            
            # Convert data to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Check if the number of candles is as expected
            expected_candles = TIMEFRAME_CHUNKS[timeframe]
            if len(df) < expected_candles * 0.9:  # Allow 10% data loss
                logger.warning(
                    f"Number of candles received ({len(df)}) is less than expected ({expected_candles}) "
                    f"for pair {pair} in timeframe {timeframe}"
                )
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None
    
    def process_data(
        self,
        df: pd.DataFrame,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Process Raw Data
        
        Args:
            df: Raw DataFrame
            normalize: Normalize Data
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            # Handle missing values
            df = handle_missing_data(df)
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df, TECHNICAL_INDICATORS)
            
            # Normalize data if required
            if normalize:
                feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                df = normalize_features(df, feature_columns)
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def process_pair(
        self,
        pair: str,
        timeframe: str,
        is_live: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Process a Specific Trading Pair
        
        Args:
            pair: Trading Pair
            timeframe: Timeframe
            is_live: Is it live data?
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        try:
            # Fetch data
            df = self.fetch_historical_data(pair, timeframe)
            if df is None:
                return None
            
            # Process data
            df = self.process_data(df, normalize=not is_live)
            
            # Save data
            self.storage.save_data(df, pair, timeframe)
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing pair {pair}: {str(e)}")
            return None
    
    def process_all_pairs(self, is_live: bool = False) -> None:
        """
        Process All Trading Pairs
        
        Args:
            is_live: Is it live data?
        """
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for pair in TRADING_PAIRS:
                    for timeframe in TIMEFRAMES:
                        executor.submit(self.process_pair, pair, timeframe, is_live)
            
            logger.info("All pairs processed successfully")
        
        except Exception as e:
            logger.error(f"Error processing all pairs: {str(e)}")
            raise
