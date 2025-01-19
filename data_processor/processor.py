"""
Data Processor
"""
import sys
import pandas as pd
from typing import Optional, Generator, List, Dict
from datetime import datetime, timedelta
import ccxt
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
from .data_storage import DataStorage
from .data_utils import calculate_technical_indicators, normalize_features, handle_missing_data
from .logger import setup_logger
from .memory_utils import log_memory_usage, optimize_dataframe, check_memory_threshold
from .config import (
    EXCHANGES,
    MAX_RETRIES,
    RETRY_DELAY,
    TRADING_PAIRS,
    TIMEFRAMES,
    MAX_WORKERS,
    START_DATE,
    END_DATE,
    PARQUET_DIR,
    TECHNICAL_INDICATORS,
    MAX_CHUNK_SIZE,
    CHUNK_TIMEOUT,
    DOWNLOAD_TIMEOUT
)
from .data_validator import DataValidator
from .sentiment_analyzer import SentimentAnalyzer
import time
import numpy as np
from pathlib import Path
import gc

logger = setup_logger(__name__)

class DataProcessor:
    """Data Processor"""
    
    def __init__(self, storage_dir: Optional[str] = PARQUET_DIR):
        """Initialize the data processor with multiple exchanges"""
        self.exchanges = {}
        
        # Exchange configurations
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        
        # Trading pairs and timeframes
        self.trading_pairs = TRADING_PAIRS
        self.timeframes = TIMEFRAMES
        
        # Data period configurations
        self.years_of_data = (END_DATE - START_DATE).days // 365
        self.end_date = END_DATE
        self.start_date = START_DATE
        
        # Processing configurations
        self.max_workers = MAX_WORKERS
        
        # Chunk sizes based on timeframe
        self.chunk_sizes = {
            'okx': {'recent': 300, 'historic': 100},
            'default': {'recent': 1000, 'historic': 1000}
        }
        
        # Technical indicators configuration
        self.indicators_config = TECHNICAL_INDICATORS
        
        # Initialize components
        self._initialize_exchanges()
        self.data_storage = DataStorage(storage_dir)
        self.validator = DataValidator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.console = Console()
        
        logger.info(f"Data Processor initialized with storage directory: {storage_dir}")
        
    def _initialize_exchange(self, exchange_id: str):
        """Initialize a single exchange"""
        try:
            if exchange_id not in ccxt.exchanges:
                logger.error(f"Exchange {exchange_id} not found in ccxt")
                return None

            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })

            # Load markets
            exchange.load_markets()
            
            return {
                'instance': exchange,
                'enabled': True
            }

        except Exception as e:
            logger.error(f"Error initializing {exchange_id}: {str(e)}")
            return None

    def _initialize_exchanges(self):
        """Initialize all enabled exchanges"""
        for exchange_id, config in EXCHANGES.items():
            if not config['enabled']:
                continue
                
            try:
                # Initialize exchange
                exchange = self._initialize_exchange(exchange_id)
                if not exchange:
                    logger.error(f"Failed to initialize {exchange_id}")
                    continue
                
                # Store exchange instance and config
                self.exchanges[exchange_id] = {
                    'instance': exchange['instance'],
                    'enabled': True,
                    'priority': config['priority']
                }
                
                logger.info(f"Successfully initialized {exchange_id} exchange")
                
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id} exchange: {str(e)}")
                self.exchanges[exchange_id] = {
                    'instance': None,
                    'enabled': False,
                    'priority': config['priority']
                }
                
    def load_config(self) -> Dict:
        """
        Load configuration for data processing
        
        Returns:
            Dictionary containing configuration parameters
        """
        return {
            'pairs': self.trading_pairs,
            'timeframes': self.timeframes,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'max_workers': self.max_workers,
            'indicators_config': self.indicators_config
        }
        
    def _get_timeframe_duration(self, timeframe: str) -> int:
        """Calculate duration in milliseconds for a given timeframe"""
        if timeframe == '1h':
            return 1000 * 60 * 60  # 1 hour in milliseconds
        elif timeframe == '4h':
            return 1000 * 60 * 60 * 4  # 4 hours in milliseconds
        else:  # 1d
            return 1000 * 60 * 60 * 24  # 24 hours in milliseconds
            
    def _calculate_chunks(self, timeframe: str, exchange_id: str) -> tuple:
        """Calculate chunk sizes and number of chunks needed"""
        # Get exchange limits
        limits = self.chunk_sizes.get(exchange_id, self.chunk_sizes['default'])
        recent_limit = limits['recent']
        historic_limit = limits['historic']
        
        # Calculate total candles needed
        candles_per_year = {
            '1h': 365 * 24,
            '4h': 365 * 6,
            '1d': 365
        }
        total_candles = self.years_of_data * candles_per_year[timeframe]
        
        # Calculate chunks needed (total_candles / historic_limit, rounded up)
        total_chunks = (total_candles + historic_limit - 1) // historic_limit
        
        return recent_limit, historic_limit, total_chunks
        
    def _fetch_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from exchange"""
        for exchange_id, exchange in self.exchanges.items():
            if not exchange['enabled']:
                continue
                
            try:
                chunk_size = self._calculate_chunks(timeframe, exchange_id)
                total_time = int((self.end_date - self.start_date).total_seconds() * 1000)
                
                recent_candles = []
                current_timestamp = int(self.end_date.timestamp() * 1000)
                start_timestamp = int(self.start_date.timestamp() * 1000)
                
                # Add timeout for the entire operation
                start_time = time.time()
                empty_chunks_count = 0
                
                while current_timestamp > start_timestamp:
                    # Check if we've exceeded the maximum operation time
                    if time.time() - start_time > DOWNLOAD_TIMEOUT:
                        logger.error(f"Operation timeout for {symbol} {timeframe}")
                        return None
                        
                    # If we get too many empty chunks, skip this pair
                    if empty_chunks_count > 10:
                        logger.error(f"Too many empty chunks for {symbol} {timeframe}, skipping...")
                        return None
                        
                    try:
                        # Get chunk size from config
                        chunk_limit = MAX_CHUNK_SIZE.get(symbol, MAX_CHUNK_SIZE['default'])
                        
                        # Calculate chunk start time
                        chunk_duration = chunk_limit * self._get_timeframe_duration(timeframe)
                        chunk_start = current_timestamp - chunk_duration
                        
                        # Make sure we don't go before start_timestamp
                        if chunk_start < start_timestamp:
                            chunk_start = start_timestamp
                            chunk_limit = min(
                                chunk_limit,
                                int((current_timestamp - start_timestamp) / self._get_timeframe_duration(timeframe))
                            )
                        
                        # Fetch chunk with retries
                        retry_count = 0
                        chunk_data = None
                        
                        while retry_count < MAX_RETRIES and not chunk_data:
                            try:
                                # Set request timeout
                                exchange['instance'].timeout = CHUNK_TIMEOUT * 1000
                                
                                candles = exchange['instance'].fetch_ohlcv(
                                    symbol,
                                    timeframe=timeframe,
                                    since=chunk_start,
                                    limit=chunk_limit
                                )
                                
                                if candles and len(candles):
                                    # Convert to DataFrame
                                    chunk_df = pd.DataFrame(
                                        candles,
                                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                                    )
                                    
                                    # التحقق من تسلسل التواريخ
                                    chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms')
                                    if not chunk_df['timestamp'].is_monotonic_increasing:
                                        logger.error(f"Non-monotonic timestamps in chunk for {symbol} {timeframe}")
                                        continue
                                        
                                    recent_candles.append(chunk_df)
                                    current_timestamp = chunk_start
                                    empty_chunks_count = 0  # Reset counter on success
                                    chunk_data = True
                                else:
                                    retry_count += 1
                                    empty_chunks_count += 1
                                    logger.warning(f"No data for {symbol} at {chunk_start}, attempt {retry_count}/{MAX_RETRIES}")
                                    time.sleep(RETRY_DELAY)
                                    
                            except Exception as e:
                                retry_count += 1
                                logger.error(f"Error fetching chunk: {str(e)}, attempt {retry_count}/{MAX_RETRIES}")
                                time.sleep(RETRY_DELAY)
                        
                        if not chunk_data:
                            # If all retries failed, move back by one timeframe
                            current_timestamp -= self._get_timeframe_duration(timeframe)
                            
                    except Exception as e:
                        logger.error(f"Error in chunk processing: {str(e)}")
                        return None
                
                if not recent_candles:
                    logger.error(f"No valid data collected for {symbol} {timeframe}")
                    return None
                
                # Combine all chunks
                try:
                    df = pd.concat(recent_candles, ignore_index=True)
                    
                    # التحقق من صحة البيانات
                    if df.isnull().any().any():
                        logger.error(f"Found null values in data for {symbol} {timeframe}")
                        return None
                        
                    # التحقق من صحة قيم High/Low فقط
                    if not (df['high'] >= df['low']).all():
                        logger.error(f"Invalid high/low values in data for {symbol} {timeframe}")
                        return None
                        
                    return df
                    
                except Exception as e:
                    logger.error(f"Error combining chunks for {symbol} {timeframe}: {str(e)}")
                    return None
                
            except Exception as e:
                logger.error(f"Failed to fetch data from {exchange_id}: {str(e)}")
                continue
                
        logger.error(f"Failed to fetch data from all exchanges for {symbol} {timeframe}")
        return None
            
    def get_available_exchanges(self) -> List[str]:
        """Return list of available exchanges"""
        return list(self.exchanges.keys())
        
    def get_exchange_status(self) -> Dict[str, bool]:
        """Check the status of all exchanges"""
        status = {}
        test_pair = self.trading_pairs[0]
        for exchange_id, exchange in self.exchanges.items():
            try:
                exchange['instance'].fetch_ticker(test_pair)
                status[exchange_id] = True
            except Exception:
                status[exchange_id] = False
        return status
        
    
    @log_memory_usage
    def process_pair(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Process a single pair with memory optimization"""
        try:
            # Check memory before starting
            if check_memory_threshold():
                gc.collect()
                logger.warning("Memory threshold reached - garbage collection triggered")
            
            # Get raw data
            data = self._get_raw_data(pair, timeframe)
            if data is None or data.empty:
                logger.error(f"No data available for {pair} {timeframe}")
                return None
            
            # Validate data structure
            is_valid, msg = self.validator.validate_data_structure(data)
            if not is_valid:
                logger.error(f"Data validation failed for {pair} {timeframe}: {msg}")
                return None
            
            # Process in chunks to avoid memory issues
            chunk_size = 10000  # Adjust based on available memory
            num_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size else 0)
            
            processed_chunks = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(data))
                
                # Process chunk
                chunk = data.iloc[start_idx:end_idx].copy()
                processed_chunk = self._process_chunk(chunk, pair, timeframe)
                
                if processed_chunk is not None:
                    processed_chunks.append(processed_chunk)
                    
                # Clear memory after each chunk
                del chunk
                gc.collect()
            
            # Combine processed chunks
            if not processed_chunks:
                logger.error(f"No valid processed chunks for {pair} {timeframe}")
                return None
                
            result = pd.concat(processed_chunks, axis=0)
            
            # Clear memory
            del processed_chunks
            gc.collect()
            
            # Final validation
            is_valid, stats = self.validator.validate_data_quality(result)
            if not is_valid:
                logger.error(f"Final validation failed for {pair} {timeframe}: {stats}")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pair} {timeframe}: {str(e)}")
            return None
            
    def _process_chunk(self, data: pd.DataFrame, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Process a single chunk of data
        
        Args:
            data: DataFrame to process
            pair: Trading pair
            timeframe: Time frame
            
        Returns:
            Processed DataFrame or None if error
        """
        try:
            if data is None or data.empty:
                logger.error(f"No data to process for {pair} {timeframe}")
                return None
                
            # Make copy to avoid modifying original
            df = data.copy()
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by time
            df = df.sort_index()
            
            # Convert OHLCV columns to float32
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(np.float32)
            
            # Add technical indicators
            df = calculate_technical_indicators(df, self.indicators_config)
            
            # Handle missing data
            df = handle_missing_data(df)
            
            # Drop any remaining NaN values
            df = df.dropna()
            
            # Optimize memory usage
            df = optimize_dataframe(df)
            
            logger.info(f"Processed chunk for {pair} {timeframe}, shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing chunk for {pair} {timeframe}: {str(e)}")
            return None
            
    def _get_raw_data(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get raw data with validation"""
        try:
            # Try to load from storage first
            data = self.data_storage.load_data(pair, timeframe)
            if data is not None:
                logger.info(f"Loaded cached data for {pair} {timeframe}")
                return data
            
            # Fetch from exchange if not in storage
            data = self._fetch_historical_data(pair, timeframe)
            if data is None:
                return None
            
            # Validate timestamps
            is_valid, msg = self.validator.validate_timestamps(data, timeframe)
            if not is_valid:
                logger.error(f"Timestamp validation failed for {pair} {timeframe}: {msg}")
                return None
            
            # Save valid data
            self.data_storage.save_data(data, pair, timeframe)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting raw data for {pair} {timeframe}: {str(e)}")
            return None
            
    def process_all_pairs(self, pairs: List[str] = None, timeframes: List[str] = None) -> None:
        """Process all pairs with parallel execution and memory management"""
        pairs = pairs or self.trading_pairs
        timeframes = timeframes or self.timeframes
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                with Progress(
                    SpinnerColumn(),
                    *Progress.get_default_columns(),
                    TimeElapsedColumn(),
                    console=self.console
                ) as progress:
                    
                    total_tasks = len(pairs) * len(timeframes)
                    task = progress.add_task("[cyan]Processing pairs...", total=total_tasks)
                    
                    # Process pairs in parallel
                    futures = []
                    for pair in pairs:
                        for timeframe in timeframes:
                            future = executor.submit(self.process_pair, pair, timeframe)
                            futures.append((future, pair, timeframe))
                    
                    # Monitor progress and handle results
                    for future, pair, timeframe in futures:
                        try:
                            result = future.result(timeout=DOWNLOAD_TIMEOUT)
                            if result is not None:
                                logger.info(f"Successfully processed {pair} {timeframe}")
                            else:
                                logger.error(f"Failed to process {pair} {timeframe}")
                                
                        except Exception as e:
                            logger.error(f"Error processing {pair} {timeframe}: {str(e)}")
                            
                        finally:
                            progress.update(task, advance=1)
                            
                            # Clear memory after each pair
                            gc.collect()
                            
        except Exception as e:
            logger.error(f"Error in process_all_pairs: {str(e)}")
