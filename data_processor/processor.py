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
        """Process data for a single pair and timeframe"""
        try:
            # First check if data exists in storage
            data = self.data_storage.load_data(pair, timeframe)
            if data is None:
                logger.info(f"No cached data found for {pair} {timeframe}, fetching from exchange...")
                # Fetch historical data first
                with Progress(
                    SpinnerColumn(),
                    *Progress.get_default_columns(),
                    TimeElapsedColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task(
                        f"[cyan]Downloading {pair} {timeframe}...",
                        total=100
                    )
                    data = self._fetch_historical_data(pair, timeframe)
                    progress.update(task, completed=100)
                
                if data is None:
                    logger.error(f"Failed to fetch data for {pair} {timeframe}")
                    return None
                
                # Save raw data
                self.data_storage.save_data(data, pair, timeframe)
            
            # Now process the data
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Processing {pair} {timeframe}...",
                    total=100
                )
                
                # Process in chunks to avoid memory issues
                chunk_size = 1000
                processed_chunks = []
                
                for i in range(0, len(data), chunk_size):
                    if check_memory_threshold():
                        logger.warning("Memory threshold exceeded, cleaning up...")
                        gc.collect()
                    
                    chunk = data.iloc[i:i + chunk_size].copy()
                    processed_chunk = self._process_data(chunk)
                    processed_chunks.append(processed_chunk)
                    
                    progress.update(task, completed=(i + chunk_size) * 100 / len(data))
                
                # Combine processed chunks
                processed_data = pd.concat(processed_chunks, axis=0)
                processed_data = optimize_dataframe(processed_data)
                
                progress.update(task, completed=100)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing {pair} {timeframe}: {str(e)}")
            return None
            
    def _get_raw_data(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get raw data from parquet file"""
        try:
            # Construct file path
            file_path = Path(PARQUET_DIR) / f"{pair.replace('/', '_')}_{timeframe}.parquet"
            
            # Check if file exists
            if not file_path.exists():
                logger.error(f"Raw data file not found: {file_path}")
                return None
                
            # Load data
            data = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(data)} raw data points from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            return None
            
    def _process_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process raw data"""
        try:
            if data is None or data.empty:
                return None
                
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                # Make copy to avoid modifying original
                df = data.copy()
                total_steps = 5  # Total number of processing steps
                task = progress.add_task("[cyan]Processing data...", total=total_steps)
                
                # Convert timestamp to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                progress.update(task, advance=1, description="[cyan]Converting timestamps...")
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Sort by time
                df = df.sort_index()
                progress.update(task, advance=1, description="[cyan]Sorting data...")
                
                # Convert OHLCV columns to float32
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(np.float32)
                progress.update(task, advance=1, description="[cyan]Converting data types...")
                
                # Add technical indicators
                df = calculate_technical_indicators(df, self.indicators_config)
                progress.update(task, advance=1, description="[cyan]Calculating indicators...")
                
                # Handle missing data
                df = handle_missing_data(df)
                
                # Drop any remaining NaN values
                df = df.dropna()
                progress.update(task, advance=1, description="[cyan]Cleaning data...")
                
                logger.info(f"Processed data shape: {df.shape}")
                return df
                
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return None
            
    def _add_indicator(self, df: pd.DataFrame, indicator: str) -> None:
        """Add a technical indicator to the data"""
        # Implement indicator calculation here
        pass
    
    def process_all_pairs(self, pairs: List[str] = None, timeframes: List[str] = None) -> None:
        """Process all trading pairs and time frames"""
        pairs = pairs or self.trading_pairs
        timeframes = timeframes or self.timeframes
        
        # Create single progress display for all operations
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            # Calculate total tasks
            total_tasks = len(pairs) * len(timeframes)
            main_task = progress.add_task(
                "[cyan]Processing all pairs...",
                total=total_tasks
            )
            
            processed_count = 0
            failed_pairs = []
            
            for pair in pairs:
                for timeframe in timeframes:
                    try:
                        # Update task description
                        progress.update(
                            main_task,
                            description=f"[cyan]Processing {pair} {timeframe}..."
                        )
                        
                        # Try to load from storage first
                        data = self.data_storage.load_data(pair, timeframe)
                        
                        if data is None:
                            # If not in storage, fetch from exchange
                            data = self._fetch_historical_data(pair, timeframe)
                            
                            if data is not None:
                                # Save raw data immediately
                                self.data_storage.save_data(data, pair, timeframe)
                        
                        if data is not None:
                            # Process in chunks
                            chunk_size = 1000
                            processed_chunks = []
                            
                            for i in range(0, len(data), chunk_size):
                                if check_memory_threshold():
                                    logger.warning("Memory threshold exceeded, cleaning up...")
                                    gc.collect()
                                
                                chunk = data.iloc[i:i + chunk_size].copy()
                                processed_chunk = self._process_data(chunk)
                                if processed_chunk is not None:
                                    processed_chunks.append(processed_chunk)
                            
                            if processed_chunks:
                                processed_data = pd.concat(processed_chunks, axis=0)
                                processed_data = optimize_dataframe(processed_data)
                                self.data_storage.save_processed_data(processed_data, pair, timeframe)
                                processed_count += 1
                            else:
                                failed_pairs.append((pair, timeframe, "Processing failed"))
                        else:
                            failed_pairs.append((pair, timeframe, "Data fetch failed"))
                            
                    except Exception as e:
                        logger.error(f"Error processing {pair} {timeframe}: {str(e)}")
                        failed_pairs.append((pair, timeframe, str(e)))
                    
                    finally:
                        # Update progress
                        progress.update(main_task, advance=1)
            
            # Final progress update
            progress.update(
                main_task,
                description=f"[green]Completed! Processed {processed_count}/{total_tasks} pairs"
            )
        
        # Log summary
        if processed_count == 0:
            logger.error("No data was processed successfully!")
            return False
            
        if failed_pairs:
            logger.warning(f"Failed to process {len(failed_pairs)} pairs:")
            for pair, timeframe, reason in failed_pairs:
                logger.warning(f"- {pair} {timeframe}: {reason}")
        
        return processed_count > 0
