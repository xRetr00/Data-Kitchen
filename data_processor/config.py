"""
Data Processor Configurations
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# API configurations
EXCHANGES = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', ''),  # Add your Binance API key here
        'secret_key': os.getenv('BINANCE_SECRET_KEY', ''),  # Add your Binance secret key here
        'enabled': os.getenv('BINANCE_ENABLED', 'False').lower() == 'true',  # Set to False to disable this exchange
        'priority': int(os.getenv('BINANCE_PRIORITY', '1')),  # Lower number means higher priority
    },
    'binanceus': {
        'api_key': os.getenv('BINANCEUS_API_KEY', ''),  # Add your Binance US API key here
        'secret_key': os.getenv('BINANCEUS_SECRET_KEY', ''),  # Add your Binance US secret key here
        'enabled': os.getenv('BINANCEUS_ENABLED', 'False').lower() == 'true',
        'priority': int(os.getenv('BINANCEUS_PRIORITY', '2')),
    },
    'bybit': {
        'api_key': os.getenv('BYBIT_API_KEY', ''),  # Add your Bybit API key here
        'secret_key': os.getenv('BYBIT_SECRET_KEY', ''),  # Add your Bybit secret key here
        'enabled': os.getenv('BYBIT_ENABLED', 'False').lower() == 'true',
        'priority': int(os.getenv('BYBIT_PRIORITY', '3')),
    },
    'okx': {
        'api_key': os.getenv('OKX_API_KEY', ''),  # Replace with your OKX API key
        'secret_key': os.getenv('OKX_SECRET_KEY', ''),  # Replace with your OKX secret key
        'passphrase': os.getenv('OKX_PASSPHRASE', ''),  # Replace with your OKX passphrase
        'enabled': os.getenv('OKX_ENABLED', 'True').lower() == 'true',
        'priority': int(os.getenv('OKX_PRIORITY', '4')),
    }
}

# Exchange retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Download settings
DOWNLOAD_TIMEOUT = 300  # 5 minutes max for entire operation
CHUNK_TIMEOUT = 60     # 60 seconds timeout per chunk
MAX_CHUNK_SIZE = {
    'default': 1000,
    'BNB/USDT': 500,   # Special handling for BNB pairs
    'BTC/USDT': 1000,
    'ETH/USDT': 1000,
    'ADA/USDT': 1000
}

# Memory settings
MEMORY_THRESHOLD = 85  # Percentage of memory usage that triggers cleanup
CHUNK_SIZE = 1000     # Size of chunks for processing

# Trading pairs and time frames
TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']  # Already in correct format
TIMEFRAMES = ['1h', '4h', '1d']  # Only these specific timeframes

# Historical data period configurations
YEARS_OF_DATA = 3  # Number of years required
START_DATE = datetime.now() - timedelta(days=YEARS_OF_DATA * 365)
END_DATE = datetime.now()

# Processing configurations
MAX_WORKERS = 4    # Number of concurrent operations

# Data paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
PARQUET_DIR = ROOT_DIR / 'data' / 'parquet'
CACHE_DIR = ROOT_DIR / 'cache'
MODEL_DIR = ROOT_DIR / 'models'
LOG_DIR = ROOT_DIR / 'logs'

# Create directories if they don't exist
for directory in [PARQUET_DIR, CACHE_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    'RSI': {
        'period': 14
    },
    'MACD': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    'MA': {
        'SMA': {'period': 20},
        'EMA': {'period': 20}
    }
}

# Logging configurations
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'logs/data_processor.log'
