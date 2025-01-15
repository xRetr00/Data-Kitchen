"""
Data Processor Configurations
"""

import os
from datetime import datetime, timedelta

# API configurations
BINANCE_API_KEY = ""  # Enter your API key here
BINANCE_SECRET_KEY = ""  # Enter your secret key here

# Trading pairs and time frames
TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']  # Main trading pairs
TIMEFRAMES = ['1h', '4h', '1d']  # Most commonly used time frames

# Historical data period configurations
YEARS_OF_DATA = 3  # Number of years required
START_DATE = datetime.now() - timedelta(days=YEARS_OF_DATA * 365)
END_DATE = datetime.now()

# Calculate the number of candles required for each time frame
TIMEFRAME_CHUNKS = {
    '1h': int((END_DATE - START_DATE).total_seconds() / 3600),  # Number of hours in 3 years
    '4h': int((END_DATE - START_DATE).total_seconds() / (3600 * 4)),  # Number of 4 hour periods
    '1d': int((END_DATE - START_DATE).days)  # Number of days
}

# Data processing configurations
CHUNK_SIZE = max(TIMEFRAME_CHUNKS.values())  # Set chunk size to accommodate the largest time frame
MAX_WORKERS = 4    # Number of concurrent operations

# File paths
DATA_DIR = "data"
LOG_DIR = "logs"
PARQUET_DIR = "data/parquet"

# Technical indicators configurations
TECHNICAL_INDICATORS = {
    'RSI': {'period': 14},  # Relative Strength Index
    'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},  # Moving Average Convergence Divergence
    'SMA': {'periods': [20, 50, 200]},  # Simple Moving Average
    'EMA': {'periods': [9, 21, 50]},  # Exponential Moving Average
}

# Logging configurations
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/data_processor.log"

# Check if paths exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(PARQUET_DIR):
    os.makedirs(PARQUET_DIR)

# Print time period information
print(f"""
Time period information:
--------------------------------
Start date: {START_DATE}
End date: {END_DATE}
Number of candles required for each time frame:
- 1h: {TIMEFRAME_CHUNKS['1h']} candles
- 4h: {TIMEFRAME_CHUNKS['4h']} candles
- 1d: {TIMEFRAME_CHUNKS['1d']} candles
""")