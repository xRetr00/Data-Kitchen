# Data Kitchen 🍳

A professional and efficient data processing system designed for high-performance financial data processing, optimized for machine learning applications and real-time trading.

## Key Features 🌟

- **Real-time Data Processing**: Collect and process live data from Binance
- **Multi-Asset Support**: Handle multiple trading pairs simultaneously
- **Technical Analysis**: Calculate key indicators (RSI, MACD, SMA, EMA)
- **Data Validation**: Robust error checking and missing data handling
- **Efficient Storage**: Optimized Parquet format with chunked processing
- **Memory Management**: Efficient handling of large datasets
- **Comprehensive Logging**: Detailed operation tracking and error reporting
- **Improved Caching**: Smart caching with automatic cleaning
- **Parallel Processing**: CPU and I/O dependent task parallelization

## Project Structure 🏗️

```
/data_processor/
├──  __init__.py    # Package initialization
├──  processor.py   # Core processing logic
├──  logger.py      # Logging utilities
├──  config.py      # Configuration settings
├──  data_utils.py  # Data handling utilities
├── data_storage.py # Data storage management
├── sentiment_analyzer.py # Sentiment analysis
├── data_validator.py # Data validation
└──  /logs/         # Log files directory
```

## Requirements 📋

```
pandas>=1.5.0
numpy>=1.21.0
ccxt>=2.0.0
ta-lib>=0.4.0  # Requires ta-lib source installation
scikit-learn>=1.0.0
pytest>=7.0.0
pytest-cov>=4.0.0
textblob>=0.17.1
beautifulsoup4>=4.12.0
requests>=2.31.0
lxml>=4.9.0
```

## Installation 💻

1. Clone the repository:
```bash
git clone https://github.com/xRetr00/data-kitchen.git
cd data-kitchen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀

### Basic Data Processing

```python
from data_processor.processor import DataProcessor

# Initialize processor
processor = DataProcessor()

# Process specific pair
processor.process_pair(
    pair='BTC/USDT',
    timeframe='1h'
)

# Process all configured pairs
processor.process_all_pairs()
```

### Real-time Processing

```python
# Start real-time processing
processor.process_all_pairs(is_live=True)
```

### Sentiment Analysis

```python
from data_processor.sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Get sentiment features
sentiment_data = analyzer.get_sentiment_features('BTC/USDT', start_date, end_date)
```

### Data Storage

```python
from data_processor.data_storage import DataStorage

# Initialize storage
storage = DataStorage()

# Save processed data
storage.save_data(data, pair='BTC/USDT', timeframe='1h')

# Load historical data
data = storage.load_data(pair='BTC/USDT', timeframe='1h')
```

## Configuration

The system is configured through `config.py`:

```python
CONFIG = {
    'pairs': ['BTC/USDT', 'ETH/USDT'],
    'timeframes': ['1h', '4h', '1d'],
    'indicators': {
        'RSI': {'period': 14},
        'MACD': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        },
        'SMA': {'periods': [20, 50, 200]},
        'EMA': {'periods': [20, 50, 200]}
    },
    'processing': {
        'chunk_size': 1000,
        'missing_threshold': 0.1
    }
}
```

## Error Handling

The system includes comprehensive error handling:
- Data validation
- API connection errors
- Missing data management
- Memory overflow protection

## Logging

Detailed logging is available in the `/logs` directory:
- Processing operations
- Error tracking
- Performance metrics
- Data validation results

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [ccxt](https://github.com/ccxt/ccxt)
- [ta-lib](https://github.com/ta-lib/ta-lib)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [pytest](https://pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)
- [textblob](https://textblob.readthedocs.io/en/dev/)
- [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [requests](https://requests.readthedocs.io/en/master/)
- [lxml](https://lxml.de/)

## Author

- [xRetr00](https://github.com/xRetr00)