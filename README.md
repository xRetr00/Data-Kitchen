# Data Processor (Data Kitchen)

A professional and efficient data processing system designed for collecting, cleaning, and analyzing trading data, optimized for machine learning model training and real-time trading with ML models.

## Features

- Collecting historical and live data from Binance
- Multi-threaded processing for pairs and time frames
- Calculating technical indicators (RSI, MACD, SMA)
- Handling missing values and normalizing data
- Efficient storage using Parquet format
- Comprehensive logging system

## Main Features

- Collecting historical and live data from Binance
- Multi-threaded processing for pairs and time frames
- Calculating technical indicators (RSI, MACD, SMA)
- Handling missing values and normalizing data
- Efficient storage using Parquet format
- Comprehensive logging system

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
ccxt>=2.0.0
#you need to download the ta-lib source to avoid build errors
ta-lib>=0.4.0
scikit-learn>=1.0.0
pytest>=7.0.0
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/xRetr00/data-kitchen.git
cd data-kitchen
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Setup configuration:
- Open `config.py`
- Add your API keys
- Modify the pairs and time frames as needed

## Usage

### Basic Processing

```python
from data_processor.processor import DataProcessor

# Create a new processor
processor = DataProcessor()

# Process all pairs
processor.process_all_pairs()

# Process a specific pair
processor.process_pair('BTC/USDT', '1h')
```

### Trading Live

```python
# Process all pairs in real-time
processor.process_all_pairs(is_live=True)
```

## Module Structure

### processor.py
The main processor:
- Collecting data from Binance
- Formatting and cleaning data
- Calculating technical indicators
- Multi-threaded processing

### data_utils.py
Utility functions for data processing:
- Indicator calculations (RSI, MACD, SMA)
- Data formatting
- Handling missing values

### data_storage.py
Data storage management:
- Saving data in Parquet format
- Loading previous data
- File management

### logger.py
Logging system:
- Logging operations
- Logging errors
- Performance tracking

### config.py
System configuration:
- API keys
- Pairs and time frames
- Indicator parameters

### memory_utils.py
Memory management:
- Memory usage tracking
- Memory release

### test units
Unit tests for the data processor:
- Test the main processor
- Test data utils
- Test data storage
- Test logger

## Best Practices

1. **Memory Management**
   - Use batch processing for large datasets
   - Release memory after processing
   - Use concurrent processing wisely

2. **Handling Missing Data**
   - Check the percentage of missing data
   - Use appropriate interpolation methods
   - Document decisions

3. **Indicator Calculations**
   - Avoid look-ahead bias
   - Validate calculations
   - Monitor performance

## License

MIT

Made by [xRetr00](https://github.com/xRetr00)


