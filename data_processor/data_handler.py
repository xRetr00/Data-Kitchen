"""
Data Handler Module

This module is responsible for preparing and validating data for machine learning models.
It ensures data quality, handles different timeframes, and provides consistent data shapes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Generator, Union
from datetime import datetime, timedelta
import logging
from .logger import setup_logger
from .config import TECHNICAL_INDICATORS, TRADING_PAIRS, TIMEFRAMES, PARQUET_DIR
from .processor import DataProcessor
from .data_storage import DataStorage
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, Console
from .memory_utils import log_memory_usage, optimize_dataframe, check_memory_threshold , get_memory_usage
import gc
import os
import tensorflow as tf

logger = setup_logger(__name__)

class DataHandler:
    """
    Handles data preparation and validation for machine learning models.
    Ensures data quality and consistency across different timeframes.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize DataHandler with default settings"""
        # Initialize components
        self.data_processor = DataProcessor()
        # Use PARQUET_DIR from config if no storage_dir provided
        self.data_storage = DataStorage(storage_dir or PARQUET_DIR)
        
        # Validation settings (more flexible)
        self.min_sequence_length = 50    # Reduced from 100
        self.max_missing_ratio = 0.35    # Increased from 0.1
        self.min_variance_threshold = 1e-8  # Reduced from 1e-6
        
        # Track pairs and timeframes
        self.num_pairs = len(TRADING_PAIRS)
        self.num_timeframes = len(TIMEFRAMES)
        self.pairs = TRADING_PAIRS
        self.timeframes = TIMEFRAMES
        
        logger.info(f"Data handler initialized. Storage path: {self.data_storage.storage_dir}")
        
    @log_memory_usage
    def get_training_data(self, pairs: List[str] = None, timeframes: List[str] = None) -> Tuple[Dict, Dict, Dict]:
        """
        Main entry point for getting processed and validated data for training
        
        Args:
            pairs: List of trading pairs to process (default: from config)
            timeframes: List of timeframes to process (default: from config)
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        pairs = pairs or TRADING_PAIRS
        timeframes = timeframes or TIMEFRAMES
        
        logger.info("=== Starting Data Preparation ===")
        logger.info(f"Pairs: {pairs}")
        logger.info(f"Timeframes: {timeframes}")
        
        # Step 1: Download all data first
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=Console()
        ) as progress:
            download_task = progress.add_task(
                "[cyan]Downloading data...",
                total=len(pairs) * len(timeframes)
            )
            
            downloaded_data = {}
            for pair in pairs:
                for timeframe in timeframes:
                    # Try to load from storage first
                    data = self.data_storage.load_data(pair, timeframe)
                    if data is None:
                        logger.info(f"Downloading new data for {pair} {timeframe}")
                        # If not in storage, download and save
                        data = self.data_processor._fetch_historical_data(pair, timeframe)
                        if data is not None:
                            self.data_storage.save_data(data, pair, timeframe)
                            downloaded_data[(pair, timeframe)] = data
                        else:
                            logger.error(f"Failed to download data for {pair} {timeframe}")
                    else:
                        downloaded_data[(pair, timeframe)] = data
                        
                    progress.update(download_task, advance=1)
            
            if not downloaded_data:
                logger.error("No data was downloaded!")
                raise ValueError("No data was downloaded")
        
        # Step 2: Process downloaded data
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=Console()
        ) as progress:
            process_task = progress.add_task(
                "[cyan]Processing data...",
                total=len(downloaded_data)
            )
            
            raw_data = {}
            for (pair, timeframe), data in downloaded_data.items():
                try:
                    if check_memory_threshold():
                        logger.warning("Memory threshold exceeded, cleaning up...")
                        gc.collect()
                    
                    processed_data = self.data_processor._process_chunk(data, pair, timeframe)
                    if processed_data is not None and not processed_data.empty:
                        raw_data[(pair, timeframe)] = processed_data
                        logger.info(f"Processed {len(processed_data)} points for {pair} {timeframe}")
                    else:
                        logger.warning(f"Processing failed for {pair} {timeframe}")
                        
                except Exception as e:
                    logger.error(f"Error processing {pair} {timeframe}: {str(e)}")
                    
                finally:
                    progress.update(process_task, advance=1)
            
            if not raw_data:
                logger.error("No data was processed!")
                raise ValueError("No data was processed")
                
            logger.info(f"Successfully processed {len(raw_data)} pair/timeframe combinations")
            
            # Step 3: Split into train/val/test
            train_data, val_data, test_data = self._prepare_datasets(raw_data)
            
            logger.info("=== Data Statistics ===")
            if train_data is not None:
                logger.info(f"Training sequences: {len(train_data['sequences'])}")
                logger.info(f"Validation sequences: {len(val_data['sequences'])}")
                logger.info(f"Test sequences: {len(test_data['sequences'])}")
            else:
                logger.error("No sequences found in prepared data")
                raise ValueError("No sequences found in prepared data")
            
            # Step 4: Validate and prepare final data
            logger.info("=== Data Validation ===")
            train_dataset, train_validation = self.validate_and_prepare_data(train_data)
            val_dataset, val_validation = self.validate_and_prepare_data(val_data)
            test_dataset, test_validation = self.validate_and_prepare_data(test_data)
            
            # Log validation results
            self._log_validation_results(train_validation, val_validation, test_validation)
            
            if train_dataset is None or val_dataset is None or test_dataset is None:
                logger.error("Data validation failed!")
                raise ValueError("Data validation failed. Cannot proceed with training.")
                
            logger.info("=== Data preparation completed successfully ===")
            return train_data, val_data, test_data
        
    @log_memory_usage
    def _collect_raw_data(self, pairs: List[str], timeframes: List[str]) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Collect and process raw data for all pairs and timeframes with memory optimization"""
        raw_data = {}
        
        try:
            for pair in pairs:
                for timeframe in timeframes:
                    logger.info(f"Processing {pair} {timeframe}...")
                    initial_memory = get_memory_usage()
                    
                    # Clear memory if threshold exceeded
                    if check_memory_threshold():
                        gc.collect()
                        logger.warning("Memory threshold reached - garbage collection triggered")
                    
                    # First, try to load cached data
                    try:
                        data = self.data_storage.load_data(pair, timeframe)
                        
                        if data is not None:
                            # Optimize dataframe memory usage
                            data = optimize_dataframe(data)
                            logger.info(f"Loaded and optimized cached data for {pair} {timeframe}")
                            raw_data[(pair, timeframe)] = data
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error loading cached data for {pair} {timeframe}: {str(e)}")
                        continue
                    
                    # If data not cached, process it
                    try:
                        data = self.data_processor.process_pair(pair, timeframe)
                        if data is not None:
                            # Optimize before saving
                            data = optimize_dataframe(data)
                            
                            # Save processed data
                            self.data_storage.save_data(data, pair, timeframe)
                            raw_data[(pair, timeframe)] = data
                            logger.info(f"Processed and saved data for {pair} {timeframe}")
                        else:
                            logger.warning(f"No data available for {pair} {timeframe}")
                            
                    except Exception as e:
                        logger.error(f"Error processing data for {pair} {timeframe}: {str(e)}")
                        continue
                    
                    # Log memory usage after processing each pair/timeframe
                    final_memory = get_memory_usage()
                    memory_diff = final_memory - initial_memory
                    logger.info(f"Memory change for {pair} {timeframe}: {memory_diff:.2f} MB")
                    
                    # Clear pair data if no longer needed
                    if pair in raw_data and all(tf in raw_data[pair] for tf in timeframes):
                        del data
                        gc.collect()
            
            return raw_data
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            return raw_data
            
    def _optimize_raw_data(self, raw_data: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Optimize memory usage of raw data dictionary"""
        try:
            optimized_data = {}
            for (pair, timeframe), data in raw_data.items():
                # Optimize each dataframe
                optimized_df = optimize_dataframe(data)
                optimized_data[(pair, timeframe)] = optimized_df
                
                # Clear original data
                del data
                
            # Force garbage collection
            gc.collect()
            return optimized_data
            
        except Exception as e:
            logger.error(f"Error optimizing raw data: {str(e)}")
        return raw_data
        
    def sequence_generator(self, data: pd.DataFrame, sequence_length: int, batch_size: int = 32):
        """Generator function for creating sequences efficiently"""
        features = data.values
        num_sequences = len(features) - sequence_length
        
        def gen_sequences():
            for i in range(0, num_sequences, batch_size):
                end_idx = min(i + batch_size, num_sequences)
                batch_sequences = []
                batch_targets = []
                
                for j in range(i, end_idx):
                    seq = features[j:j + sequence_length]
                    target = features[j + sequence_length, 3]  # Close price as target
                    batch_sequences.append(seq)
                    batch_targets.append(target)
                    
                yield np.array(batch_sequences, dtype=np.float32), np.array(batch_targets, dtype=np.float32)

        return gen_sequences

    def create_tf_dataset(self, data: pd.DataFrame, sequence_length: int, batch_size: int = 32) -> Optional[Dict]:
        """Create a dictionary containing sequences and targets"""
        try:
            if data is None or len(data) < sequence_length:
                logger.warning(f"Insufficient data points ({len(data) if data is not None else 0}) for sequence length {sequence_length}")
                return None
            
            # Convert to numpy array for faster processing
            features = data.values.astype(np.float32)
            
            # Create sequences and targets
            sequences = []
            targets = []
            
            for i in range(len(features) - sequence_length):
                seq = features[i:(i + sequence_length)]
                target = features[i + sequence_length, 3]  # Close price as target
                sequences.append(seq)
                targets.append(target)
            
            if not sequences:
                logger.warning("No sequences created")
                return None
            
            # Convert to numpy arrays and create dictionary
            return {
                'sequences': np.array(sequences, dtype=np.float32),
                'targets': np.array(targets, dtype=np.float32)
            }
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            return None

    def _prepare_datasets(self, raw_data: Dict[Tuple[str, str], pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
        """Prepare train/val/test datasets as dictionaries"""
        try:
            train_sequences = []
            train_targets = []
            val_sequences = []
            val_targets = []
            test_sequences = []
            test_targets = []
            
            for (pair, timeframe), data in raw_data.items():
                # Split data
                train, val, test = self._split_data(data)
                
                # Create datasets for each split
                train_ds = self.create_tf_dataset(train, self.min_sequence_length)
                val_ds = self.create_tf_dataset(val, self.min_sequence_length)
                test_ds = self.create_tf_dataset(test, self.min_sequence_length)
                
                if train_ds and val_ds and test_ds:
                    train_sequences.append(train_ds['sequences'])
                    train_targets.append(train_ds['targets'])
                    val_sequences.append(val_ds['sequences'])
                    val_targets.append(val_ds['targets'])
                    test_sequences.append(test_ds['sequences'])
                    test_targets.append(test_ds['targets'])
            
            if not train_sequences:
                logger.error("No valid datasets created")
                return None, None, None
            
            # Combine all sequences and targets
            train_data = {
                'sequences': np.concatenate(train_sequences, axis=0),
                'targets': np.concatenate(train_targets, axis=0)
            }
            
            val_data = {
                'sequences': np.concatenate(val_sequences, axis=0),
                'targets': np.concatenate(val_targets, axis=0)
            }
            
            test_data = {
                'sequences': np.concatenate(test_sequences, axis=0),
                'targets': np.concatenate(test_targets, axis=0)
            }
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error preparing datasets: {str(e)}")
            return None, None, None
        
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets"""
        train_size = int(len(data) * 0.8)
        val_size = int(len(data) * 0.1)
        
        train = data[:train_size]
        val = data[train_size:train_size + val_size]
        test = data[train_size + val_size:]
        
        return train, val, test
        
    @log_memory_usage
    def _add_to_dataset(self, dataset: Dict, data: pd.DataFrame, pair_id: int, timeframe_id: int):
        """Add processed data to dataset dictionary"""
        try:
            # Validate input data
            if data is None or data.empty:
                logger.error(f"Empty data for pair {pair_id}, timeframe {timeframe_id}")
                return
                
            # Extract features and targets
            if 'close' not in data.columns:
                logger.error(f"No 'close' column in data for pair {pair_id}, timeframe {timeframe_id}")
                return
                
            features = data.values
            targets = data['close'].values  # Store all close prices
            
            # Log data info
            logger.info(f"Processing data for pair {pair_id}, timeframe {timeframe_id}")
            logger.info(f"Features shape: {features.shape}, columns: {data.columns.tolist()}")
            
            # Check if we have enough data
            sequence_length = self.min_sequence_length
            num_sequences = len(features) - sequence_length - 1  # -1 because we need one more point for target
            
            if num_sequences <= 0:
                logger.warning(f"Not enough data points for pair {pair_id}, timeframe {timeframe_id}")
                return
                
            logger.info(f"Creating {num_sequences} sequences of length {sequence_length}")
            
            # Check memory before processing
            if check_memory_threshold():
                logger.error("Memory threshold exceeded before processing")
                return
                
            # Pre-allocate arrays if dataset is empty
            if 'sequences' not in dataset:
                dataset['sequences'] = []
                dataset['targets'] = []
                dataset['pair_idx'] = []
                dataset['timeframe_idx'] = []
                dataset['weights'] = []
            
            # Create sequences using sliding window
            try:
                sequences = np.array([features[i:i + sequence_length] for i in range(num_sequences)], dtype=np.float32)
                sequence_targets = targets[sequence_length + 1:len(targets)]  # Use next price after sequence as target
                
                # Reshape targets to (batch, target_features)
                sequence_targets = np.array(sequence_targets, dtype=np.float32).reshape(-1, 1)
                
                pair_indices = np.full(num_sequences, pair_id, dtype=np.int32)
                timeframe_indices = np.full(num_sequences, timeframe_id, dtype=np.int32)
                sequence_weights = np.ones(num_sequences, dtype=np.float32)
                
                # Log shapes
                logger.info(f"Created arrays with shapes:")
                logger.info(f"- sequences: {sequences.shape}")
                logger.info(f"- targets: {sequence_targets.shape}")
                logger.info(f"- pair_idx: {pair_indices.shape}")
                logger.info(f"- timeframe_idx: {timeframe_indices.shape}")
                
            except Exception as e:
                logger.error(f"Error creating sequences: {str(e)}")
                return
            
            # Verify lengths before adding
            if not (len(sequences) == len(sequence_targets) == len(pair_indices) == len(timeframe_indices) == len(sequence_weights)):
                logger.error(f"Length mismatch before adding to dataset: sequences={len(sequences)}, targets={len(sequence_targets)}")
                return
                
            # Add to dataset
            try:
                initial_len = len(dataset['sequences'])
                
                dataset['sequences'].extend(sequences)
                dataset['targets'].extend(sequence_targets)
                dataset['pair_idx'].extend(pair_indices)
                dataset['timeframe_idx'].extend(timeframe_indices)
                dataset['weights'].extend(sequence_weights)
                
                final_len = len(dataset['sequences'])
                added = final_len - initial_len
                
                logger.info(f"Successfully added {added} sequences to dataset")
                logger.info(f"Dataset now contains {final_len} total sequences")
                
            except Exception as e:
                logger.error(f"Error adding sequences to dataset: {str(e)}")
                return
            
            # Check memory after processing
            if check_memory_threshold():
                logger.warning("Memory threshold exceeded after processing")
            
            logger.debug(f"Added {num_sequences} sequences for pair {pair_id}, timeframe {timeframe_id}")
            logger.debug(f"Sequence shape: {sequences.shape}, target shape: {sequence_targets.shape}")
            
        except Exception as e:
            logger.error(f"Error adding data to dataset: {str(e)}")
        
    def prepare_sequences(self, data_dict: dict, sequence_length: int = 50) -> Optional[Dict]:
        """Prepare sequences for training from raw data"""
        try:
            if not data_dict or 'sequences' not in data_dict:
                logger.error("Invalid data dictionary format")
                return None
                
            sequences = np.array(data_dict['sequences'])
            targets = np.array(data_dict['targets'])
            
            if len(sequences) == 0 or len(targets) == 0:
                logger.error("Empty sequences or targets")
                return None
                
            # Validate dimensions
            if len(sequences.shape) != 3:
                logger.error(f"Invalid sequence shape: {sequences.shape}")
                return None
                
            # Prepare final data
            prepared_data = {
                'sequences': sequences,
                'targets': targets
            }
            
            if 'pair_indices' in data_dict:
                prepared_data['pair_indices'] = np.array(data_dict['pair_indices'])
            if 'timeframe_indices' in data_dict:
                prepared_data['timeframe_indices'] = np.array(data_dict['timeframe_indices'])
                
            logger.info("Prepared data shapes:")
            for key, value in prepared_data.items():
                logger.info(f"- {key}: {value.shape}")
                
            return prepared_data
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            return None

    def _validate_data_structure(self, data_dict: Dict) -> Tuple[bool, str]:
        """
        التحقق من صحة هيكل البيانات المدخلة.
        
        Args:
            data_dict: قاموس يحتوي على البيانات المراد التحقق منها
            
        Returns:
            Tuple[bool, str]: زوج يحتوي على نتيجة التحقق (صح/خطأ) ورسالة توضيحية
        """
        try:
            if not isinstance(data_dict, dict):
                return False, "Input must be a dictionary"

            # التحقق من وجود المفاتيح المطلوبة
            required_keys = {'sequences', 'targets'}
            optional_keys = {'pair_idx', 'timeframe_idx', 'weights'}
            
            # التحقق من المفاتيح المطلوبة
            if not all(key in data_dict for key in required_keys):
                missing_keys = required_keys - set(data_dict.keys())
                return False, f"Missing required keys: {missing_keys}"

            # التحقق من أن البيانات ليست فارغة
            if any(len(data_dict[key]) == 0 for key in required_keys):
                empty_keys = [key for key in required_keys if len(data_dict[key]) == 0]
                return False, f"Empty data arrays found for: {empty_keys}"

            # التحقق من تناسق أحجام المصفوفات للمفاتيح المطلوبة
            first_length = len(data_dict['sequences'])
            if not all(len(data_dict[key]) == first_length for key in required_keys):
                inconsistent_keys = [key for key in required_keys if len(data_dict[key]) != first_length]
                return False, f"Inconsistent array lengths for: {inconsistent_keys}"
            
            # التحقق من المفاتيح الاختيارية إذا وجدت
            for key in optional_keys:
                if key in data_dict and len(data_dict[key]) != first_length:
                    return False, f"Optional key {key} has inconsistent length"
            
            return True, "Data structure is valid"
            
        except Exception as e:
            return False, f"Error validating data structure: {str(e)}"


    def validate_and_prepare_data(self, data: Union[Dict, tf.data.Dataset], sequence_length: Optional[int] = None) -> Tuple[Optional[Union[Dict, tf.data.Dataset]], Dict]:
        """
        Validate and prepare data for training
        
        Args:
            data: Dictionary or tf.data.Dataset containing data
            sequence_length: Sequence length (optional)
            
        Returns:
            Tuple of (prepared data, validation results dict)
        """
        validation_results = {'valid': False, 'error': None}
        try:
            # التحقق من نوع البيانات
            if isinstance(data, tf.data.Dataset):
                # التحقق من صحة Dataset
                try:
                    # فحص عينة من البيانات
                    sample = next(iter(data))
                    if not isinstance(sample, tuple) or len(sample) != 2:
                        return None, {'valid': False, 'error': 'Dataset must yield (features, targets) tuples'}
                    
                    validation_results = {'valid': True, 'error': None}
                    return data, validation_results
                    
                except Exception as e:
                    return None, {'valid': False, 'error': f'Invalid dataset structure: {str(e)}'}
            
            # إذا كانت البيانات قاموساً
            elif isinstance(data, dict):
                # التحقق من صحة هيكل البيانات
                is_valid, message = self._validate_data_structure(data)
                validation_results['valid'] = is_valid
                validation_results['error'] = None if is_valid else message

                if not is_valid:
                    logger.error(f"Data validation failed: {message}")
                    return None, validation_results
                
                # باقي عمليات التحقق والإعداد
                if sequence_length:
                    prepared_data = self.prepare_sequences(data, sequence_length)
                else:
                    prepared_data = data

                return prepared_data, validation_results
            
            else:
                return None, {'valid': False, 'error': f'Unsupported data type: {type(data)}'}
            
        except Exception as e:
            logger.error(f"Error in validate_and_prepare_data: {str(e)}")
            validation_results['error'] = str(e)
            return None, validation_results
        
    def _log_validation_results(self, train_validation: Dict, val_validation: Dict, test_validation: Dict):
        """Log validation results for all datasets"""
        if not all(train_validation.values()):
            logger.warning(f"Training data validation issues: {train_validation}")
        if not all(val_validation.values()):
            logger.warning(f"Validation data validation issues: {val_validation}")
        if not all(test_validation.values()):
            logger.warning(f"Test data validation issues: {test_validation}")
            
    def validate_sequence_length(self, data: pd.DataFrame) -> bool:
        """
        Validate if sequence length is sufficient for training
        
        Args:
            data: Input DataFrame
            
        Returns:
            bool: True if length is sufficient
        """
        if len(data) < self.min_sequence_length:
            logger.warning(f"Sequence length {len(data)} is less than minimum required {self.min_sequence_length}")
            return False
        return True
        
    def validate_missing_values(self, data: pd.DataFrame) -> bool:
        """
        Check for missing values in data
        
        Args:
            data: Input DataFrame
            
        Returns:
            bool: True if missing values are within acceptable range
        """
        missing_ratio = data.isnull().sum().max() / len(data)
        if missing_ratio > self.max_missing_ratio:
            logger.warning(f"Missing value ratio {missing_ratio:.2%} exceeds threshold {self.max_missing_ratio:.2%}")
            return False
        return True
        
    def validate_variance(self, data: pd.DataFrame) -> bool:
        """
        Check if features have sufficient variance
        
        Args:
            data: Input DataFrame
            
        Returns:
            bool: True if variance is sufficient
        """
        # Remove timestamp column if exists
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Calculate variance for each column
        variances = numeric_data.var()
        low_variance = variances[variances < self.min_variance_threshold]
        
        if not low_variance.empty:
            logger.warning(f"Low variance features detected: {low_variance.index.tolist()}")
            return False
        return True
        
    def validate_data_dimensions(self, X: np.ndarray, y: np.ndarray, expected_shape: tuple) -> Tuple[bool, str]:
        """Validate data dimensions"""
        try:
            if len(X.shape) != len(expected_shape):
                return False, f"Input dimensions mismatch. Expected {len(expected_shape)} dimensions, got {len(X.shape)}"
                
            for i, (actual, expected) in enumerate(zip(X.shape[1:], expected_shape[1:])):
                if expected is not None and actual != expected:
                    return False, f"Dimension {i+1} mismatch. Expected {expected}, got {actual}"
                    
            if len(y.shape) != 2:
                return False, "Target must be 2-dimensional (samples, features)"
                
            if X.shape[0] != y.shape[0]:
                return False, f"Number of samples mismatch. X: {X.shape[0]}, y: {y.shape[0]}"
                
            return True, "Data dimensions validation passed"
            
        except Exception as e:
            return False, f"Error in dimension validation: {str(e)}"
            

    def load_data_generator(self, file_path: str, batch_size: int = 32) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Load data using generator for memory efficient data loading
        
        Args:
            file_path (str): Path to the parquet file
            batch_size (int): Size of each batch, default is 32
            
        Yields:
            Tuple[np.ndarray, np.ndarray]: Batches of (X, y) data
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If data validation fails
            Exception: For other data loading errors
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"Empty file: {file_path}")
            
            # Read data in chunks
            chunk_size = 10000
            for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
                try:
                    # Check memory usage
                    if check_memory_threshold():
                        logger.warning("Memory threshold exceeded, cleaning up...")
                        gc.collect()
                    
                    # Validate data
                    is_valid, msg = self.validate_data_quality(chunk)
                    if not is_valid:
                        logger.warning(f"Skipping invalid chunk: {msg}")
                        continue
                    
                    # Prepare data
                    X, y = self.prepare_data(chunk)
                    
                    # Validate dimensions
                    expected_shape = (None, 60, 5)
                    is_valid, msg = self.validate_data_dimensions(X, y, expected_shape)
                    
                    if not is_valid:
                        logger.warning(f"Skipping chunk due to dimension mismatch: {msg}")
                        continue
                    
                    # Yield data in batches
                    n_samples = len(X)
                    for i in range(0, n_samples, batch_size):
                        try:
                            batch_X = X[i:i + batch_size]
                            batch_y = y[i:i + batch_size]
                            
                            # Validate batch data
                            if len(batch_X) == batch_size and not np.isnan(batch_X).any() and not np.isnan(batch_y).any():
                                yield batch_X, batch_y
                            else:
                                logger.warning(f"Skipping incomplete or invalid batch at index {i}")
                                
                        except Exception as e:
                            logger.error(f"Error processing batch at index {i}: {str(e)}")
                            continue
                    
                    # Clean up memory
                    del X, y
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    continue
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data quality"""
        try:
            # Check for missing values
            if data.isnull().values.any():
                return False, "Missing values detected"
                
            # Check for non-numeric values
            if not np.issubdtype(data.dtype, np.number):
                return False, "Non-numeric values detected"
                
            return True, "Data quality validation passed"
            
        except Exception as e:
            return False, f"Error in data quality validation: {str(e)}"
            
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data"""
        try:
            # Convert data to arrays
            X = data.values
            y = data['close'].values
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def validate_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
        """Validate training data"""
        try:
            # Check for NaN values
            if np.isnan(X).any() or np.isnan(y).any():
                return False, "Training data contains NaN values"
                
            # Check for infinite values
            if np.isinf(X).any() or np.isinf(y).any():
                return False, "Training data contains infinite values"
                
            # Check data range
            if not np.all((X >= -1) & (X <= 1)):
                return False, "Input features not properly normalized (should be in [-1, 1])"
                
            if not np.all((y >= -1) & (y <= 1)):
                return False, "Target values not properly normalized (should be in [-1, 1])"
                
            # Check sufficient data for training
            min_samples = 1000
            if len(X) < min_samples:
                return False, f"Insufficient training samples. Got {len(X)}, need at least {min_samples}"
                
            # Check data distribution
            if np.std(X) < 1e-6:
                return False, "Input features have near-zero variance"
                
            if np.std(y) < 1e-6:
                return False, "Target values have near-zero variance"
                
            return True, "Training data validation passed"
            
        except Exception as e:
            return False, f"Error in training data validation: {str(e)}"

    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and validate training data"""
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            # Validate dimensions
            expected_shape = (None, 60, 5)
            is_valid, msg = self.validate_data_dimensions(X, y, expected_shape)
            if not is_valid:
                raise ValueError(f"Data dimensions validation failed: {msg}")
                
            # Validate data
            is_valid, msg = self.validate_training_data(X, y)
            if not is_valid:
                raise ValueError(f"Training data validation failed: {msg}")
                
            # Split data
            train_size = 0.8
            split_idx = int(len(X) * train_size)
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Final validation
            for name, data in [('Training', X_train), ('Validation', X_val)]:
                if len(data) < 100:
                    raise ValueError(f"{name} set too small: {len(data)} samples")
                    
            return (X_train, y_train), (X_val, y_val)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

    def check_nan_values(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check for missing values in the data
        
        Args:
            data: DataFrame to check
            
        Returns:
            Tuple[bool, str]: Validation result and error message
        """
        try:
            # Calculate missing value ratio for each column
            nan_ratios = data.isnull().mean()
            
            # Check essential columns
            essential_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in essential_columns:
                if col in data.columns and nan_ratios[col] > 0.1:  # More than 10% missing values
                    return False, f"High ratio of missing values in column {col}: {nan_ratios[col]:.2%}"
            
            # Check technical indicators
            indicator_columns = [col for col in data.columns if col not in essential_columns]
            for col in indicator_columns:
                if nan_ratios[col] > 0.2:  # More than 20% missing values
                    logger.warning(f"High ratio of missing values in indicator {col}: {nan_ratios[col]:.2%}")
            
            return True, "Missing values check passed successfully"
            
        except Exception as e:
            return False, f"Error checking missing values: {str(e)}"

    def _process_raw_data(self, raw_data: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Process raw data for all pairs and timeframes"""
        processed_data = {}
        
        try:
            for (pair, timeframe), data in raw_data.items():
                try:
                    # Process data using processor
                    processed = self.processor._process_chunk(data, pair, timeframe)
                    
                    if processed is not None:
                        processed_data[(pair, timeframe)] = processed
                        logger.info(f"Successfully processed {pair} {timeframe}")
                    else:
                        logger.error(f"Error processing {pair} {timeframe}")
                        
                except Exception as e:
                    logger.error(f"Error processing {pair} {timeframe}: {str(e)}")
                    
            if not processed_data:
                logger.error("No data was processed!")
                return None
                
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in _process_raw_data: {str(e)}")
            return None