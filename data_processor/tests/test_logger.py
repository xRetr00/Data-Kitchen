"""
Test Logger Module
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_test_logger():
    """
    Set up the logging system for tests
    """
    # Create the log directory if it doesn't exist
    log_dir = Path("logs/tests")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the log file name with the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_results_{timestamp}.log"
    
    # Set up the logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Print to the console as well
        ]
    )
    
    return logging.getLogger("TestLogger")
