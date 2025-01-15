"""
Data Processor Logger Module
"""

import logging
import os
from .config import LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_FILE

def setup_logger(name: str) -> logging.Logger:
    """
    Set up the logger with the specified configuration
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger object
    """
    # Create the log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Set up the file handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Set up the console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Add the handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
