"""
Logger module with colored output
"""

import logging
import colorlog
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from .config import LOG_DIR

def setup_logger(name: str) -> logging.Logger:
    """
    Setup colored logger with rich progress bar support
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    
    # Create console for rich output
    console = Console(file=sys.stderr)
    
    # Create root logger
    root_logger = logging.getLogger()
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='[%(filename)s:%(lineno)d] %(message)s',
        datefmt='%m/%d/%y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / 'data_processor.log'),
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_time=False,
                show_path=True,
                markup=True,
                enable_link_path=True,
                show_level=True,
                log_time_format='[%m/%d/%y %H:%M:%S]'
            )
        ]
    )
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file formatter with more detailed format
    file_formatter = logging.Formatter(
        '[%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%m/%d/%y %H:%M:%S'
    )
    
    # Update file handler formatter
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(file_formatter)
    
    return logger