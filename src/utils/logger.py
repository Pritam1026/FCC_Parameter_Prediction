import logging
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def get_logger(
        name:str,
        log_dir:str='logs',
        level:int=logging.INFO,
        log_to_file:bool=True
)->logging.Logger:
    '''
    Create and configure a logger object with console handler and optional File handler

    Args:
        name : str Logger name typically __name__ of calling module
        log_dir : str Directory to store log file
        level : int Logging level default INFO
        log_to_file : Whether to write logs to file or not

    Return:
        configured logger instance
        
    '''
    # Creating logger
    logger=logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated call
    if logger.handlers:
        return logger
    
    # Setting the default level
    logger.setLevel(level=level)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file handler
    if log_to_file :
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger