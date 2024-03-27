""" 
This script sets up the configuration of the root logger for the application.

The script defines a function called setup_root_logger() that configures the root logger of the application. 
It sets up the log format, console handler, rotating file handler, and logger level. The configured handlers 
are added to the logger and the logger level is set to INFO.

Usage:
    - Import the logging_setup module.
    - Call the setup_root_logger() function to configure the root logger.

Example:
    import logging_setup
    
    logging_setup.setup_root_logger()
"""

import logging
import os


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

def setup_root_logger():
    """ 
    Setup configuration of the root logger of the application.
    
    This function configures the root logger of the application by setting up the log format, 
    console handler, rotating file handler, and logger level. It adds the configured handlers 
    to the logger and sets the logger level to INFO.
    
    Parameters:
    None
    
    Returns:
    None
    """
    logger = logging.getLogger('')
    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    # check if the logs directory exists
    os.makedirs("src/fos/logs", exist_ok=True)
    file = logging.handlers.RotatingFileHandler(filename="src/fos/logs/fastapi-fos-logs.log", mode='a', maxBytes=15000000, backupCount=5)
    file.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file)
    logger.setLevel(logging.INFO)