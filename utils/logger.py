'''
 # @ Author: Pallab Maji
 # @ Create Time: 2024-09-29 03:05:27
 # @ Modified time: 2024-09-29 03:05:30
 # @ Description: This is logger class to log the information, debug and error messages acros the repository.
    The `Logger` class is a simple class that creates a logger object and logs messages to a file and console. The `Logger` class has the following methods:
        - `info`: Logs an info message.
        - `debug`: Logs a debug message.
        - `error`: Logs an error message.
        - `warning`: Logs a warning message.
        - `critical`: Logs a critical message.
        - `exception`: Logs an exception message.
        - `log`: Logs a message with the specified log level.
 '''

import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, logdir, log_level=logging.DEBUG):
        self.logdir = logdir
        self.logger = logging.getLogger()
        self.log_level = log_level
        self.logger.setLevel(log_level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create the directory if it does not exist
        if not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)
        
        # Create the file handler
        self.file_handler = logging.FileHandler(os.path.join(logdir, 'logfile.log'))
        self.file_handler.setLevel(self.log_level)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        
        # Create the console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(self.log_level)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)

    def exception(self, message):
        self.logger.exception(message)

    def log(self, level, message):
        self.logger.log(level, message)




