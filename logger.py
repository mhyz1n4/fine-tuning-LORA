"""
Custom logger configuration for the project.
"""
import logging
import sys

class Logger:
    def __init__(self, name="ProjectLogger", level=logging.INFO, log_file=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent adding multiple handlers if the logger is retrieved again
        if not self.logger.handlers:
            # Stream Handler for terminal output
            stream_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            # File Handler for file output
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def log_block(self, messages, level="info"):
        """
        Formats multiple lines (passed as a list of strings) into a single log entry.
        """
        if isinstance(messages, str):
            msg = messages
        else:
            msg = "\n".join(str(m) for m in messages)
        
        if level.lower() == "info":
            self.info(msg)
        elif level.lower() == "warning":
            self.warning(msg)
        elif level.lower() == "error":
            self.error(msg)

# Global instance for easy import if needed, or users can instantiate their own.
project_logger = Logger()