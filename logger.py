"""
Custom logger configuration for the project.
"""
import logging
import sys

class Logger:
    def __init__(self, name="ProjectLogger", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent adding multiple handlers if the logger is retrieved again
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            # Simple format: Time - Level - Message
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

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