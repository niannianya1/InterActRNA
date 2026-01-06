# utils/logger_utils.py
import sys
import datetime
import re
import os
_ORIGINAL_STDOUT_LOGGER = sys.stdout

LOGGER_INSTANCE_FOR_UTILS = None  # Logger instance specific to utils


class Logger(object):
    def __init__(self, filename="logfile.log", terminal_stream=_ORIGINAL_STDOUT_LOGGER):
        self.terminal_stream = terminal_stream
        # Ensure log directory exists
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.log_file_handle = open(filename, "w", encoding='utf-8')
        self.ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        if self.terminal_stream:
            try:
                self.terminal_stream.write(message)
                self.terminal_stream.flush()
            except Exception:
                sys.__stdout__.write(message)
                sys.__stdout__.flush()
        if self.log_file_handle and not self.log_file_handle.closed:
            cleaned_message = self.ansi_escape_pattern.sub('', message)
            self.log_file_handle.write(cleaned_message)
            self.log_file_handle.flush()

    def flush(self):
        if self.terminal_stream:
            try:
                self.terminal_stream.flush()
            except Exception:
                sys.__stdout__.flush()
        if self.log_file_handle and not self.log_file_handle.closed:
            self.log_file_handle.flush()

    def close(self):
        if self.log_file_handle and not self.log_file_handle.closed:
            self.log_file_handle.close()
            self.log_file_handle = None

    def isatty(self):
        if self.terminal_stream and hasattr(self.terminal_stream, 'isatty'):
            return self.terminal_stream.isatty()
        return False


def set_utils_logger(logger_instance):
    """Sets the logger instance for all utils modules that use log_print."""
    global LOGGER_INSTANCE_FOR_UTILS
    LOGGER_INSTANCE_FOR_UTILS = logger_instance


# Global LOG_LEVEL_INFO_UTILS and LOG_LEVEL_DEBUG_UTILS for utils' log_print
# These should be set by main_train.py using a dedicated function if their values vary
LOG_LEVEL_INFO_UTILS = True  # Default
LOG_LEVEL_DEBUG_UTILS = False  # Default


def set_utils_log_levels(log_info, log_debug):
    global LOG_LEVEL_INFO_UTILS, LOG_LEVEL_DEBUG_UTILS
    LOG_LEVEL_INFO_UTILS = log_info
    LOG_LEVEL_DEBUG_UTILS = log_debug


def log_print(message, level="info"):
    global LOGGER_INSTANCE_FOR_UTILS
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    prefix_map = {"info": "[INFO]", "debug": "[DEBUG]", "warn": "[WARN]", "error": "[ERROR]"}
    prefix = prefix_map.get(level.lower(), "[LOG]")
    formatted_message = f"{timestamp} {prefix} {message}\n"

    if LOGGER_INSTANCE_FOR_UTILS and hasattr(LOGGER_INSTANCE_FOR_UTILS, 'write'):
        if level == "info" and LOG_LEVEL_INFO_UTILS:
            LOGGER_INSTANCE_FOR_UTILS.write(formatted_message)
        elif level == "debug" and LOG_LEVEL_DEBUG_UTILS:
            LOGGER_INSTANCE_FOR_UTILS.write(formatted_message)
        elif level in ["warn", "error"]:  # Always log warnings and errors
            LOGGER_INSTANCE_FOR_UTILS.write(formatted_message)
    else:
        _ORIGINAL_STDOUT_LOGGER.write(formatted_message)  # Fallback
        _ORIGINAL_STDOUT_LOGGER.flush()