import logging
import sys
from datetime import datetime
from pathlib import Path

# Default logger (used before exp_name is set)
_default_logger = logging.getLogger(__name__)

def setup_logger(exp_name: str | None = None, log_dir: str = "logs"):
    """
    Setup logging with both console and file handlers.

    Args:
        exp_name: Experiment name for log file (uses timestamp if None)
        log_dir: Directory for log files (default: logs/)
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate log filename
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{exp_name}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger

# Initial console-only logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
