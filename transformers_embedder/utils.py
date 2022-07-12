import importlib.util
import logging

_torch_available = importlib.util.find_spec("torch") is not None


def is_torch_available():
    """Check if PyTorch is available."""
    return _torch_available


def get_logger(name: str) -> logging.Logger:
    """
    Return the logger of the given name.

    Args:
        name (`str`): The name of the logger.

    Returns:
        `logging.Logger`: The logger of the given name.
    """
    return logging.getLogger(name)
