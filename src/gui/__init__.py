"""GUI components for mood classifier."""

try:
    from .main_window import MainWindow
    __all__ = ['MainWindow']
except ImportError:
    # PyQt5 not installed
    pass
