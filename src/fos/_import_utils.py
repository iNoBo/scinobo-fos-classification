"""
This module provides utility functions for importing resources and handling data paths.
"""

from pathlib import Path
import os


DATA_PATH = os.path.join(Path(__file__).parent, "data")
LOGGING_PATH = os.path.join(Path(__file__).parent, "logs")