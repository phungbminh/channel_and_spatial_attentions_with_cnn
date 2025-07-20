"""
Utility functions and analysis tools.

This package contains:
- Metrics tracking and analysis
- Result processing and visualization
- Statistical analysis tools
- Paper-ready output generation
"""

from .metrics_tracker import create_metrics_tracker
from .result_analyzer import *

__all__ = [
    "create_metrics_tracker"
]