"""
Experiment management and batch processing.

This package contains:
- Systematic experiment execution
- Batch processing capabilities
- Result collection and analysis
- Statistical significance testing
"""

from .experiment_manager import ExperimentManager, ExperimentGenerator, ExperimentBatch, ExperimentExecutor

__all__ = [
    "ExperimentManager",
    "ExperimentGenerator",
    "ExperimentBatch", 
    "ExperimentExecutor"
]