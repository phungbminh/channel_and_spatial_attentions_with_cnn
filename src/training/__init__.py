"""
Training orchestration and management.

This package contains:
- Modular training system
- Data management
- Callback management
- WandB integration
"""

from .trainer import Trainer, run_single_experiment, DataManager, CallbackManager, WandBManager

__all__ = [
    "Trainer",
    "run_single_experiment",
    "DataManager",
    "CallbackManager", 
    "WandBManager"
]