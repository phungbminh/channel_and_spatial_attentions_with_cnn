"""
CNN Attention Mechanisms Research Framework

A modular, research-grade implementation for comparing attention mechanisms
(CBAM, BAM, scSE) with CNN architectures (ResNet50, ResNet18, VGG16).
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Main API imports for convenience
from .config.config import Config, ConfigManager
from .models.model_factory import ModelFactory, create_model_from_config
from .training.trainer import Trainer, run_single_experiment
from .experiments.experiment_manager import ExperimentManager

__all__ = [
    "Config",
    "ConfigManager", 
    "ModelFactory",
    "create_model_from_config",
    "Trainer",
    "run_single_experiment",
    "ExperimentManager"
]