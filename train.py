#!/usr/bin/env python3
"""
Modular training script for CNN models with attention mechanisms.
Research-grade implementation with comprehensive experiment management.

Usage:
    Single experiment:   python train.py --model resnet18 --attention-type CBAM
    Batch experiments:   python train.py --experiment-mode batch
    Statistical tests:   python train.py --experiment-mode statistical
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

from src.config.config import Config, ConfigManager
from src.training.trainer import Trainer, run_single_experiment
from src.experiments.experiment_manager import ExperimentManager


def setup_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_banner() -> None:
    """Print application banner."""
    print("=" * 80)
    print("🔬 CNN ATTENTION MECHANISMS RESEARCH FRAMEWORK")
    print("=" * 80)
    print("📊 Models: ResNet50, ResNet18, VGG16")
    print("🎯 Attention: CBAM, BAM, scSE")
    print("🧪 Experiment Modes: Single, Batch, Statistical")
    print("=" * 80)


def run_single_mode(config: Config) -> Dict[str, Any]:
    """
    Run single experiment mode.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 Running single experiment mode")
    
    try:
        results = run_single_experiment(config)
        
        if results.get('training_completed', False):
            logger.info("✅ Single experiment completed successfully")
            logger.info(f"📊 Best validation accuracy: {results.get('best_val_accuracy', 'N/A'):.4f}")
        else:
            logger.error("❌ Single experiment failed")
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        logger.error(f"💥 Single experiment crashed: {str(e)}")
        return {'training_completed': False, 'error': str(e)}


def run_batch_mode(config: Config) -> None:
    """
    Run batch experiment mode.
    
    Args:
        config: Base configuration for batch experiments
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 Running batch experiment mode")
    
    try:
        # Create experiment manager
        manager = ExperimentManager(config.experiment.batch_results_dir)
        
        # Run batch experiments
        batch = manager.run_experiment_batch(config)
        
        # Get summary
        summary = batch.get_summary()
        
        logger.info("✅ Batch experiments completed")
        logger.info(f"📊 Success rate: {summary['success_count']}/{summary['total_experiments']}")
        
        if summary.get('best_accuracy'):
            logger.info(f"🏆 Best accuracy: {summary['best_accuracy']:.4f}")
            logger.info(f"🥇 Best experiment: {summary.get('best_experiment', 'N/A')}")
        
        logger.info(f"💾 Results saved to: {summary['batch_dir']}")
        
    except Exception as e:
        logger.error(f"💥 Batch experiments failed: {str(e)}")
        raise


def run_statistical_mode(config: Config) -> None:
    """
    Run statistical significance experiment mode.
    
    Args:
        config: Base configuration for statistical experiments
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 Running statistical significance mode")
    
    try:
        # Create experiment manager
        manager = ExperimentManager(config.experiment.batch_results_dir)
        
        # Run statistical experiments
        batch = manager.run_experiment_batch(config)
        
        # Get summary
        summary = batch.get_summary()
        
        logger.info("✅ Statistical experiments completed")
        logger.info(f"📊 Success rate: {summary['success_count']}/{summary['total_experiments']}")
        logger.info(f"🔬 Statistical seeds tested: {len(config.experiment.statistical_seeds)}")
        
        if summary.get('best_accuracy'):
            logger.info(f"🏆 Best accuracy: {summary['best_accuracy']:.4f}")
        
        logger.info(f"💾 Results saved to: {summary['batch_dir']}")
        logger.info("📈 Run result_analyzer.py for statistical analysis")
        
    except Exception as e:
        logger.error(f"💥 Statistical experiments failed: {str(e)}")
        raise


def validate_environment(config: Config) -> None:
    """
    Validate the execution environment.
    
    Args:
        config: Configuration to validate
        
    Raises:
        SystemExit: If validation fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        config.validate()
        logger.info("✅ Configuration validated successfully")
        
        # Check required directories
        os.makedirs(config.logging.result_path, exist_ok=True)
        os.makedirs(config.experiment.batch_results_dir, exist_ok=True)
        
        # Check TensorFlow GPU availability
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        logger.info(f"🖥️  GPU Available: {gpu_available}")
        
        if gpu_available:
            for gpu in tf.config.list_physical_devices('GPU'):
                logger.info(f"   {gpu}")
        
        # Memory configuration for GPU
        if gpu_available:
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("🔧 GPU memory growth enabled")
            except Exception as e:
                logger.warning(f"Could not configure GPU memory: {str(e)}")
        
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {str(e)}")
        sys.exit(1)


def print_experiment_summary(config: Config) -> None:
    """
    Print experiment configuration summary.
    
    Args:
        config: Experiment configuration
    """
    logger = logging.getLogger(__name__)
    
    logger.info("📋 Experiment Configuration:")
    logger.info(f"   Mode: {config.experiment.experiment_mode}")
    logger.info(f"   Model: {config.model.model}")
    logger.info(f"   Attention: {config.model.attention_type}")
    logger.info(f"   Image Size: {config.model.image_size}x{config.model.image_size}")
    logger.info(f"   Epochs: {config.training.epochs}")
    logger.info(f"   Batch Size: {config.training.batch_size}")
    logger.info(f"   Learning Rate: {config.training.lr}")
    logger.info(f"   Optimizer: {config.training.optimizer}")
    logger.info(f"   Seed: {config.training.seed}")
    
    if config.experiment.experiment_mode in ['batch', 'statistical']:
        logger.info(f"   Batch Results: {config.experiment.batch_results_dir}")
    
    if config.logging.use_wandb:
        logger.info(f"   WandB Project: {config.logging.wandb_project_name}")


def main() -> int:
    """
    Main entry point for the training script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup basic logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Print banner
        print_banner()
        
        # Parse arguments and load configuration
        parser = ConfigManager.create_argument_parser()
        args = parser.parse_args()
        config = ConfigManager.load_config(args)
        
        # Validate environment
        validate_environment(config)
        
        # Print experiment summary
        print_experiment_summary(config)
        
        # Route to appropriate experiment mode
        if config.experiment.experiment_mode == 'single':
            results = run_single_mode(config)
            return 0 if results.get('training_completed', False) else 1
            
        elif config.experiment.experiment_mode == 'batch':
            run_batch_mode(config)
            return 0
            
        elif config.experiment.experiment_mode == 'statistical':
            run_statistical_mode(config)
            return 0
            
        else:
            logger.error(f"❌ Unsupported experiment mode: {config.experiment.experiment_mode}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        logger.exception("Full traceback:")
        return 1
    
    finally:
        logger.info("🏁 Training script finished")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)