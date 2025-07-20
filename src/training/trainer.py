"""
Training module for CNN models with attention mechanisms.
Handles data preparation, training execution, and result collection.
"""

import os
import pickle
import logging
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras import Model

import wandb
from wandb.integration.keras import WandbMetricsLogger

from ..config.config import Config
from ..models.model_factory import ModelFactory, create_model_from_config
from ..utils.metrics_tracker import create_metrics_tracker

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading and preprocessing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.train_generator = None
        self.val_generator = None
        self.class_names = []
        
    def setup_data_generators(self) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence, List[str]]:
        """
        Setup training and validation data generators.
        
        Returns:
            Tuple of (train_generator, val_generator, class_names)
        """
        logger.info("Setting up data generators...")
        
        # Validate data paths
        if not os.path.exists(self.config.data.train_folder):
            raise FileNotFoundError(f"Training folder not found: {self.config.data.train_folder}")
        if not os.path.exists(self.config.data.valid_folder):
            raise FileNotFoundError(f"Validation folder not found: {self.config.data.valid_folder}")
        
        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        
        # Setup generators
        target_size = (self.config.model.image_size, self.config.model.image_size)
        
        self.train_generator = train_datagen.flow_from_directory(
            self.config.data.train_folder,
            target_size=target_size,
            batch_size=self.config.training.batch_size,
            class_mode=self.config.data.class_mode,
            color_mode=self.config.data.color_mode,
            seed=self.config.training.seed,
            shuffle=True
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.config.data.valid_folder,
            target_size=target_size,
            batch_size=self.config.training.batch_size,
            class_mode=self.config.data.class_mode,
            color_mode=self.config.data.color_mode,
            seed=self.config.training.seed,
            shuffle=False
        )
        
        # Extract class names
        self.class_names = list(self.train_generator.class_indices.keys())
        
        logger.info(f"Data generators created successfully:")
        logger.info(f"  Training samples: {self.train_generator.samples}")
        logger.info(f"  Validation samples: {self.val_generator.samples}")
        logger.info(f"  Classes: {self.class_names}")
        logger.info(f"  Image size: {target_size}")
        logger.info(f"  Color mode: {self.config.data.color_mode}")
        
        return self.train_generator, self.val_generator, self.class_names
    
    def save_class_names(self, save_path: str) -> None:
        """Save class names for later use."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.class_names, f)
        logger.info(f"Class names saved to: {save_path}")


class CallbackManager:
    """Manages training callbacks."""
    
    def __init__(self, config: Config, save_dir: str, model_name: str):
        self.config = config
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.callbacks = []
        
    def setup_callbacks(self, val_generator, class_names: List[str]) -> List:
        """
        Setup training callbacks.
        
        Args:
            val_generator: Validation data generator
            class_names: List of class names
            
        Returns:
            List of configured callbacks
        """
        logger.info("Setting up training callbacks...")
        
        # WandB callback
        if self.config.logging.use_wandb:
            wandb_callback = WandbMetricsLogger(log_freq=1)
            self.callbacks.append(wandb_callback)
            logger.info("Added WandB metrics logger")
        
        # CSV logger
        csv_path = self.save_dir / 'training_log.csv'
        csv_callback = CSVLogger(str(csv_path))
        self.callbacks.append(csv_callback)
        logger.info(f"Added CSV logger: {csv_path}")
        
        # Comprehensive metrics tracker
        metrics_callback = create_metrics_tracker(
            validation_data=val_generator,
            class_names=class_names,
            save_dir=str(self.save_dir),
            model_name=self.model_name
        )
        self.callbacks.append(metrics_callback)
        logger.info("Added comprehensive metrics tracker")
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.training.early_stopping,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        self.callbacks.append(early_stopping)
        logger.info(f"Added early stopping with patience: {self.config.training.early_stopping}")
        
        # Model checkpoint
        model_path = self.save_dir / f"{self.model_name}_best.h5"
        checkpoint = ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        self.callbacks.append(checkpoint)
        logger.info(f"Added model checkpoint: {model_path}")
        
        return self.callbacks


class WandBManager:
    """Manages Weights & Biases integration."""
    
    @staticmethod
    def setup_wandb(config: Config) -> None:
        """Setup WandB logging."""
        if not config.logging.use_wandb:
            return
        
        try:
            # Login to WandB
            api_key = config.logging.wandb_api_key or os.getenv('WANDB_API_KEY')
            if api_key:
                wandb.login(key=api_key)
            else:
                wandb.login()  # Use saved credentials
            
            # Initialize WandB run
            run_name = config.logging.wandb_run_name or config.get_experiment_name()
            
            wandb.init(
                project=config.logging.wandb_project_name,
                name=run_name,
                config=config.to_dict(),
                reinit=True
            )
            
            logger.info(f"WandB initialized: {config.logging.wandb_project_name}/{run_name}")
            
        except Exception as e:
            logger.warning(f"Failed to setup WandB: {str(e)}")
            logger.warning("Continuing without WandB logging")
    
    @staticmethod
    def finish_wandb() -> None:
        """Finish WandB run."""
        try:
            if wandb.run is not None:
                wandb.finish()
                logger.info("WandB run finished")
        except Exception as e:
            logger.warning(f"Error finishing WandB run: {str(e)}")


class Trainer:
    """Main training orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.data_manager = None
        self.callback_manager = None
        
        # Set up logging
        self._setup_logging()
        
        # Set random seed for reproducibility
        self._set_seed()
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(self.config.logging.result_path, 'training.log')
                )
            ]
        )
        
    def _set_seed(self) -> None:
        """Set random seed for reproducibility."""
        import random
        
        seed = self.config.training.seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Enable deterministic operations
        tf.config.experimental.enable_op_determinism()
        
        logger.info(f"Random seed set to: {seed}")
    
    def setup_experiment(self) -> None:
        """Setup experiment environment."""
        # Create result directory
        os.makedirs(self.config.logging.result_path, exist_ok=True)
        
        # Setup WandB
        WandBManager.setup_wandb(self.config)
        
        # Setup data
        self.data_manager = DataManager(self.config)
        
        # Create model
        self.model, self.optimizer = create_model_from_config(
            self.config.model, 
            self.config.training
        )
        
        logger.info("Experiment setup completed")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute training process.
        
        Returns:
            Dictionary with training results
        """
        logger.info("🚀 Starting training process...")
        
        try:
            # Setup data generators
            train_gen, val_gen, class_names = self.data_manager.setup_data_generators()
            
            # Save class names
            class_names_path = os.path.join(self.config.logging.result_path, 'class_names.pkl')
            self.data_manager.save_class_names(class_names_path)
            
            # Setup callbacks
            model_name = self.config.get_experiment_name()
            self.callback_manager = CallbackManager(
                self.config, 
                self.config.logging.result_path, 
                model_name
            )
            callbacks = self.callback_manager.setup_callbacks(val_gen, class_names)
            
            # Display model summary
            self.model.summary()
            model_info = ModelFactory.get_model_info(self.model)
            logger.info(f"Model info: {model_info}")
            
            # Train model
            logger.info(f"Training for {self.config.training.epochs} epochs...")
            
            history = self.model.fit(
                train_gen,
                epochs=self.config.training.epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
            
            # Extract training results
            results = self._extract_training_results(history)
            
            logger.info("✅ Training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise
        
        finally:
            # Cleanup WandB
            WandBManager.finish_wandb()
    
    def _extract_training_results(self, history) -> Dict[str, Any]:
        """Extract and format training results."""
        try:
            # Get best metrics
            val_accuracies = history.history.get('val_accuracy', [])
            val_losses = history.history.get('val_loss', [])
            train_accuracies = history.history.get('accuracy', [])
            train_losses = history.history.get('loss', [])
            
            results = {
                'experiment_name': self.config.get_experiment_name(),
                'config': self.config.to_dict(),
                'training_completed': True,
                'total_epochs': len(val_accuracies),
                'best_val_accuracy': max(val_accuracies) if val_accuracies else None,
                'best_val_loss': min(val_losses) if val_losses else None,
                'final_train_accuracy': train_accuracies[-1] if train_accuracies else None,
                'final_train_loss': train_losses[-1] if train_losses else None,
                'model_info': ModelFactory.get_model_info(self.model)
            }
            
            # Log final results
            logger.info("📊 Training Results Summary:")
            logger.info(f"  Best validation accuracy: {results['best_val_accuracy']:.4f}")
            logger.info(f"  Best validation loss: {results['best_val_loss']:.4f}")
            logger.info(f"  Total epochs completed: {results['total_epochs']}")
            logger.info(f"  Model parameters: {results['model_info']['total_params']:,}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting training results: {str(e)}")
            return {'training_completed': False, 'error': str(e)}


def run_single_experiment(config: Config) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    trainer = Trainer(config)
    
    try:
        trainer.setup_experiment()
        results = trainer.train()
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        return {
            'experiment_name': config.get_experiment_name(),
            'training_completed': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test trainer
    from ..config.config import Config, ModelConfig, TrainingConfig
    
    # Create test configuration
    config = Config(
        model=ModelConfig(
            model='resnet18',
            attention_type='CBAM',
            image_size=48
        ),
        training=TrainingConfig(
            epochs=5,  # Short for testing
            batch_size=16
        )
    )
    
    config.logging.use_wandb = False  # Disable WandB for testing
    config.logging.result_path = "./test_training"
    
    try:
        results = run_single_experiment(config)
        print("Test experiment completed:")
        print(f"Success: {results.get('training_completed', False)}")
        if results.get('best_val_accuracy'):
            print(f"Best accuracy: {results['best_val_accuracy']:.4f}")
    except Exception as e:
        print(f"Test failed: {str(e)}")