"""
Configuration management for attention mechanism research.
Handles experiment configurations, validation, and serialization.
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model: str = 'resnet18'
    attention_type: str = 'None'
    num_classes: int = 7
    image_size: int = 48
    image_channels: int = 1
    
    def __post_init__(self):
        """Validate model configuration."""
        valid_models = ['resnet50', 'resnet18', 'vgg16']
        valid_attentions = ['None', 'CBAM', 'BAM', 'scSE']
        
        if self.model not in valid_models:
            raise ValueError(f"Invalid model: {self.model}. Valid options: {valid_models}")
        if self.attention_type not in valid_attentions:
            raise ValueError(f"Invalid attention: {self.attention_type}. Valid options: {valid_attentions}")
        if self.num_classes <= 0:
            raise ValueError(f"Number of classes must be positive: {self.num_classes}")
        if self.image_size <= 0:
            raise ValueError(f"Image size must be positive: {self.image_size}")


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 120
    batch_size: int = 32
    lr: float = 0.0001
    optimizer: str = 'adamax'
    lr_scheduler: str = 'ExponentialDecay'
    early_stopping: int = 50
    seed: int = 42
    
    def __post_init__(self):
        """Validate training configuration."""
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adadelta', 'adamax']
        valid_schedulers = ['ExponentialDecay', 'CosineDecay', 'None']
        
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {self.optimizer}. Valid options: {valid_optimizers}")
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {self.lr_scheduler}. Valid options: {valid_schedulers}")
        if self.epochs <= 0:
            raise ValueError(f"Epochs must be positive: {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive: {self.lr}")


@dataclass
class DataConfig:
    """Data configuration parameters."""
    train_folder: str = './data/dataset/FER-2013/train'
    valid_folder: str = './data/dataset/FER-2013/test'
    color_mode: str = 'grayscale'
    class_mode: str = 'sparse'
    
    def __post_init__(self):
        """Validate data configuration."""
        valid_color_modes = ['grayscale', 'rgb']
        valid_class_modes = ['sparse', 'categorical']
        
        if self.color_mode not in valid_color_modes:
            raise ValueError(f"Invalid color mode: {self.color_mode}. Valid options: {valid_color_modes}")
        if self.class_mode not in valid_class_modes:
            raise ValueError(f"Invalid class mode: {self.class_mode}. Valid options: {valid_class_modes}")


@dataclass
class LoggingConfig:
    """Logging and tracking configuration."""
    use_wandb: bool = True
    wandb_project_name: str = 'Attention_CNN_Research'
    wandb_run_name: str = ''
    wandb_api_key: Optional[str] = None
    result_path: str = './working'
    
    def __post_init__(self):
        """Validate logging configuration."""
        if self.use_wandb and not self.wandb_project_name:
            raise ValueError("WandB project name is required when WandB is enabled")


@dataclass
class ExperimentConfig:
    """Experiment configuration parameters."""
    experiment_mode: str = 'single'
    batch_results_dir: str = './experiment_results'
    statistical_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])
    batch_timeout: int = 7200
    
    def __post_init__(self):
        """Validate experiment configuration."""
        valid_modes = ['single', 'batch', 'statistical']
        if self.experiment_mode not in valid_modes:
            raise ValueError(f"Invalid experiment mode: {self.experiment_mode}. Valid options: {valid_modes}")
        if self.batch_timeout <= 0:
            raise ValueError(f"Batch timeout must be positive: {self.batch_timeout}")


@dataclass
class Config:
    """Complete configuration for experiments."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create configuration from command line arguments."""
        
        # Parse statistical seeds if provided as string
        statistical_seeds = args.statistical_seeds
        if isinstance(statistical_seeds, str):
            statistical_seeds = [int(s.strip()) for s in statistical_seeds.split(',')]
        
        return cls(
            model=ModelConfig(
                model=args.model,
                attention_type=args.attention_type,
                num_classes=args.num_classes,
                image_size=args.image_size,
                image_channels=args.image_channels
            ),
            training=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                optimizer=args.optimizer,
                lr_scheduler=args.lr_scheduler,
                early_stopping=args.early_stopping,
                seed=args.seed
            ),
            data=DataConfig(
                train_folder=args.train_folder,
                valid_folder=args.valid_folder,
                color_mode=args.color_mode,
                class_mode=args.class_mode
            ),
            logging=LoggingConfig(
                use_wandb=bool(args.use_wandb),
                wandb_project_name=args.wandb_project_name,
                wandb_run_name=args.wandb_run_name,
                wandb_api_key=args.wandb_api_key,
                result_path=args.result_path
            ),
            experiment=ExperimentConfig(
                experiment_mode=args.experiment_mode,
                batch_results_dir=args.batch_results_dir,
                statistical_seeds=statistical_seeds,
                batch_timeout=args.batch_timeout
            )
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_experiment_name(self) -> str:
        """Generate experiment name from configuration."""
        return f"{self.model.model}_{self.model.attention_type}_{self.model.image_size}x{self.model.image_size}"
    
    def validate(self) -> None:
        """Validate complete configuration."""
        # Individual validation is handled by __post_init__ methods
        
        # Cross-validation checks
        if self.data.color_mode == 'grayscale' and self.model.image_channels != 1:
            raise ValueError("Grayscale mode requires 1 image channel")
        if self.data.color_mode == 'rgb' and self.model.image_channels != 3:
            raise ValueError("RGB mode requires 3 image channels")
        
        # Path validation
        if not os.path.exists(self.data.train_folder):
            raise ValueError(f"Training folder does not exist: {self.data.train_folder}")
        if not os.path.exists(self.data.valid_folder):
            raise ValueError(f"Validation folder does not exist: {self.data.valid_folder}")


class ConfigManager:
    """Manages configuration creation, validation, and serialization."""
    
    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        """Create argument parser with all configuration options."""
        parser = argparse.ArgumentParser(
            description="Training script for CNN models with attention mechanisms",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Model configuration
        model_group = parser.add_argument_group('Model Configuration')
        model_group.add_argument('--model', default='resnet18', type=str,
                                choices=['resnet50', 'resnet18', 'vgg16'],
                                help='Model architecture')
        model_group.add_argument('--attention-type', default='None', type=str,
                                choices=['None', 'CBAM', 'BAM', 'scSE'],
                                help='Attention mechanism type')
        model_group.add_argument('--num-classes', default=7, type=int,
                                help='Number of classes in dataset')
        model_group.add_argument('--image-size', default=48, type=int,
                                help='Size of input image (square)')
        model_group.add_argument('--image-channels', default=1, type=int,
                                help='Number of input image channels')
        
        # Training configuration
        training_group = parser.add_argument_group('Training Configuration')
        training_group.add_argument('--epochs', default=120, type=int,
                                   help='Number of training epochs')
        training_group.add_argument('--batch-size', default=32, type=int,
                                   help='Batch size for training')
        training_group.add_argument('--lr', default=0.0001, type=float,
                                   help='Initial learning rate')
        training_group.add_argument('--optimizer', default='adamax', type=str,
                                   choices=['adam', 'sgd', 'rmsprop', 'adadelta', 'adamax'],
                                   help='Optimizer type')
        training_group.add_argument('--lr-scheduler', default='ExponentialDecay', type=str,
                                   choices=['ExponentialDecay', 'CosineDecay', 'None'],
                                   help='Learning rate scheduler')
        training_group.add_argument('--early-stopping', default=50, type=int,
                                   help='Early stopping patience')
        training_group.add_argument('--seed', default=42, type=int,
                                   help='Random seed for reproducibility')
        
        # Data configuration
        data_group = parser.add_argument_group('Data Configuration')
        data_group.add_argument('--train-folder', default='/kaggle/input/fer2013/train', type=str,
                               help='Path to training data directory')
        data_group.add_argument('--valid-folder', default='/kaggle/input/fer2013/test', type=str,
                               help='Path to validation data directory')
        data_group.add_argument('--color-mode', default='grayscale', type=str,
                               choices=['grayscale', 'rgb'],
                               help='Color mode for input images')
        data_group.add_argument('--class-mode', default='sparse', type=str,
                               choices=['sparse', 'categorical'],
                               help='Class mode for data generator')
        
        # Logging configuration
        logging_group = parser.add_argument_group('Logging Configuration')
        logging_group.add_argument('--use-wandb', default=1, type=int,
                                  help='Enable Weights & Biases logging (1/0)')
        logging_group.add_argument('--wandb-project-name', default='Attention_CNN_Research', type=str,
                                  help='Wandb project name')
        logging_group.add_argument('--wandb-run-name', default='', type=str,
                                  help='Wandb run name')
        logging_group.add_argument('--wandb-api-key', default=None, type=str,
                                  help='Wandb API key (use env var WANDB_API_KEY)')
        logging_group.add_argument('--result-path', type=str, default="./working",
                                  help='Directory to save results')
        
        # Experiment configuration
        experiment_group = parser.add_argument_group('Experiment Configuration')
        experiment_group.add_argument('--experiment-mode', default='single', type=str,
                                     choices=['single', 'batch', 'statistical'],
                                     help='Experiment mode')
        experiment_group.add_argument('--batch-results-dir', default='./experiment_results', type=str,
                                     help='Directory to save batch experiment results')
        experiment_group.add_argument('--statistical-seeds', default='42,123,456,789,999', type=str,
                                     help='Comma-separated seeds for statistical mode')
        experiment_group.add_argument('--batch-timeout', default=7200, type=int,
                                     help='Timeout for each experiment in batch mode (seconds)')
        
        # Additional options
        parser.add_argument('--config-file', type=str, default=None,
                           help='Load configuration from JSON file')
        parser.add_argument('--save-config', type=str, default=None,
                           help='Save configuration to JSON file')
        
        return parser
    
    @staticmethod
    def load_config(args: argparse.Namespace) -> Config:
        """Load configuration from arguments or file."""
        if args.config_file:
            config = Config.from_file(args.config_file)
        else:
            config = Config.from_args(args)
        
        # Validate configuration
        config.validate()
        
        # Save configuration if requested
        if args.save_config:
            config.to_file(args.save_config)
            print(f"Configuration saved to: {args.save_config}")
        
        return config


def create_default_configs() -> Dict[str, Config]:
    """Create default configurations for common experiments."""
    
    configs = {}
    
    # ResNet50 + CBAM configuration
    configs['resnet50_cbam'] = Config(
        model=ModelConfig(model='resnet50', attention_type='CBAM'),
        training=TrainingConfig(epochs=120, lr=0.0001),
        experiment=ExperimentConfig(experiment_mode='single')
    )
    
    # ResNet18 adaptive configuration
    configs['resnet18_adaptive'] = Config(
        model=ModelConfig(model='resnet18', attention_type='CBAM', image_size=32),
        training=TrainingConfig(epochs=120, lr=0.0001),
        experiment=ExperimentConfig(experiment_mode='single')
    )
    
    # Batch experiment configuration
    configs['batch_experiment'] = Config(
        experiment=ExperimentConfig(experiment_mode='batch')
    )
    
    # Statistical experiment configuration
    configs['statistical_experiment'] = Config(
        training=TrainingConfig(epochs=50),  # Shorter for statistical runs
        experiment=ExperimentConfig(experiment_mode='statistical')
    )
    
    return configs


if __name__ == "__main__":
    # Test configuration system
    parser = ConfigManager.create_argument_parser()
    args = parser.parse_args(['--model', 'resnet18', '--attention-type', 'CBAM'])
    
    config = ConfigManager.load_config(args)
    print("Configuration loaded successfully:")
    print(json.dumps(config.to_dict(), indent=2))