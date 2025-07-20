"""
Model factory for creating CNN models with attention mechanisms.
Provides a unified interface for model creation and configuration.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adamax
from typing import Dict, Any, Tuple
import logging

from ..config.config import ModelConfig, TrainingConfig
from .model_cnn import resnet50
from .model_cnn_v2 import ResNet, VGG16

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating models with attention mechanisms."""
    
    @staticmethod
    def create_model(model_config: ModelConfig) -> Model:
        """
        Create a model based on configuration.
        
        Args:
            model_config: Model configuration parameters
            
        Returns:
            Configured Keras model
            
        Raises:
            ValueError: If model type is not supported
        """
        input_shape = (
            model_config.image_size, 
            model_config.image_size, 
            model_config.image_channels
        )
        
        logger.info(f"Creating model: {model_config.model} with {model_config.attention_type} attention")
        logger.info(f"Input shape: {input_shape}, Classes: {model_config.num_classes}")
        
        try:
            if model_config.model == 'resnet50':
                model = ModelFactory._create_resnet50(model_config, input_shape)
            elif model_config.model == 'resnet18':
                model = ModelFactory._create_resnet18(model_config, input_shape)
            elif model_config.model == 'vgg16':
                model = ModelFactory._create_vgg16(model_config, input_shape)
            else:
                raise ValueError(f"Unsupported model: {model_config.model}")
            
            # Build model to initialize weights
            model.build(input_shape=(None, *input_shape))
            
            # Log model information
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            
            logger.info(f"Model created successfully:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_config.model}: {str(e)}")
            raise
    
    @staticmethod
    def _create_resnet50(model_config: ModelConfig, input_shape: Tuple[int, int, int]) -> Model:
        """Create ResNet50 model."""
        return resnet50(
            input_shape=input_shape,
            num_classes=model_config.num_classes,
            attention_type=model_config.attention_type
        )
    
    @staticmethod
    def _create_resnet18(model_config: ModelConfig, input_shape: Tuple[int, int, int]) -> Model:
        """Create ResNet18 model with adaptive architecture."""
        return ResNet(
            model_name="ResNet18",
            input_shape=input_shape,
            attention=model_config.attention_type,
            pooling="avg",
            num_classes=model_config.num_classes
        )
    
    @staticmethod
    def _create_vgg16(model_config: ModelConfig, input_shape: Tuple[int, int, int]) -> Model:
        """Create VGG16 model."""
        return VGG16(
            input_shape=input_shape,
            num_classes=model_config.num_classes,
            attention_type=model_config.attention_type
        )
    
    @staticmethod
    def create_optimizer(training_config: TrainingConfig) -> tf.keras.optimizers.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            training_config: Training configuration parameters
            
        Returns:
            Configured optimizer
            
        Raises:
            ValueError: If optimizer type is not supported
        """
        lr_schedule = ModelFactory._create_lr_schedule(training_config)
        
        optimizer_map = {
            'adam': Adam,
            'sgd': SGD,
            'rmsprop': RMSprop,
            'adadelta': Adadelta,
            'adamax': Adamax
        }
        
        if training_config.optimizer not in optimizer_map:
            raise ValueError(
                f"Unsupported optimizer: {training_config.optimizer}. "
                f"Supported: {list(optimizer_map.keys())}"
            )
        
        optimizer_class = optimizer_map[training_config.optimizer]
        optimizer = optimizer_class(learning_rate=lr_schedule)
        
        logger.info(f"Created optimizer: {training_config.optimizer} with LR: {training_config.lr}")
        
        return optimizer
    
    @staticmethod
    def _create_lr_schedule(training_config: TrainingConfig):
        """Create learning rate schedule."""
        if training_config.lr_scheduler == 'ExponentialDecay':
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=training_config.lr,
                decay_steps=10000,
                decay_rate=0.9,
                name='ExponentialDecay'
            )
        elif training_config.lr_scheduler == 'CosineDecay':
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=training_config.lr,
                decay_steps=5,
                alpha=0.0,
                name="CosineDecay",
                warmup_target=None,
                warmup_steps=0
            )
        else:
            # Return constant learning rate
            return training_config.lr
    
    @staticmethod
    def get_model_info(model: Model) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model: Keras model
            
        Returns:
            Dictionary with model information
        """
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Estimate model size (assuming 32-bit floats)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            'name': model.name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'model_size_mb': model_size_mb,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
    
    @staticmethod
    def validate_model_config(model_config: ModelConfig) -> None:
        """
        Validate model configuration for compatibility.
        
        Args:
            model_config: Model configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check input size compatibility
        if model_config.model == 'resnet18' and model_config.image_size not in [32, 48, 64]:
            logger.warning(
                f"ResNet18 is optimized for sizes 32, 48, 64. "
                f"Using {model_config.image_size} may affect performance."
            )
        
        # Check attention mechanism compatibility
        if model_config.attention_type not in ['None', 'CBAM', 'BAM', 'scSE']:
            raise ValueError(f"Unsupported attention type: {model_config.attention_type}")
        
        # Check channel compatibility with color mode
        if model_config.image_channels == 1:
            expected_color = 'grayscale'
        elif model_config.image_channels == 3:
            expected_color = 'rgb'
        else:
            raise ValueError(f"Unsupported number of channels: {model_config.image_channels}")
        
        logger.info(f"Model configuration validated successfully")
        logger.info(f"Expected color mode: {expected_color}")


class ModelCompiler:
    """Handles model compilation with appropriate loss and metrics."""
    
    @staticmethod
    def compile_model(
        model: Model,
        optimizer: tf.keras.optimizers.Optimizer,
        num_classes: int,
        class_mode: str = 'sparse'
    ) -> None:
        """
        Compile model with appropriate loss function and metrics.
        
        Args:
            model: Keras model to compile
            optimizer: Configured optimizer
            num_classes: Number of output classes
            class_mode: Type of class encoding ('sparse' or 'categorical')
        """
        # Choose loss function based on class mode
        if class_mode == 'sparse':
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            metrics = ['accuracy']
        elif class_mode == 'categorical':
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            metrics = ['accuracy', 'top_2_accuracy'] if num_classes > 2 else ['accuracy']
        else:
            raise ValueError(f"Unsupported class mode: {class_mode}")
        
        # Additional metrics for multi-class problems
        if num_classes > 2:
            metrics.extend([
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ])
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {class_mode} loss and {len(metrics)} metrics")


def create_model_from_config(model_config: ModelConfig, training_config: TrainingConfig) -> Tuple[Model, tf.keras.optimizers.Optimizer]:
    """
    Convenience function to create and configure a complete model.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        
    Returns:
        Tuple of (compiled_model, optimizer)
    """
    # Validate configuration
    ModelFactory.validate_model_config(model_config)
    
    # Create model and optimizer
    model = ModelFactory.create_model(model_config)
    optimizer = ModelFactory.create_optimizer(training_config)
    
    # Compile model
    ModelCompiler.compile_model(model, optimizer, model_config.num_classes)
    
    return model, optimizer


if __name__ == "__main__":
    # Test model factory
    from ..config.config import ModelConfig, TrainingConfig
    
    # Test ResNet18 with CBAM
    model_config = ModelConfig(
        model='resnet18',
        attention_type='CBAM',
        image_size=48,
        num_classes=7
    )
    
    training_config = TrainingConfig(
        optimizer='adamax',
        lr=0.0001
    )
    
    model, optimizer = create_model_from_config(model_config, training_config)
    print("Model created and compiled successfully!")
    
    # Print model info
    info = ModelFactory.get_model_info(model)
    for key, value in info.items():
        print(f"{key}: {value}")