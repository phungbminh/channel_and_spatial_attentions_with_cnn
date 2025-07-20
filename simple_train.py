#!/usr/bin/env python3
"""
Simple training script for 3 datasets
Optimized for Kaggle and minimal dependencies
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add src to path for imports
sys.path.append('src')

try:
    from models.model_cnn import vgg16, resnet18
    from models.model_cnn_v2 import VGG16, ResNet
except ImportError:
    print("❌ Error importing models. Make sure src/models/ exists and is accessible.")
    sys.exit(1)

def create_data_generators(train_path, valid_path, image_size, batch_size, color_mode='rgb'):
    """Create data generators for training and validation"""
    
    if color_mode == 'rgb':
        color_mode_keras = 'rgb'
        channels = 3
    else:
        color_mode_keras = 'grayscale'
        channels = 1
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode=color_mode_keras,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        valid_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode=color_mode_keras,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    return train_generator, valid_generator

def create_model(model_name, attention_type, num_classes, input_shape):
    """Create model with specified architecture and attention"""
    
    if model_name == 'vgg16':
        if len(input_shape) == 3 and input_shape[2] == 3:
            # Use RGB version
            model = vgg16(num_classes=num_classes, input_shape=input_shape, attention_type=attention_type)
        else:
            # Use grayscale version (will be converted to RGB)
            model = vgg16(num_classes=num_classes, input_shape=input_shape, attention_type=attention_type)
    
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, input_shape=input_shape, attention_type=attention_type)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def train_model(args):
    """Main training function"""
    
    print(f"🚀 Starting training: {args.model} + {args.attention} on {args.dataset}")
    print(f"📐 Image size: {args.image_size}x{args.image_size}")
    print(f"🎯 Classes: {args.num_classes}")
    print(f"📊 Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Create input shape
    input_shape = (args.image_size, args.image_size, 3)  # Always use RGB
    
    # Create data generators
    train_gen, valid_gen = create_data_generators(
        train_path=args.train_path,
        valid_path=args.valid_path, 
        image_size=args.image_size,
        batch_size=args.batch_size,
        color_mode='rgb'
    )
    
    print(f"📁 Training samples: {train_gen.samples}")
    print(f"📁 Validation samples: {valid_gen.samples}")
    print(f"🏷️  Classes found: {list(train_gen.class_indices.keys())}")
    
    # Create model
    model = create_model(
        model_name=args.model,
        attention_type=args.attention,
        num_classes=args.num_classes,
        input_shape=input_shape
    )
    
    # Compile model
    optimizer = Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created and compiled")
    print(f"📊 Total parameters: {model.count_params():,}")
    
    # Create output directory
    output_dir = f"models/{args.dataset}_{args.model}_{args.attention}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=f"{output_dir}/best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Calculate steps
    steps_per_epoch = train_gen.samples // args.batch_size
    validation_steps = valid_gen.samples // args.batch_size
    
    print(f"🔢 Steps per epoch: {steps_per_epoch}")
    print(f"🔢 Validation steps: {validation_steps}")
    
    # Train model
    print(f"\n🎯 Starting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = f"{output_dir}/{args.model}_{args.attention}_{args.dataset}.h5"
    model.save(final_model_path)
    
    print(f"\n🎉 Training completed!")
    print(f"💾 Best model saved: {output_dir}/best_model.h5")
    print(f"💾 Final model saved: {final_model_path}")
    
    # Print best results
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    return history

def main():
    parser = argparse.ArgumentParser(description='Simple training script for CNN with attention')
    
    parser.add_argument('--dataset', required=True, choices=['fer2013', 'rafdb', 'cifar10'],
                        help='Dataset name')
    parser.add_argument('--model', required=True, choices=['vgg16', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--attention', required=True, choices=['CBAM', 'BAM', 'scSE', 'None'],
                        help='Attention mechanism')
    parser.add_argument('--train-path', required=True, help='Path to training data')
    parser.add_argument('--valid-path', required=True, help='Path to validation data')
    parser.add_argument('--image-size', type=int, required=True, help='Image size (square)')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # GPU configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🖥️  GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("💻 Using CPU")
    
    # Start training
    train_model(args)

if __name__ == "__main__":
    main()