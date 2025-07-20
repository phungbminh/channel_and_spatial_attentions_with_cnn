"""
Comprehensive Metrics Tracker for Research Paper
Tracks additional metrics beyond basic accuracy/loss
"""

import os
import json
import time
import psutil
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class ComprehensiveMetricsTracker(Callback):
    """Custom callback to track comprehensive metrics for research paper"""
    
    def __init__(self, validation_data, class_names, save_dir, model_name):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.save_dir = save_dir
        self.model_name = model_name
        
        # Metrics storage
        self.metrics_history = {
            'epoch': [],
            'train_accuracy': [],
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': [],
            'epoch_time': [],
            'memory_usage_mb': [],
            'gpu_usage_percent': []
        }
        
        # Model info
        self.model_info = {
            'model_name': model_name,
            'total_params': 0,
            'trainable_params': 0,
            'non_trainable_params': 0,
            'model_size_mb': 0
        }
        
        # Training start time
        self.training_start_time = None
        self.epoch_start_time = None
        
        # Best metrics
        self.best_metrics = {
            'best_val_accuracy': 0.0,
            'best_val_loss': float('inf'),
            'best_val_f1': 0.0,
            'best_epoch': 0
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.training_start_time = time.time()
        
        # Calculate model parameters
        self.model_info['total_params'] = self.model.count_params()
        trainable_count = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        self.model_info['trainable_params'] = trainable_count
        self.model_info['non_trainable_params'] = non_trainable_count
        
        # Estimate model size
        self.model_info['model_size_mb'] = self.estimate_model_size()
        
        print(f"\n📊 Model Info:")
        print(f"   Total params: {self.model_info['total_params']:,}")
        print(f"   Trainable params: {trainable_count:,}")
        print(f"   Non-trainable params: {non_trainable_count:,}")
        print(f"   Estimated size: {self.model_info['model_size_mb']:.2f} MB")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        epoch_time = time.time() - self.epoch_start_time
        
        # Get current learning rate
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        # Get memory usage
        memory_usage = psutil.virtual_memory().percent
        
        # Get GPU usage (if available)
        gpu_usage = self.get_gpu_usage()
        
        # Store basic metrics
        self.metrics_history['epoch'].append(epoch + 1)
        self.metrics_history['train_accuracy'].append(logs.get('accuracy', 0))
        self.metrics_history['train_loss'].append(logs.get('loss', 0))
        self.metrics_history['val_accuracy'].append(logs.get('val_accuracy', 0))
        self.metrics_history['val_loss'].append(logs.get('val_loss', 0))
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['epoch_time'].append(epoch_time)
        self.metrics_history['memory_usage_mb'].append(memory_usage)
        self.metrics_history['gpu_usage_percent'].append(gpu_usage)
        
        # Calculate detailed metrics on validation set
        val_metrics = self.calculate_detailed_metrics(self.validation_data)
        
        self.metrics_history['val_precision'].append(val_metrics['precision'])
        self.metrics_history['val_recall'].append(val_metrics['recall'])
        self.metrics_history['val_f1'].append(val_metrics['f1'])
        
        # For training metrics, we'll use the same values as validation for now
        # (calculating on full training set every epoch is expensive)
        self.metrics_history['train_precision'].append(val_metrics['precision'])
        self.metrics_history['train_recall'].append(val_metrics['recall'])
        self.metrics_history['train_f1'].append(val_metrics['f1'])
        
        # Update best metrics
        val_acc = logs.get('val_accuracy', 0)
        val_loss = logs.get('val_loss', float('inf'))
        val_f1 = val_metrics['f1']
        
        if val_acc > self.best_metrics['best_val_accuracy']:
            self.best_metrics['best_val_accuracy'] = val_acc
            self.best_metrics['best_val_loss'] = val_loss
            self.best_metrics['best_val_f1'] = val_f1
            self.best_metrics['best_epoch'] = epoch + 1
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}: "
              f"val_acc={val_acc:.4f}, "
              f"val_f1={val_f1:.4f}, "
              f"lr={lr:.6f}, "
              f"time={epoch_time:.1f}s")
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        training_time = time.time() - self.training_start_time
        
        # Save all metrics
        self.save_metrics(training_time)
        
        # Generate confusion matrix
        self.generate_confusion_matrix()
        
        # Generate classification report
        self.generate_classification_report()
        
        # Save model architecture
        self.save_model_architecture()
        
        print(f"\n🎯 Training completed in {training_time/60:.1f} minutes")
        print(f"📊 Best validation accuracy: {self.best_metrics['best_val_accuracy']:.4f} (epoch {self.best_metrics['best_epoch']})")
        print(f"🔍 All metrics saved to: {self.save_dir}")
    
    def calculate_detailed_metrics(self, data_generator):
        """Calculate precision, recall, F1 score"""
        # Get predictions
        predictions = self.model.predict(data_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = data_generator.classes
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def estimate_model_size(self):
        """Estimate model size in MB"""
        total_params = self.model.count_params()
        # Assume 32-bit floats (4 bytes per parameter)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    
    def get_gpu_usage(self):
        """Get GPU memory usage percentage"""
        try:
            if tf.config.list_physical_devices('GPU'):
                gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
                current = gpu_stats['current'] / (1024**2)  # MB
                peak = gpu_stats['peak'] / (1024**2)  # MB
                return (current / peak) * 100 if peak > 0 else 0
        except:
            pass
        return 0
    
    def save_metrics(self, training_time):
        """Save all metrics to JSON file"""
        
        # Prepare final metrics
        final_metrics = {
            'model_info': self.model_info,
            'training_summary': {
                'total_epochs': len(self.metrics_history['epoch']),
                'training_time_minutes': training_time / 60,
                'average_epoch_time': np.mean(self.metrics_history['epoch_time']),
                'final_learning_rate': self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else 0
            },
            'best_metrics': self.best_metrics,
            'epoch_history': self.metrics_history
        }
        
        # Save to JSON
        metrics_path = os.path.join(self.save_dir, 'comprehensive_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"📊 Comprehensive metrics saved to: {metrics_path}")
    
    def generate_confusion_matrix(self):
        """Generate and save confusion matrix"""
        val_metrics = self.calculate_detailed_metrics(self.validation_data)
        
        # Create confusion matrix
        cm = confusion_matrix(val_metrics['y_true'], val_metrics['y_pred'])
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        cm_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix data
        cm_data = {
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names,
            'accuracy_per_class': cm.diagonal() / cm.sum(axis=1)
        }
        
        cm_json_path = os.path.join(self.save_dir, 'confusion_matrix.json')
        with open(cm_json_path, 'w') as f:
            json.dump(cm_data, f, indent=2)
        
        print(f"📈 Confusion matrix saved to: {cm_path}")
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        val_metrics = self.calculate_detailed_metrics(self.validation_data)
        
        # Generate classification report
        report = classification_report(
            val_metrics['y_true'], val_metrics['y_pred'],
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save report
        report_path = os.path.join(self.save_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save readable report
        report_text = classification_report(
            val_metrics['y_true'], val_metrics['y_pred'],
            target_names=self.class_names
        )
        
        report_txt_path = os.path.join(self.save_dir, 'classification_report.txt')
        with open(report_txt_path, 'w') as f:
            f.write(f"Classification Report - {self.model_name}\n")
            f.write("="*50 + "\n")
            f.write(report_text)
        
        print(f"📋 Classification report saved to: {report_path}")
    
    def save_model_architecture(self):
        """Save model architecture summary"""
        
        # Save model summary
        summary_path = os.path.join(self.save_dir, 'model_architecture.txt')
        with open(summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Save model plot (if possible)
        try:
            plot_path = os.path.join(self.save_dir, 'model_architecture.png')
            tf.keras.utils.plot_model(
                self.model, 
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                dpi=150
            )
            print(f"🏗️  Model architecture plot saved to: {plot_path}")
        except:
            print("⚠️  Could not generate model architecture plot")
        
        print(f"📝 Model architecture summary saved to: {summary_path}")


def create_metrics_tracker(validation_data, class_names, save_dir, model_name):
    """Factory function to create metrics tracker"""
    return ComprehensiveMetricsTracker(
        validation_data=validation_data,
        class_names=class_names,
        save_dir=save_dir,
        model_name=model_name
    )