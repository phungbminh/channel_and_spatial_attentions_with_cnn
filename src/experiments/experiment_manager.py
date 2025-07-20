"""
Experiment management for systematic research execution.
Handles experiment creation, execution, monitoring, and result collection.
"""

import os
import json
import subprocess
import time
import logging
import itertools
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict

from ..config.config import Config, ModelConfig, TrainingConfig, ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_name: str
    config: Dict[str, Any]
    status: str  # 'success', 'failed', 'timeout', 'pending'
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    best_val_accuracy: Optional[float] = None
    best_val_loss: Optional[float] = None
    final_train_accuracy: Optional[float] = None
    final_train_loss: Optional[float] = None
    total_params: Optional[int] = None
    training_time_minutes: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ExperimentGenerator:
    """Generates experiment configurations for different research scenarios."""
    
    @staticmethod
    def generate_main_comparison_experiments(base_config: Config) -> List[Config]:
        """Generate experiments for main model-attention comparison."""
        experiments = []
        
        models = ['resnet50', 'resnet18', 'vgg16']
        attention_types = ['None', 'CBAM', 'BAM', 'scSE']
        image_sizes = [48]  # Standard size for main comparison
        
        for model, attention, img_size in itertools.product(models, attention_types, image_sizes):
            config = Config.from_dict(base_config.to_dict())  # Deep copy
            
            # Update model configuration
            config.model.model = model
            config.model.attention_type = attention
            config.model.image_size = img_size
            
            # Update experiment name
            config.logging.wandb_run_name = f'{model}_{attention}_{img_size}x{img_size}'
            
            experiments.append(config)
        
        logger.info(f"Generated {len(experiments)} main comparison experiments")
        return experiments
    
    @staticmethod
    def generate_adaptive_architecture_experiments(base_config: Config) -> List[Config]:
        """Generate experiments for ResNet18 adaptive architecture study."""
        experiments = []
        
        attention_types = ['None', 'CBAM', 'BAM', 'scSE']
        image_sizes = [32, 48]  # Test adaptive capability
        
        for attention, img_size in itertools.product(attention_types, image_sizes):
            config = Config.from_dict(base_config.to_dict())  # Deep copy
            
            # Set to ResNet18 for adaptive testing
            config.model.model = 'resnet18'
            config.model.attention_type = attention
            config.model.image_size = img_size
            
            # Update experiment name
            config.logging.wandb_run_name = f'resnet18_{attention}_{img_size}x{img_size}_adaptive'
            
            experiments.append(config)
        
        logger.info(f"Generated {len(experiments)} adaptive architecture experiments")
        return experiments
    
    @staticmethod
    def generate_statistical_significance_experiments(base_config: Config) -> List[Config]:
        """Generate experiments for statistical significance testing."""
        experiments = []
        
        # Best configurations for statistical testing
        best_configs = [
            {'model': 'resnet50', 'attention_type': 'CBAM', 'image_size': 48},
            {'model': 'resnet18', 'attention_type': 'CBAM', 'image_size': 48},
            {'model': 'vgg16', 'attention_type': 'CBAM', 'image_size': 48}
        ]
        
        seeds = base_config.experiment.statistical_seeds
        
        for config_template in best_configs:
            for seed in seeds:
                config = Config.from_dict(base_config.to_dict())  # Deep copy
                
                # Update model configuration
                config.model.model = config_template['model']
                config.model.attention_type = config_template['attention_type']
                config.model.image_size = config_template['image_size']
                
                # Update training configuration for shorter runs
                config.training.seed = seed
                config.training.epochs = 50  # Shorter for statistical runs
                
                # Update experiment name
                config.logging.wandb_run_name = f"{config_template['model']}_{config_template['attention_type']}_seed{seed}"
                
                experiments.append(config)
        
        logger.info(f"Generated {len(experiments)} statistical significance experiments")
        return experiments
    
    @staticmethod
    def generate_ablation_study_experiments(base_config: Config) -> List[Config]:
        """Generate experiments for ablation studies."""
        experiments = []
        
        # Ablation study: Different optimizers
        optimizers = ['adam', 'sgd', 'adamax']
        for optimizer in optimizers:
            config = Config.from_dict(base_config.to_dict())
            config.training.optimizer = optimizer
            config.model.attention_type = 'CBAM'  # Fix attention for optimizer study
            config.logging.wandb_run_name = f'ablation_optimizer_{optimizer}'
            experiments.append(config)
        
        # Ablation study: Different learning rates
        learning_rates = [0.01, 0.001, 0.0001, 0.00001]
        for lr in learning_rates:
            config = Config.from_dict(base_config.to_dict())
            config.training.lr = lr
            config.model.attention_type = 'CBAM'  # Fix attention for LR study
            config.logging.wandb_run_name = f'ablation_lr_{lr}'
            experiments.append(config)
        
        logger.info(f"Generated {len(experiments)} ablation study experiments")
        return experiments


class ExperimentBatch:
    """Manages a batch of experiments."""
    
    def __init__(self, name: str, experiments: List[Config], results_dir: str):
        self.name = name
        self.experiments = experiments
        self.results_dir = Path(results_dir)
        self.batch_dir = self.results_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[ExperimentResult] = []
        self.current_experiment = 0
        
        logger.info(f"Created experiment batch '{name}' with {len(experiments)} experiments")
        logger.info(f"Results will be saved to: {self.batch_dir}")
    
    def save_batch_info(self) -> None:
        """Save batch information and configurations."""
        batch_info = {
            'name': self.name,
            'total_experiments': len(self.experiments),
            'created_at': datetime.now().isoformat(),
            'results_dir': str(self.batch_dir),
            'experiments': [exp.to_dict() for exp in self.experiments]
        }
        
        info_path = self.batch_dir / 'batch_info.json'
        with open(info_path, 'w') as f:
            json.dump(batch_info, f, indent=2)
        
        logger.info(f"Batch information saved to: {info_path}")
    
    def save_results(self) -> None:
        """Save current results to files."""
        # Save as JSON
        results_json = [result.to_dict() for result in self.results]
        json_path = self.batch_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save as CSV
        if self.results:
            df = pd.DataFrame([result.to_dict() for result in self.results])
            csv_path = self.batch_dir / 'results.csv'
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Results saved to: {json_path} and {csv_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get batch execution summary."""
        if not self.results:
            return {'status': 'no_results'}
        
        df = pd.DataFrame([result.to_dict() for result in self.results])
        
        summary = {
            'total_experiments': len(self.experiments),
            'completed_experiments': len(self.results),
            'success_count': len(df[df['status'] == 'success']),
            'failed_count': len(df[df['status'] == 'failed']),
            'timeout_count': len(df[df['status'] == 'timeout']),
            'average_duration': df['duration_seconds'].mean() if 'duration_seconds' in df else None,
            'best_accuracy': df['best_val_accuracy'].max() if 'best_val_accuracy' in df else None,
            'batch_dir': str(self.batch_dir)
        }
        
        if summary['success_count'] > 0:
            successful_df = df[df['status'] == 'success']
            best_idx = successful_df['best_val_accuracy'].idxmax()
            summary['best_experiment'] = successful_df.loc[best_idx, 'experiment_name']
        
        return summary


class ExperimentExecutor:
    """Executes individual experiments and collects results."""
    
    def __init__(self, timeout: int = 7200):
        self.timeout = timeout
        
    def execute_experiment(self, config: Config, experiment_dir: Path) -> ExperimentResult:
        """
        Execute a single experiment.
        
        Args:
            config: Experiment configuration
            experiment_dir: Directory to save experiment results
            
        Returns:
            ExperimentResult with execution details
        """
        experiment_name = config.get_experiment_name()
        start_time = datetime.now()
        
        logger.info(f"Executing experiment: {experiment_name}")
        
        # Create experiment directory
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment configuration
        config_path = experiment_dir / 'config.json'
        config.to_file(config_path)
        
        result = ExperimentResult(
            experiment_name=experiment_name,
            config=config.to_dict(),
            status='pending',
            start_time=start_time.isoformat()
        )
        
        try:
            # Build command for subprocess execution
            cmd = self._build_command(config, experiment_dir)
            logger.debug(f"Executing command: {cmd}")
            
            # Execute experiment
            process_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.getcwd()
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update result
            result.end_time = end_time.isoformat()
            result.duration_seconds = duration
            
            if process_result.returncode == 0:
                result.status = 'success'
                logger.info(f"✅ Experiment {experiment_name} completed successfully")
                
                # Extract metrics from experiment results
                self._extract_metrics(result, experiment_dir)
                
            else:
                result.status = 'failed'
                result.error_message = process_result.stderr
                logger.error(f"❌ Experiment {experiment_name} failed: {process_result.stderr}")
                
        except subprocess.TimeoutExpired:
            result.status = 'timeout'
            result.error_message = f"Experiment timed out after {self.timeout} seconds"
            result.end_time = datetime.now().isoformat()
            result.duration_seconds = self.timeout
            logger.warning(f"⏰ Experiment {experiment_name} timed out")
            
        except Exception as e:
            result.status = 'failed'
            result.error_message = str(e)
            result.end_time = datetime.now().isoformat()
            logger.error(f"💥 Experiment {experiment_name} crashed: {str(e)}")
        
        return result
    
    def _build_command(self, config: Config, experiment_dir: Path) -> str:
        """Build command line for experiment execution."""
        import sys
        
        cmd_parts = [
            sys.executable,  # Use current Python interpreter
            "train.py",
            "--experiment-mode", "single",  # Force single mode for subprocess
            "--model", config.model.model,
            "--attention-type", config.model.attention_type,
            "--image-size", str(config.model.image_size),
            "--image-channels", str(config.model.image_channels),
            "--num-classes", str(config.model.num_classes),
            "--epochs", str(config.training.epochs),
            "--batch-size", str(config.training.batch_size),
            "--lr", str(config.training.lr),
            "--optimizer", config.training.optimizer,
            "--lr-scheduler", config.training.lr_scheduler,
            "--early-stopping", str(config.training.early_stopping),
            "--seed", str(config.training.seed),
            "--train-folder", config.data.train_folder,
            "--valid-folder", config.data.valid_folder,
            "--color-mode", config.data.color_mode,
            "--class-mode", config.data.class_mode,
            "--result-path", str(experiment_dir),
            "--use-wandb", "1" if config.logging.use_wandb else "0",
            "--wandb-project-name", config.logging.wandb_project_name,
            "--wandb-run-name", config.logging.wandb_run_name or config.get_experiment_name()
        ]
        
        return " ".join(cmd_parts)
    
    def _extract_metrics(self, result: ExperimentResult, experiment_dir: Path) -> None:
        """Extract metrics from experiment results."""
        try:
            # Try to read CSV log
            csv_path = experiment_dir / 'log.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if not df.empty:
                    result.best_val_accuracy = df['val_accuracy'].max()
                    result.best_val_loss = df['val_loss'].min()
                    result.final_train_accuracy = df['accuracy'].iloc[-1]
                    result.final_train_loss = df['loss'].iloc[-1]
            
            # Try to read comprehensive metrics
            metrics_path = experiment_dir / 'comprehensive_metrics.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    model_info = metrics.get('model_info', {})
                    training_summary = metrics.get('training_summary', {})
                    
                    result.total_params = model_info.get('total_params')
                    result.training_time_minutes = training_summary.get('training_time_minutes')
                    
        except Exception as e:
            logger.warning(f"Could not extract metrics: {str(e)}")


class ExperimentManager:
    """High-level experiment management."""
    
    def __init__(self, results_dir: str = "./experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor = ExperimentExecutor()
        
    def run_experiment_batch(self, config: Config) -> ExperimentBatch:
        """
        Run a batch of experiments based on experiment mode.
        
        Args:
            config: Base configuration for experiments
            
        Returns:
            ExperimentBatch with results
        """
        # Generate experiments based on mode
        if config.experiment.experiment_mode == 'batch':
            experiments = self._generate_batch_experiments(config)
            batch_name = "Main_Comparison_Batch"
        elif config.experiment.experiment_mode == 'statistical':
            experiments = ExperimentGenerator.generate_statistical_significance_experiments(config)
            batch_name = "Statistical_Significance_Batch"
        else:
            raise ValueError(f"Unsupported batch experiment mode: {config.experiment.experiment_mode}")
        
        # Create batch
        batch = ExperimentBatch(batch_name, experiments, str(self.results_dir))
        batch.save_batch_info()
        
        # Execute experiments
        logger.info(f"🚀 Starting batch execution: {len(experiments)} experiments")
        
        for i, exp_config in enumerate(experiments, 1):
            logger.info(f"📊 Running experiment {i}/{len(experiments)}")
            
            # Create experiment directory
            exp_dir = batch.batch_dir / exp_config.get_experiment_name()
            
            # Execute experiment
            result = self.executor.execute_experiment(exp_config, exp_dir)
            batch.results.append(result)
            
            # Save intermediate results every 5 experiments
            if i % 5 == 0:
                batch.save_results()
                logger.info(f"💾 Intermediate results saved ({i}/{len(experiments)} completed)")
        
        # Save final results
        batch.save_results()
        
        # Print summary
        summary = batch.get_summary()
        self._print_batch_summary(summary)
        
        return batch
    
    def _generate_batch_experiments(self, config: Config) -> List[Config]:
        """Generate all experiments for batch mode."""
        experiments = []
        
        # Main comparison experiments
        experiments.extend(ExperimentGenerator.generate_main_comparison_experiments(config))
        
        # Adaptive architecture experiments
        experiments.extend(ExperimentGenerator.generate_adaptive_architecture_experiments(config))
        
        return experiments
    
    def _print_batch_summary(self, summary: Dict[str, Any]) -> None:
        """Print batch execution summary."""
        print("\n" + "="*60)
        print("BATCH EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Completed: {summary['completed_experiments']}")
        print(f"Successful: {summary['success_count']}")
        print(f"Failed: {summary['failed_count']}")
        print(f"Timeout: {summary['timeout_count']}")
        
        if summary.get('best_accuracy'):
            print(f"\nBest validation accuracy: {summary['best_accuracy']:.4f}")
            if summary.get('best_experiment'):
                print(f"Best experiment: {summary['best_experiment']}")
        
        if summary.get('average_duration'):
            print(f"Average duration: {summary['average_duration']/60:.1f} minutes")
        
        print(f"\nResults saved to: {summary['batch_dir']}")
        print("="*60)


if __name__ == "__main__":
    # Test experiment manager
    from config import Config
    
    # Create test configuration
    config = Config()
    config.experiment.experiment_mode = 'batch'
    config.training.epochs = 5  # Short for testing
    
    # Create experiment manager
    manager = ExperimentManager("./test_experiments")
    
    # Generate some test experiments
    experiments = ExperimentGenerator.generate_main_comparison_experiments(config)
    print(f"Generated {len(experiments)} test experiments")
    
    # Test batch creation
    batch = ExperimentBatch("Test_Batch", experiments[:2], "./test_experiments")
    batch.save_batch_info()
    print(f"Test batch created: {batch.batch_dir}")