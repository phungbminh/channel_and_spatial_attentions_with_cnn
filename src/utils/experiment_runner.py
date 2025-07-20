"""
Experiment Runner for Attention Mechanisms Research
Systematically runs experiments and collects results for paper
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import itertools
import time

class ExperimentRunner:
    def __init__(self, base_results_dir: str = "./experiment_results"):
        self.base_results_dir = base_results_dir
        self.experiment_log = []
        self.results_summary = []
        
        # Create results directory
        os.makedirs(base_results_dir, exist_ok=True)
        
        # Current timestamp for this experiment batch
        self.batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir = os.path.join(base_results_dir, f"batch_{self.batch_timestamp}")
        os.makedirs(self.batch_dir, exist_ok=True)
        
        print(f"Experiment batch directory: {self.batch_dir}")
    
    def define_experiment_configs(self) -> List[Dict]:
        """Define all experiment configurations for paper"""
        
        # Base configuration
        base_config = {
            'train_folder': './dataset/FER-2013/train',
            'valid_folder': './dataset/FER-2013/test',
            'epochs': 120,
            'batch_size': 32,
            'lr': 0.0001,
            'optimizer': 'adamax',
            'lr_scheduler': 'ExponentialDecay',
            'color_mode': 'grayscale',
            'image_channels': 1,
            'num_classes': 7,
            'early_stopping': 15,
            'use_wandb': 1,
            'wandb_project_name': 'Attention_CNN_Research_Paper'
        }
        
        # Experiment configurations
        experiments = []
        
        # 1. Main comparison: Different models with different attention mechanisms
        models = ['resnet50', 'resnet18', 'vgg16']
        attention_types = ['None', 'CBAM', 'BAM', 'scSE']
        image_sizes = [48]  # Standard size
        
        for model, attention, img_size in itertools.product(models, attention_types, image_sizes):
            config = base_config.copy()
            config.update({
                'model': model,
                'attention_type': attention,
                'image_size': img_size,
                'experiment_name': f'{model}_{attention}_{img_size}x{img_size}',
                'seed': 42  # Fixed seed for reproducibility
            })
            experiments.append(config)
        
        # 2. ResNet18 adaptive architecture comparison (32x32 vs 48x48)
        for attention in attention_types:
            for img_size in [32, 48]:
                config = base_config.copy()
                config.update({
                    'model': 'resnet18',
                    'attention_type': attention,
                    'image_size': img_size,
                    'experiment_name': f'resnet18_{attention}_{img_size}x{img_size}_adaptive',
                    'seed': 42
                })
                experiments.append(config)
        
        # 3. Multiple seeds for statistical significance
        best_configs = [
            {'model': 'resnet50', 'attention_type': 'CBAM', 'image_size': 48},
            {'model': 'resnet18', 'attention_type': 'CBAM', 'image_size': 48},
            {'model': 'vgg16', 'attention_type': 'CBAM', 'image_size': 48}
        ]
        
        seeds = [42, 123, 456, 789, 999]  # 5 different seeds
        for config_template in best_configs:
            for seed in seeds:
                config = base_config.copy()
                config.update(config_template)
                config.update({
                    'seed': seed,
                    'experiment_name': f"{config_template['model']}_{config_template['attention_type']}_seed{seed}",
                    'epochs': 50  # Shorter for statistical runs
                })
                experiments.append(config)
        
        print(f"Total experiments defined: {len(experiments)}")
        return experiments
    
    def run_single_experiment(self, config: Dict) -> Dict:
        """Run a single experiment and collect results"""
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {config['experiment_name']}")
        print(f"{'='*60}")
        
        # Create experiment directory
        exp_dir = os.path.join(self.batch_dir, config['experiment_name'])
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save experiment config
        config_path = os.path.join(exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Build command
        cmd = self.build_command(config, exp_dir)
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✅ Experiment completed successfully")
                status = "success"
                error_msg = None
            else:
                print(f"❌ Experiment failed with return code: {result.returncode}")
                print(f"Error: {result.stderr}")
                status = "failed"
                error_msg = result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Experiment timed out after 2 hours")
            status = "timeout"
            error_msg = "Experiment timed out"
            end_time = time.time()
        
        # Extract results
        results = self.extract_results(exp_dir, config)
        results.update({
            'status': status,
            'error_msg': error_msg,
            'duration_seconds': end_time - start_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log experiment
        self.experiment_log.append({
            'config': config,
            'results': results,
            'cmd': cmd
        })
        
        return results
    
    def build_command(self, config: Dict, exp_dir: str) -> str:
        """Build training command from config"""
        
        cmd_parts = [
            f"python train.py",
            f"--model {config['model']}",
            f"--attention-type {config['attention_type']}",
            f"--train-folder {config['train_folder']}",
            f"--valid-folder {config['valid_folder']}",
            f"--epochs {config['epochs']}",
            f"--batch-size {config['batch_size']}",
            f"--lr {config['lr']}",
            f"--optimizer {config['optimizer']}",
            f"--lr-scheduler {config['lr_scheduler']}",
            f"--image-size {config['image_size']}",
            f"--image-channels {config['image_channels']}",
            f"--num-classes {config['num_classes']}",
            f"--color-mode {config['color_mode']}",
            f"--early-stopping {config['early_stopping']}",
            f"--seed {config['seed']}",
            f"--result-path {exp_dir}",
            f"--use-wandb {config['use_wandb']}",
            f"--wandb-project-name {config['wandb_project_name']}",
            f"--wandb-run-name {config['experiment_name']}"
        ]
        
        return " ".join(cmd_parts)
    
    def extract_results(self, exp_dir: str, config: Dict) -> Dict:
        """Extract results from experiment directory"""
        
        results = {
            'experiment_name': config['experiment_name'],
            'model': config['model'],
            'attention_type': config['attention_type'],
            'image_size': config['image_size'],
            'seed': config['seed'],
            'best_val_accuracy': None,
            'best_val_loss': None,
            'final_train_accuracy': None,
            'final_train_loss': None,
            'total_params': None,
            'training_time': None
        }
        
        # Try to read CSV log
        csv_files = [f for f in os.listdir(exp_dir) if f.endswith('.csv')]
        if csv_files:
            try:
                log_path = os.path.join(exp_dir, csv_files[0])
                df = pd.read_csv(log_path)
                
                if not df.empty:
                    results['best_val_accuracy'] = df['val_accuracy'].max()
                    results['best_val_loss'] = df['val_loss'].min()
                    results['final_train_accuracy'] = df['accuracy'].iloc[-1]
                    results['final_train_loss'] = df['loss'].iloc[-1]
                    results['total_epochs'] = len(df)
                    
            except Exception as e:
                print(f"Warning: Could not read CSV log: {e}")
        
        return results
    
    def run_all_experiments(self, configs: List[Dict] = None) -> pd.DataFrame:
        """Run all experiments and return results DataFrame"""
        
        if configs is None:
            configs = self.define_experiment_configs()
        
        print(f"\n🚀 Starting {len(configs)} experiments...")
        print(f"Results will be saved to: {self.batch_dir}")
        
        all_results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n📊 Progress: {i}/{len(configs)}")
            results = self.run_single_experiment(config)
            all_results.append(results)
            
            # Save intermediate results
            if i % 5 == 0:  # Save every 5 experiments
                self.save_results(all_results)
        
        # Save final results
        df_results = self.save_results(all_results)
        
        print(f"\n🎉 All experiments completed!")
        print(f"Results saved to: {self.batch_dir}")
        
        return df_results
    
    def save_results(self, results: List[Dict]) -> pd.DataFrame:
        """Save results to CSV and JSON"""
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        csv_path = os.path.join(self.batch_dir, 'experiment_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Save to JSON
        json_path = os.path.join(self.batch_dir, 'experiment_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save experiment log
        log_path = os.path.join(self.batch_dir, 'experiment_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        
        print(f"Results saved to: {csv_path}")
        return df

def main():
    """Main function to run experiments"""
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Define and run experiments
    configs = runner.define_experiment_configs()
    
    # Optional: Run subset for testing
    # configs = configs[:3]  # Run only first 3 experiments for testing
    
    # Run all experiments
    results_df = runner.run_all_experiments(configs)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT BATCH SUMMARY")
    print("="*60)
    
    successful = results_df[results_df['status'] == 'success']
    failed = results_df[results_df['status'] == 'failed']
    timeout = results_df[results_df['status'] == 'timeout']
    
    print(f"Total experiments: {len(results_df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Timeout: {len(timeout)}")
    
    if len(successful) > 0:
        print(f"\nBest validation accuracy: {successful['best_val_accuracy'].max():.4f}")
        best_exp = successful.loc[successful['best_val_accuracy'].idxmax()]
        print(f"Best experiment: {best_exp['experiment_name']}")
    
    print(f"\nResults saved to: {runner.batch_dir}")

if __name__ == "__main__":
    main()