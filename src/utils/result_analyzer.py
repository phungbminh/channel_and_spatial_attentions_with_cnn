"""
Result Analyzer for Paper Generation
Analyzes experiment results and generates paper-ready tables and figures
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import glob
from pathlib import Path

class ResultAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.df = None
        self.paper_dir = os.path.join(results_dir, 'paper_materials')
        os.makedirs(self.paper_dir, exist_ok=True)
        
        # Set style for paper figures
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_results(self, csv_path: str = None) -> pd.DataFrame:
        """Load experiment results from CSV file"""
        if csv_path is None:
            # Find the most recent results file
            pattern = os.path.join(self.results_dir, "**/experiment_results.csv")
            files = glob.glob(pattern, recursive=True)
            if not files:
                raise FileNotFoundError("No experiment results found")
            csv_path = max(files, key=os.path.getctime)
        
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} experiments from {csv_path}")
        
        # Clean and process data
        self.df = self.clean_data(self.df)
        return self.df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the results data"""
        # Remove failed experiments
        df = df[df['status'] == 'success'].copy()
        
        # Convert attention_type None to 'Baseline'
        df['attention_type'] = df['attention_type'].replace('None', 'Baseline')
        
        # Create model-attention combination
        df['model_attention'] = df['model'] + '_' + df['attention_type']
        
        # Convert accuracy to percentage
        df['accuracy_pct'] = df['best_val_accuracy'] * 100
        
        # Sort by model and attention type
        df = df.sort_values(['model', 'attention_type'])
        
        return df
    
    def generate_main_comparison_table(self) -> pd.DataFrame:
        """Generate main comparison table for the paper"""
        
        # Group by model and attention type
        grouped = self.df.groupby(['model', 'attention_type']).agg({
            'best_val_accuracy': ['mean', 'std', 'count'],
            'best_val_loss': ['mean', 'std'],
            'total_params': 'mean',
            'final_train_accuracy': 'mean'
        }).round(4)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
        
        # Create formatted table
        table_data = []
        for (model, attention), row in grouped.iterrows():
            table_data.append({
                'Model': model.upper(),
                'Attention': attention,
                'Val Accuracy (%)': f"{row['best_val_accuracy_mean']*100:.2f} ± {row['best_val_accuracy_std']*100:.2f}",
                'Val Loss': f"{row['best_val_loss_mean']:.4f} ± {row['best_val_loss_std']:.4f}",
                'Params (M)': f"{row['total_params_mean']/1e6:.2f}",
                'Runs': int(row['best_val_accuracy_count'])
            })
        
        table_df = pd.DataFrame(table_data)
        
        # Save to CSV
        table_path = os.path.join(self.paper_dir, 'main_comparison_table.csv')
        table_df.to_csv(table_path, index=False)
        
        # Save to LaTeX
        latex_path = os.path.join(self.paper_dir, 'main_comparison_table.tex')
        table_df.to_latex(latex_path, index=False, escape=False)
        
        print(f"Main comparison table saved to: {table_path}")
        return table_df
    
    def generate_statistical_significance_table(self) -> pd.DataFrame:
        """Generate statistical significance tests between methods"""
        
        # Get data for statistical tests
        models = self.df['model'].unique()
        attention_types = self.df['attention_type'].unique()
        
        significance_results = []
        
        for model in models:
            model_data = self.df[self.df['model'] == model]
            
            # Compare each attention type against baseline
            baseline_data = model_data[model_data['attention_type'] == 'Baseline']['best_val_accuracy']
            
            for attention in attention_types:
                if attention == 'Baseline':
                    continue
                    
                attention_data = model_data[model_data['attention_type'] == attention]['best_val_accuracy']
                
                if len(attention_data) > 1 and len(baseline_data) > 1:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(attention_data, baseline_data)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(attention_data) - 1) * attention_data.std()**2 + 
                                         (len(baseline_data) - 1) * baseline_data.std()**2) / 
                                        (len(attention_data) + len(baseline_data) - 2))
                    cohens_d = (attention_data.mean() - baseline_data.mean()) / pooled_std
                    
                    significance_results.append({
                        'Model': model.upper(),
                        'Attention': attention,
                        'Mean Diff (%)': f"{(attention_data.mean() - baseline_data.mean())*100:.2f}",
                        'p-value': f"{p_value:.4f}",
                        'Cohen\\'s d': f"{cohens_d:.3f}",
                        'Significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                    })
        
        sig_df = pd.DataFrame(significance_results)
        
        # Save results
        sig_path = os.path.join(self.paper_dir, 'statistical_significance.csv')
        sig_df.to_csv(sig_path, index=False)
        
        print(f"Statistical significance table saved to: {sig_path}")
        return sig_df
    
    def generate_performance_comparison_plot(self) -> None:
        """Generate performance comparison bar plot"""
        
        # Create grouped bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        accuracy_data = self.df.groupby(['model', 'attention_type'])['best_val_accuracy'].agg(['mean', 'std'])
        accuracy_data.reset_index(inplace=True)
        
        # Create pivot for easier plotting
        accuracy_pivot = accuracy_data.pivot(index='model', columns='attention_type', values='mean')
        accuracy_std = accuracy_data.pivot(index='model', columns='attention_type', values='std')
        
        accuracy_pivot.plot(kind='bar', ax=ax1, yerr=accuracy_std, capsize=4)
        ax1.set_title('Validation Accuracy by Model and Attention Type')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_xlabel('Model')
        ax1.legend(title='Attention Type')
        ax1.grid(axis='y', alpha=0.3)
        
        # Parameters comparison
        params_data = self.df.groupby(['model', 'attention_type'])['total_params'].mean()
        params_data.reset_index(inplace=True)
        params_pivot = params_data.pivot(index='model', columns='attention_type', values='total_params')
        
        (params_pivot / 1e6).plot(kind='bar', ax=ax2)
        ax2.set_title('Model Parameters by Architecture and Attention Type')
        ax2.set_ylabel('Parameters (Millions)')
        ax2.set_xlabel('Model')
        ax2.legend(title='Attention Type')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.paper_dir, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved to: {plot_path}")
    
    def generate_attention_effectiveness_plot(self) -> None:
        """Generate plot showing attention mechanism effectiveness"""
        
        # Calculate improvement over baseline for each model
        improvements = []
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            baseline_acc = model_data[model_data['attention_type'] == 'Baseline']['best_val_accuracy'].mean()
            
            for attention in ['CBAM', 'BAM', 'scSE']:
                attention_data = model_data[model_data['attention_type'] == attention]
                if not attention_data.empty:
                    attention_acc = attention_data['best_val_accuracy'].mean()
                    improvement = (attention_acc - baseline_acc) * 100
                    improvements.append({
                        'Model': model,
                        'Attention': attention,
                        'Improvement (%)': improvement
                    })
        
        improvements_df = pd.DataFrame(improvements)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot_data = improvements_df.pivot(index='Model', columns='Attention', values='Improvement (%)')
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Accuracy Improvement (%)'}, ax=ax)
        
        ax.set_title('Attention Mechanism Effectiveness Across Models')
        ax.set_xlabel('Attention Mechanism')
        ax.set_ylabel('Model Architecture')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.paper_dir, 'attention_effectiveness.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention effectiveness plot saved to: {plot_path}")
    
    def generate_model_complexity_analysis(self) -> None:
        """Generate model complexity vs performance analysis"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot
        for attention in self.df['attention_type'].unique():
            data = self.df[self.df['attention_type'] == attention]
            ax.scatter(data['total_params']/1e6, data['best_val_accuracy']*100, 
                      label=attention, s=100, alpha=0.7)
        
        ax.set_xlabel('Model Parameters (Millions)')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('Model Complexity vs Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add model labels
        for idx, row in self.df.iterrows():
            ax.annotate(row['model'], 
                       (row['total_params']/1e6, row['best_val_accuracy']*100),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.paper_dir, 'complexity_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Complexity vs performance plot saved to: {plot_path}")
    
    def generate_adaptive_architecture_analysis(self) -> None:
        """Analyze ResNet18 adaptive architecture performance"""
        
        # Filter ResNet18 data
        resnet18_data = self.df[self.df['model'] == 'resnet18'].copy()
        
        if resnet18_data.empty:
            print("No ResNet18 data found for adaptive analysis")
            return
        
        # Group by attention type and image size
        grouped = resnet18_data.groupby(['attention_type', 'image_size']).agg({
            'best_val_accuracy': ['mean', 'std'],
            'total_params': 'mean'
        }).round(4)
        
        # Create comparison table
        table_data = []
        for (attention, size), row in grouped.iterrows():
            table_data.append({
                'Attention': attention,
                'Image Size': f"{size}x{size}",
                'Accuracy (%)': f"{row[('best_val_accuracy', 'mean')]*100:.2f} ± {row[('best_val_accuracy', 'std')]*100:.2f}",
                'Parameters (M)': f"{row[('total_params', 'mean')]/1e6:.2f}"
            })
        
        adaptive_df = pd.DataFrame(table_data)
        
        # Save table
        table_path = os.path.join(self.paper_dir, 'adaptive_architecture_analysis.csv')
        adaptive_df.to_csv(table_path, index=False)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for attention in resnet18_data['attention_type'].unique():
            data = resnet18_data[resnet18_data['attention_type'] == attention]
            sizes = data['image_size'].values
            accuracies = data['best_val_accuracy'].values * 100
            
            ax.plot(sizes, accuracies, marker='o', label=attention, linewidth=2, markersize=8)
        
        ax.set_xlabel('Input Image Size')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.set_title('ResNet18 Adaptive Architecture Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([32, 48])
        ax.set_xticklabels(['32x32', '48x48'])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.paper_dir, 'adaptive_architecture.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Adaptive architecture analysis saved to: {table_path}")
        print(f"Adaptive architecture plot saved to: {plot_path}")
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive analysis report"""
        
        report_lines = []
        report_lines.append("# Attention Mechanisms in CNN Architectures - Experimental Results")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Dataset info
        report_lines.append("## Dataset Information")
        report_lines.append(f"- Total successful experiments: {len(self.df)}")
        report_lines.append(f"- Models tested: {', '.join(self.df['model'].unique())}")
        report_lines.append(f"- Attention mechanisms: {', '.join(self.df['attention_type'].unique())}")
        report_lines.append("")
        
        # Best results
        report_lines.append("## Best Results")
        best_result = self.df.loc[self.df['best_val_accuracy'].idxmax()]
        report_lines.append(f"- Best validation accuracy: {best_result['best_val_accuracy']:.4f} ({best_result['best_val_accuracy']*100:.2f}%)")
        report_lines.append(f"- Best configuration: {best_result['model']} + {best_result['attention_type']}")
        report_lines.append(f"- Image size: {best_result['image_size']}x{best_result['image_size']}")
        report_lines.append("")
        
        # Model comparison
        report_lines.append("## Model Performance Summary")
        model_summary = self.df.groupby('model')['best_val_accuracy'].agg(['mean', 'std', 'max'])
        for model, stats in model_summary.iterrows():
            report_lines.append(f"- {model.upper()}: {stats['mean']*100:.2f}% ± {stats['std']*100:.2f}% (max: {stats['max']*100:.2f}%)")
        report_lines.append("")
        
        # Attention effectiveness
        report_lines.append("## Attention Mechanism Effectiveness")
        attention_summary = self.df.groupby('attention_type')['best_val_accuracy'].agg(['mean', 'std'])
        for attention, stats in attention_summary.iterrows():
            report_lines.append(f"- {attention}: {stats['mean']*100:.2f}% ± {stats['std']*100:.2f}%")
        report_lines.append("")
        
        # Save report
        report_path = os.path.join(self.paper_dir, 'comprehensive_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Comprehensive report saved to: {report_path}")
    
    def generate_all_paper_materials(self) -> None:
        """Generate all paper materials"""
        
        print("🎯 Generating paper materials...")
        
        # Tables
        print("\n📊 Generating tables...")
        self.generate_main_comparison_table()
        self.generate_statistical_significance_table()
        
        # Plots
        print("\n📈 Generating plots...")
        self.generate_performance_comparison_plot()
        self.generate_attention_effectiveness_plot()
        self.generate_model_complexity_analysis()
        self.generate_adaptive_architecture_analysis()
        
        # Report
        print("\n📝 Generating report...")
        self.generate_comprehensive_report()
        
        print(f"\n✅ All paper materials generated in: {self.paper_dir}")

def main():
    """Main function for result analysis"""
    
    # Initialize analyzer
    analyzer = ResultAnalyzer('./experiment_results')
    
    # Load results
    analyzer.load_results()
    
    # Generate all materials
    analyzer.generate_all_paper_materials()

if __name__ == "__main__":
    main()