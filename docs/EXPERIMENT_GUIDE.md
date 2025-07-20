# Experiment Guide for Paper Results

This guide helps you run systematic experiments to collect results for your research paper.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirement.txt

# Or use conda
conda env create -f environment.yml
conda activate myenv

# Install additional dependencies for experiments
pip install psutil seaborn scikit-learn scipy
```

### 2. Prepare Dataset
Ensure your dataset is structured correctly:
```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

### 3. Run All Experiments

#### Option A: Integrated Approach (Recommended)
```bash
# Run comprehensive experiments for paper (all combinations)
python train.py --experiment-mode batch --batch-results-dir ./paper_results

# Run statistical significance testing
python train.py --experiment-mode statistical --statistical-seeds "42,123,456,789,999"

# This will run:
# - All model-attention combinations
# - ResNet18 adaptive architecture tests  
# - Multiple seeds for statistical significance
```

#### Option B: Standalone Experiment Runner
```bash
# Alternative: Use separate experiment runner
python experiment_runner.py

# Same functionality as integrated approach
```

### 4. Analyze Results
```bash
# Generate paper materials
python result_analyzer.py

# This creates:
# - Tables in CSV and LaTeX format
# - Statistical significance tests
# - Performance comparison plots
# - Comprehensive report
```

## 📊 Experiment Types

### Main Comparison Experiments
Tests all combinations of:
- **Models**: ResNet50, ResNet18, VGG16
- **Attention**: None (Baseline), CBAM, BAM, scSE
- **Image Size**: 48x48 (standard)

### Adaptive Architecture Tests
Tests ResNet18 with:
- **Image Sizes**: 32x32, 48x48
- **All Attention Types**: None, CBAM, BAM, scSE

### Statistical Significance Tests
Runs each best configuration with:
- **Multiple Seeds**: 42, 123, 456, 789, 999
- **Shorter Epochs**: 50 (for efficiency)

## 🎯 Individual Experiment Examples

### Single Model Training
```bash
# Train ResNet50 with CBAM attention
python train.py --model resnet50 --attention-type CBAM --seed 42

# Train ResNet18 with adaptive architecture (32x32)
python train.py --model resnet18 --attention-type BAM --image-size 32 --seed 42

# Train VGG16 baseline
python train.py --model vgg16 --attention-type None --seed 42
```

### Custom Experiments
```bash
# Custom dataset paths
python train.py --train-folder /path/to/train --valid-folder /path/to/test

# Custom hyperparameters
python train.py --lr 0.001 --batch-size 64 --epochs 100 --optimizer adam

# Use Wandb for tracking
export WANDB_API_KEY=your_api_key_here
python train.py --use-wandb 1 --wandb-project-name "My_Paper_Experiments"
```

## 📈 Output Structure

### Experiment Results
```
experiment_results/
└── batch_20240117_143022/
    ├── experiment_results.csv           # Main results table
    ├── experiment_results.json          # Detailed results
    ├── experiment_log.json              # Execution log
    └── individual_experiments/
        ├── resnet50_CBAM_48x48/
        │   ├── config.json              # Experiment config
        │   ├── log.csv                  # Training log
        │   ├── comprehensive_metrics.json # Detailed metrics
        │   ├── confusion_matrix.png     # Confusion matrix
        │   ├── classification_report.txt # Classification report
        │   └── model_architecture.txt   # Model summary
        └── ...
```

### Paper Materials
```
experiment_results/batch_*/paper_materials/
├── main_comparison_table.csv           # Main results table
├── main_comparison_table.tex           # LaTeX table
├── statistical_significance.csv        # Statistical tests
├── performance_comparison.png          # Performance plots
├── attention_effectiveness.png         # Attention heatmap
├── complexity_performance.png          # Parameters vs accuracy
├── adaptive_architecture.png           # ResNet18 adaptive analysis
└── comprehensive_report.md             # Full analysis report
```

## 🔧 Configuration Options

### Training Parameters
```python
# Core training settings
--model: resnet50, resnet18, vgg16
--attention-type: None, CBAM, BAM, scSE
--image-size: 32, 48, 64 (ResNet18 adaptive)
--epochs: 120 (default)
--batch-size: 32 (default)
--lr: 0.0001 (default)
--optimizer: adam, sgd, rmsprop, adadelta, adamax
--seed: 42 (for reproducibility)
```

### Experiment Tracking
```python
# Wandb integration
--use-wandb: 1 (enable) or 0 (disable)
--wandb-project-name: "Your_Project_Name"
--wandb-run-name: "experiment_name"

# Local logging
--result-path: "./working" (default)
--early-stopping: 15 (patience)
```

## 📋 Metrics Collected

### Basic Metrics
- Validation accuracy and loss
- Training accuracy and loss
- Model parameters count
- Training time

### Advanced Metrics
- Precision, Recall, F1-score (per class and weighted)
- Confusion matrix
- Classification report
- Memory usage during training
- GPU utilization
- Learning rate schedule

### Statistical Analysis
- Mean and standard deviation across runs
- Statistical significance tests (t-tests)
- Effect size (Cohen's d)
- Confidence intervals

## 🎨 Paper-Ready Outputs

### Tables
- **Main Comparison Table**: All model-attention combinations
- **Statistical Significance**: p-values and effect sizes
- **Adaptive Architecture**: ResNet18 size comparison

### Figures
- **Performance Comparison**: Bar charts with error bars
- **Attention Effectiveness**: Heatmap showing improvements
- **Model Complexity**: Parameters vs accuracy scatter plot
- **Adaptive Architecture**: Line plot for different input sizes

### Reports
- **Comprehensive Report**: Markdown summary of all results
- **Best Configurations**: Top performing setups
- **Statistical Summary**: Significance test results

## 🔍 Tips for Paper Writing

### Result Interpretation
1. **Baseline Comparison**: Always compare against "None" attention
2. **Statistical Significance**: Use p-values and effect sizes
3. **Practical Significance**: Consider accuracy improvements vs complexity
4. **Consistency**: Report means ± standard deviations

### Common Findings to Report
- Which attention mechanism works best overall
- Model-specific attention preferences
- Parameter overhead vs performance gains
- Adaptive architecture benefits
- Statistical significance of improvements

### LaTeX Integration
Tables are automatically generated in LaTeX format:
```latex
\input{main_comparison_table.tex}
```

Figures can be included as:
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{performance_comparison.png}
    \caption{Performance comparison across models and attention mechanisms}
    \label{fig:performance}
\end{figure}
```

## 🚦 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or image size
2. **Long Training Time**: Reduce epochs or use early stopping
3. **Poor Results**: Check dataset quality and preprocessing
4. **Missing Dependencies**: Install all required packages

### Performance Optimization
```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Reduce memory usage
python train.py --batch-size 16 --image-size 32

# Faster experiments
python train.py --epochs 50 --early-stopping 10
```

## 📞 Support

If you encounter issues:
1. Check the comprehensive report for experiment status
2. Review individual experiment logs
3. Verify dataset structure and paths
4. Ensure all dependencies are installed

## 🎯 Next Steps

After collecting results:
1. Review the comprehensive report
2. Analyze statistical significance
3. Create additional visualizations if needed
4. Write your paper sections based on findings
5. Include generated tables and figures

---

**Happy Experimenting! 🧪**