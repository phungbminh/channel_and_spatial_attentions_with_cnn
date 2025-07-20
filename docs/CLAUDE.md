# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research-grade deep learning project that implements and compares different attention mechanisms (CBAM, BAM, scSE) with CNN architectures (ResNet50, ResNet18, VGG16) for facial emotion recognition. The project features a modular, professional architecture with comprehensive experiment management, statistical analysis capabilities, and research paper automation.

## Modular Architecture Components

### Core Modules (Research-Grade Structure)
- **Configuration Management**: `config.py` - Professional configuration system with dataclasses, validation, and serialization
- **Model Factory**: `model_factory.py` - Factory patterns for model creation, compilation, and optimization
- **Training Orchestrator**: `trainer.py` - Modular training system with data management, callbacks, and metrics
- **Experiment Manager**: `experiment_manager.py` - Systematic experiment execution, batch processing, and result collection
- **Main Entry Point**: `train.py` - Clean, professional main script with proper error handling and logging

### Model Implementations
- **ResNet variants**: `model_cnn.py` contains ResNet50 implementation
- **Adaptive ResNet**: `model_cnn_v2.py` contains ResNet18 with adaptive architecture for multiple input sizes
- **VGG16**: `model_cnn_v2.py` contains VGG16 implementation
- **Model Factory**: Unified interface for creating and configuring all models

### Attention Mechanisms
- **Attention modules**: `attentions_module.py` implements CBAM, BAM, and scSE attention blocks
- **Layer utilities**: `layers.py` and `layers_v2.py` provide building blocks and attention integration

### Research Infrastructure
- **Metrics Tracking**: `metrics_tracker.py` - Comprehensive metrics beyond basic accuracy/loss
- **Result Analysis**: `result_analyzer.py` - Statistical analysis and paper-ready output generation
- **Experiment Automation**: Full batch processing with timeout handling and error recovery

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate myenv

# Or install dependencies via pip
pip install -r requirement.txt
```

### Training Models

#### Single Experiment Mode
```bash
# Basic training with default settings (VGG16 with no attention)
python train.py

# Train ResNet50 with different attention mechanisms
python train.py --model resnet50 --attention-type CBAM --epochs 120 --seed 42
python train.py --model resnet50 --attention-type BAM --epochs 120 --seed 42
python train.py --model resnet50 --attention-type scSE --epochs 120 --seed 42

# Train ResNet18 with adaptive architecture (works with 32x32 and 48x48)
python train.py --model resnet18 --attention-type CBAM --image-size 32 --seed 42
python train.py --model resnet18 --attention-type CBAM --image-size 48 --seed 42

# Train with custom dataset paths and configuration
python train.py --train-folder /path/to/train --valid-folder /path/to/test --seed 42

# Configure training parameters with reproducibility
python train.py --batch-size 32 --lr 0.0001 --optimizer adamax --epochs 120 --seed 42
```

#### Batch Experiment Mode (For Paper Results)
```bash
# Run all model-attention combinations for comprehensive comparison
python train.py --experiment-mode batch --batch-results-dir ./paper_results

# Run statistical significance testing with multiple seeds
python train.py --experiment-mode statistical --statistical-seeds "42,123,456,789,999"

# Custom batch experiment with specific dataset
python train.py --experiment-mode batch --train-folder /path/to/train --valid-folder /path/to/test

# Use environment variable for Wandb API key (recommended)
export WANDB_API_KEY=your_api_key_here
python train.py --experiment-mode batch --use-wandb 1 --wandb-project-name "My_Paper_Experiments"
```

#### Result Analysis
```bash
# After running batch experiments, analyze results for paper
python result_analyzer.py ./paper_results/batch_YYYYMMDD_HHMMSS/

# This generates:
# - Performance comparison tables (CSV + LaTeX)
# - Statistical significance tests  
# - Paper-ready figures and plots
# - Comprehensive analysis report
```

### Key Training Arguments

#### Core Training Parameters
- `--model`: Choose from 'resnet50', 'resnet18', 'vgg16'
- `--attention-type`: Choose from 'None', 'CBAM', 'BAM', 'scSE'
- `--train-folder`: Path to training data directory
- `--valid-folder`: Path to validation data directory
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.0001)
- `--epochs`: Number of training epochs (default: 120)
- `--seed`: Random seed for reproducibility (default: 42)
- `--use-wandb`: Enable Wandb logging (1 for enable, 0 for disable)
- `--image-size`: Input image size - ResNet18 now supports both 32x32 and 48x48+
- `--optimizer`: Choose from 'adam', 'sgd', 'rmsprop', 'adadelta', 'adamax'
- `--lr-scheduler`: Learning rate scheduler ('ExponentialDecay', 'CosineDecay', 'None')

#### Experiment Mode Parameters
- `--experiment-mode`: Choose from 'single', 'batch', 'statistical'
- `--batch-results-dir`: Directory to save batch experiment results (default: './experiment_results')
- `--statistical-seeds`: Comma-separated seeds for statistical mode (default: '42,123,456,789,999')
- `--batch-timeout`: Timeout for each experiment in batch mode (default: 7200 seconds)

### GradCAM Visualization
The project includes GradCAM++ implementation in the `tf_keras_gradcamplusplus/` directory for visualizing attention maps.

## Dataset Structure
The project expects datasets in the following structure:
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

Supported datasets:
- FER-2013 (7 emotion classes)
- RAF-DB (basic emotions)

## Experiment Tracking
- Training logs are saved in `working/experiment/` with timestamps
- Wandb integration provides comprehensive experiment tracking
- Model checkpoints are saved with `.h5.keras` extension
- Class names are saved as `class_names.pkl` for inference

## Code Architecture Notes
- Models are built using functional API approach
- Attention mechanisms are modular and can be easily swapped
- The codebase supports both grayscale and RGB images
- Image preprocessing includes rescaling and data augmentation
- All models use sparse categorical crossentropy loss for multi-class classification

## Research Standards & Best Practices
- **Reproducibility**: Random seeds are set for TensorFlow, NumPy, and Python
- **Security**: API keys should be stored in environment variables, not hardcoded
- **Experiment Tracking**: Comprehensive logging with Wandb and CSV logs
- **Code Quality**: Standardized imports, proper error handling, and documentation
- **Adaptive Architecture**: ResNet18 automatically adapts to different input sizes (32x32 vs 48x48+)

## File Organization (Modular Structure)

### Core Research Framework
- **Main Entry**: `train.py` - Professional main script with proper error handling
- **Configuration**: `config.py` - Dataclass-based configuration with validation
- **Training**: `trainer.py` - Modular training orchestrator with data/callback management
- **Model Factory**: `model_factory.py` - Factory patterns for model creation and compilation
- **Experiment Management**: `experiment_manager.py` - Batch processing and result collection

### Model and Attention Components
- **ResNet50**: `model_cnn.py` - Standard ResNet50 implementation
- **Adaptive Models**: `model_cnn_v2.py` - ResNet18 with adaptive architecture, VGG16
- **Attention Modules**: `attentions_module.py` - CBAM, BAM, scSE implementations
- **Layer Utilities**: `layers.py`, `layers_v2.py` - Building blocks and attention integration

### Research Infrastructure
- **Metrics Tracking**: `metrics_tracker.py` - Comprehensive metrics beyond accuracy/loss
- **Result Analysis**: `result_analyzer.py` - Statistical analysis and paper-ready outputs
- **Experiment Guide**: `EXPERIMENT_GUIDE.md` - Detailed guide for running paper experiments

### Dependencies and Configuration
- **Python Dependencies**: `requirement.txt`
- **Conda Environment**: `environment.yml`
- **Project Documentation**: `CLAUDE.md` (this file)

### Results and Outputs
- **Single Experiments**: `working/experiment/` - Individual training results
- **Batch Experiments**: `experiment_results/` - Systematic experiment batches
- **GradCAM Tools**: `tf_keras_gradcamplusplus/` - Attention visualization

## Usage Patterns for Different Research Tasks

### Quick Single Experiment
```bash
python train.py --model resnet18 --attention-type CBAM --seed 42
```

### Comprehensive Paper Results
```bash
# Generate all experimental data for paper
python train.py --experiment-mode batch --batch-results-dir ./paper_results

# Run statistical significance tests
python train.py --experiment-mode statistical

# Analyze results for paper
python result_analyzer.py ./paper_results/batch_*/
```

### Development and Testing
```bash
# Quick test run with reduced epochs
python train.py --epochs 5 --batch-size 16

# Test specific configuration
python train.py --config-file my_config.json

# Save configuration for reuse
python train.py --model resnet50 --save-config my_config.json
```