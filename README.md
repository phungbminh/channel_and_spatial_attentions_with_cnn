# The Effectiveness of Channel and Spatial Attention for Improving Image Classification

**🎯 Conference Presentation Research**

A comprehensive research implementation demonstrating the effectiveness of Channel and Spatial Attention mechanisms (CBAM, BAM, scSE) integrated with CNN architectures (ResNet18, VGG16) for enhanced image classification performance.

## <� Project Structure

```
   src/                          # Source code (modular architecture)
      __init__1.py              # Main package initialization
      config/                  # Configuration management
         __init__.py
         config.py            # Professional config with dataclasses
      models/                  # Model implementations
         __init__.py
         model_factory.py     # Factory for model creation
         model_cnn.py         # ResNet50 implementation
         model_cnn_v2.py      # ResNet18 (adaptive) + VGG16
         vgg.py               # VGG16 alternative implementation
      attention/               # Attention mechanisms
         __init__.py
         attentions_module.py # CBAM, BAM, scSE implementations
         attention_modules_v2.py
      layers/                  # Layer utilities
         __init__.py
         layers.py            # Building blocks and utilities
         layers_v2.py         # Alternative layer implementations
      training/                # Training orchestration
         __init__.py
         trainer.py           # Modular training system
      experiments/             # Experiment management
         __init__.py
         experiment_manager.py # Batch processing and result collection
      utils/                   # Utilities and analysis
          __init__.py
          metrics_tracker.py   # Comprehensive metrics tracking
          result_analyzer.py   # Statistical analysis for papers
          experiment_runner.py # Standalone experiment runner
   data/                        # Data directory
      dataset/                 # Training and test datasets
   results/                     # Results and outputs
      working/                 # Individual experiment results
      wandb/                   # WandB experiment logs
   docs/                        # Documentation
      CLAUDE.md               # Claude Code guidance
      EXPERIMENT_GUIDE.md     # Detailed experiment guide
   scripts/                     # Additional scripts
      gradcam/                # GradCAM visualization tools
   tests/                       # Unit tests (future)
   train.py                     # Main entry point
   requirement.txt              # Python dependencies
   environment.yml              # Conda environment
   README.md                    # This file
```

## =� Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd channel_and_spatial_attentions_with_cnn

# Create environment
conda env create -f environment.yml
conda activate myenv

# Or use pip
pip install -r requirement.txt
```

### Basic Usage

```bash
# Single experiment
python train.py --model resnet18 --attention-type CBAM --seed 42

# Batch experiments for paper
python train.py --experiment-mode batch --batch-results-dir ./paper_results

# Statistical significance testing
python train.py --experiment-mode statistical --statistical-seeds "42,123,456,789,999"
```

## =� Features

### Research-Grade Architecture
- **Modular Design**: Clean separation of concerns
- **Factory Patterns**: Unified model and experiment creation
- **Professional Configuration**: Dataclass-based with validation
- **Type Safety**: Comprehensive type hints throughout

### Experiment Management
- **Batch Processing**: Automated experiment batches
- **Statistical Testing**: Multiple seeds for significance testing
- **Result Analysis**: Paper-ready tables and figures
- **Reproducibility**: Comprehensive seed management

### Model Support
- **CNN Architectures**: ResNet50, ResNet18, VGG16
- **Attention Mechanisms**: CBAM, BAM, scSE
- **Adaptive Architecture**: ResNet18 supports 32x32 and 48x48+ inputs
- **Unified Interface**: Factory pattern for all models

### Tracking & Analysis
- **WandB Integration**: Professional experiment tracking
- **Comprehensive Metrics**: Beyond accuracy/loss
- **Statistical Analysis**: p-values, effect sizes, confidence intervals
- **Paper Automation**: LaTeX tables and publication-ready figures

## =' Configuration

The framework uses a professional configuration system with dataclasses:

```python
from src.config.config import Config, ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(
        model='resnet18',
        attention_type='CBAM',
        image_size=48
    ),
    training=TrainingConfig(
        epochs=120,
        batch_size=32,
        lr=0.0001,
        optimizer='adamax'
    )
)
```

## =� Experiment Modes

### Single Mode (Default)
```bash
python train.py --model resnet50 --attention-type CBAM --epochs 120
```

### Batch Mode (All Combinations)
```bash
python train.py --experiment-mode batch
# Tests all model-attention combinations with adaptive architecture
```

### Statistical Mode (Multiple Seeds)
```bash
python train.py --experiment-mode statistical
# Runs best configurations with multiple seeds for statistical significance
```

## =� Research Usage

### For Paper Results
1. **Run comprehensive experiments**:
   ```bash
   python train.py --experiment-mode batch --batch-results-dir ./paper_results
   ```

2. **Run statistical significance tests**:
   ```bash
   python train.py --experiment-mode statistical
   ```

3. **Analyze results**:
   ```bash
   python src/utils/result_analyzer.py ./paper_results/batch_*/
   ```

### Generated Outputs
- **Tables**: CSV and LaTeX format for papers
- **Figures**: Performance plots with error bars
- **Statistics**: p-values, effect sizes, confidence intervals
- **Reports**: Comprehensive analysis summaries

## =, Architecture Details

### Modular Components
- **Config System**: Professional configuration with validation
- **Model Factory**: Unified interface for all architectures
- **Training Orchestrator**: Modular training with callback management
- **Experiment Manager**: Systematic batch processing
- **Result Analyzer**: Statistical analysis and paper outputs

### Key Features
- **Adaptive ResNet18**: Automatically handles different input sizes
- **Attention Integration**: Seamless attention mechanism integration
- **Reproducibility**: Comprehensive seed management for TensorFlow/NumPy
- **Error Handling**: Professional error handling and recovery
- **Logging**: Structured logging throughout the framework

## <� Supported Models

| Model | Attention | Input Sizes | Notes |
|-------|-----------|-------------|-------|
| ResNet50 | CBAM, BAM, scSE, None | 48x48+ | Standard implementation |
| ResNet18 | CBAM, BAM, scSE, None | 32x32, 48x48+ | Adaptive architecture |
| VGG16 | CBAM, BAM, scSE, None | 48x48+ | Global average pooling |

## =� Dataset Support

- **FER-2013**: 7 emotion classes
- **RAF-DB**: Basic emotions
- **Custom datasets**: Following standard folder structure

Expected structure:
```
dataset/
   train/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
   test/
       angry/
       disgust/
       fear/
       happy/
       neutral/
       sad/
       surprise/
```

## =� Development

### Adding New Models
1. Implement in `src/models/`
2. Register in `ModelFactory`
3. Update configuration options

### Adding New Attention Mechanisms
1. Implement in `src/attention/`
2. Register in layer selection functions
3. Update model factory

### Running Tests
```bash
# Unit tests (future)
pytest tests/

# Integration test
python train.py --epochs 1 --batch-size 8
```

## =� Documentation

- **[CLAUDE.md](docs/CLAUDE.md)**: Guidance for Claude Code integration
- **[EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)**: Detailed experiment guide
- **Code Documentation**: Comprehensive docstrings throughout

## > Contributing

1. Follow the modular architecture patterns
2. Add comprehensive type hints
3. Include docstrings for all functions/classes
4. Update configuration system for new parameters
5. Maintain compatibility with existing experiments

## =� License

[Add your license information here]

## = Citation

If you use this framework in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

---

**Built with research excellence in mind** >�