# FL-DomainNet with FAP-Float(S) and EAPH-LoRA

Federated Learning system for DomainNet's 6 visual domains with fairness-aware aggregation point selection and domain-specific personalization.

## Features

- **FAP-Float(S)**: Fairness-Aware floating aggregation Point selector with Softmax sampling
- **EAPH-LoRA**: Edge-Assisted Personalized Hierarchical LoRA for domain-specific adaptation
- **Multi-domain FL**: Support for 6 visual domains (clipart, infograph, painting, quickdraw, real, sketch)
- **Data heterogeneity handling**: Dirichlet-based client partitioning with α=0.1
- **Prototype-based drift monitoring**: 64-dimensional feature prototypes for distribution shift detection

## Quick Start

See [quickstart.md](quickstart.md) for detailed setup instructions.

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (16GB+ VRAM recommended)
- DomainNet dataset

### Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv sync
```

### Running the Experiment

```bash
# Default configuration (200 rounds, 6 domains, 144 clients)
python run_experiment.py --config configs/default.yaml

# Or use the convenience script
bash scripts/run_default.sh

# Quick validation (2 domains, 3 rounds)
bash scripts/validate_quickstart.sh
```

### Configuration Options

```bash
# Override specific settings
python run_experiment.py \
    --config configs/default.yaml \
    --domains clipart real \
    --rounds 50 \
    --device cuda

# Custom experiment tag
python run_experiment.py \
    --config configs/default.yaml \
    --exp-tag my-experiment
```

### Monitoring

- Training logs: `outputs/exp1/train.log`
- Checkpoints: `outputs/exp1/checkpoints/`
- Metrics tracked:
  - Average accuracy across domains
  - Worst-domain accuracy
  - Variance of domain accuracies
  - Per-domain loss and drift

## Project Structure

```
├── configs/          # Hyperparameter configurations
├── data/            # Data loading and partitioning
├── models/          # Model architectures (ResNet18 + LoRA)
├── core/            # Core FL components
│   ├── trainer.py   # Local training
│   ├── aggregator.py # FedAvg aggregation
│   ├── edge_manager.py # Domain state management
│   ├── selector.py  # FAP-Float(S) selector
│   └── loop.py      # Main FL orchestration
├── utils/           # Utilities (metrics, transforms, logging)
├── tests/           # Unit and integration tests
└── scripts/         # Helper scripts
```

## Key Hyperparameters

- `K=5`: Aggregation point selection period
- `r=16`: LoRA rank
- `α=0.1`: Dirichlet concentration (high heterogeneity)
- `ρ=0.2`: Data offload ratio to edge servers
- `τ=1.0`: Softmax temperature for selection

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov=data --cov=models --cov-report=term

# Run specific test modules
pytest tests/test_selector.py -v
pytest tests/test_edge_manager.py -v
pytest tests/test_integration.py -v
```

## Visualization

```bash
# Generate all visualizations from experiment log
python scripts/visualize_results.py \
    --log outputs/my-experiment/train.log \
    --output-dir outputs/my-experiment/plots

# Visualize drift heatmap only
python scripts/visualize_drift.py \
    --log outputs/my-experiment/train.log \
    --output outputs/my-experiment/drift.png
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce `batch_size` in config (try 16 or 8)
- Reduce number of domains or clients
- Use smaller LoRA rank

**Dataset Not Found**
- Ensure DomainNet is downloaded and extracted
- Update `data.root` in config to correct path
- Run `scripts/prepare_domainnet.sh` if needed

**Slow Training**
- Enable `pretrained: true` for faster convergence
- Reduce `total_rounds` for initial testing
- Use `--device cuda` if available

**Import Errors**
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Reinstall dependencies: `uv sync`

### Getting Help

For detailed setup and usage, see [quickstart.md](specs/001-fl-domainnet-fap-lora/quickstart.md)

## Citation

Based on research combining FAP-Float(S) fairness-aware selection with EAPH-LoRA personalization for federated learning in heterogeneous environments.

## License

Research prototype for academic use.