# Baseline FL Implementations

This directory contains baseline federated learning implementations (FedAvg, FedProx) for comparison with the main FAP-Float + EAPH-LoRA method.

## Overview

The baseline implementations use:
- **Single-head global model** (no domain personalization)
- **Fixed aggregation point** (no dynamic selection)
- **Standard FedAvg/FedProx** algorithms

This provides a fair comparison baseline to demonstrate the advantages of:
- Domain-specific adaptation (EAPH-LoRA)
- Dynamic aggregation point selection (FAP-Float)

## Directory Structure

```
baseline/
├── __init__.py
├── README.md
├── models/
│   ├── __init__.py
│   └── resnet18_single.py    # Single-head ResNet-18 (no LoRA, no domain heads)
└── core/
    ├── __init__.py
    ├── trainer_fedavg.py     # FedAvg local trainer (CE loss)
    ├── trainer_fedprox.py    # FedProx local trainer (CE + proximal term)
    ├── selector_fixed.py     # Fixed aggregator selector
    └── loop_baseline.py      # Baseline FL training loop
```

## Algorithms

### FedAvg-Fixed

Standard Federated Averaging with:
- Cross-entropy loss
- Sample-weighted aggregation
- Fixed aggregation point (default: "real" domain)

$$
\begin{aligned}
Loss &= CrossEntropy(y, ŷ) \\
w_{global} &= \sum_{i=1}^{n} \frac{n_i}{N} \cdot w_i
\end{aligned}
$$

### FedProx-Fixed

FedProx extends FedAvg with a proximal term:
- Adds regularization toward global model
- Helps with heterogeneous data


$$Loss = CrossEntropy(y, ŷ) + \frac{\mu}{2} ||w - w_t||²$$


Default: μ = 0.01

## Usage

### Running FedAvg Baseline

```bash
python run_baseline.py --config configs/fedavg_fixed.yaml
```

### Running FedProx Baseline

```bash
python run_baseline.py --config configs/fedprox_fixed.yaml
```

### Command Line Options

```bash
python run_baseline.py \
    --config configs/fedavg_fixed.yaml \
    --algo fedprox \              # Override algorithm
    --mu 0.05 \                   # Override proximal coefficient
    --alpha 0.5 \                 # Override Dirichlet alpha
    --rounds 100 \                # Override number of rounds
    --fixed-domain quickdraw \    # Override fixed aggregator domain
    --exp-tag my-experiment       # Add experiment tag
```

## Configuration

### FedAvg Configuration (`configs/fedavg_fixed.yaml`)

```yaml
algo: "fedavg"
selector:
  type: "fixed"
  fixed_domain: "real"
```

### FedProx Configuration (`configs/fedprox_fixed.yaml`)

```yaml
algo: "fedprox"
prox:
  mu: 0.01  # Proximal coefficient
selector:
  type: "fixed"
  fixed_domain: "real"
```

## Key Differences from Main Method

| Component | Main Method | Baseline |
|-----------|-------------|----------|
| Model | ResNet18_EAPH (LoRA + domain heads) | ResNet18Single (single head) |
| Aggregator Selection | FAP-Float(S) (dynamic) | FixedSelector (static) |
| Personalization | θ (global) + φ_e (domain-specific) | w (global only) |
| Loss | CE | FedAvg: CE / FedProx: CE + Prox |

## Running Tests

```bash
# Test baseline models
pytest tests/test_baseline_models.py -v

# Test baseline trainers
pytest tests/test_baseline_trainer.py -v

# Run all baseline tests
pytest tests/test_baseline*.py -v
```
