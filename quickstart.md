# Quickstart Guide: FL-DomainNet-FAP-LoRA

**Last Updated**: 2025-11-03
**Estimated Setup Time**: 30-45 minutes
**Requirements**: CUDA GPU (≥16GB VRAM), ~500GB disk space

This guide will walk you through setting up and running the FL-DomainNet-FAP-LoRA federated learning system from scratch.

---

## Prerequisites

### Hardware

- **GPU**: NVIDIA GPU with ≥16GB VRAM (tested on RTX 3090, RTX 4080, A100-40GB)
- **RAM**: 32GB+ recommended (64GB for comfortable multi-worker DataLoader)
- **Disk**: ~500GB free space (DomainNet dataset ~300GB uncompressed, outputs/checkpoints ~50GB)
- **CPU**: 8+ cores recommended for DataLoader workers

### Software

- **OS**: Linux (Ubuntu 20.04+ or Arch), also works on macOS with CUDA via Docker
- **CUDA**: 11.8+ (check with `nvidia-smi`)
- **Python**: 3.11+ (managed by uv, see installation below)

---

## Installation

### Step 1: Install uv (Package Manager)

uv is a fast Python package manager that replaces pip. Install it globally:

```bash
# Install uv via official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (the installer will prompt you, or add manually to ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uv --version
# Expected output: uv 0.1.x or higher
```

### Step 2: Clone Repository and Setup Project

```bash
# Navigate to project root
cd /root/xlw_exp  # Or your preferred directory

# Create Python 3.11 virtual environment
uv venv --python 3.11

# Activate environment
source .venv/bin/activate

# Verify Python version
python --version
# Expected: Python 3.11.x
```

### Step 3: Install Dependencies

Create `pyproject.toml` (if not already present):

```toml
[project]
name = "fl-domainnet"
version = "1.0.0"
description = "Federated Learning with FAP-Float(S) and EAPH-LoRA for DomainNet"
requires-python = ">=3.11"
dependencies = [
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "numpy>=1.23.0",
    "pillow>=9.2.0",
    "pyyaml>=6.0",
    "pytest>=7.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

Install dependencies:

```bash
# Install all dependencies from pyproject.toml
uv pip install -e .

# Verify PyTorch installation with CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
# Expected: PyTorch 1.12.x+, CUDA available: True
```

---

## Data Preparation

### Step 4: Download DomainNet Dataset

DomainNet is a large-scale dataset (~300GB). Download from official sources:

```bash
# Create data directory
mkdir -p datasets/domainnet

# Download via official script (or manual download from http://ai.bu.edu/M3SDA/)
cd datasets/domainnet

# Download each domain (6 domains: clipart, infograph, painting, quickdraw, real, sketch)
# Example for one domain (repeat for all 6):
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
unzip clipart.zip
rm clipart.zip

# Repeat for all domains or use provided script
bash ../../scripts/prepare_domainnet.sh
```

**Expected structure**:
```
datasets/domainnet/
├── clipart/
│   ├── aircraft_carrier/
│   │   ├── 00001.jpg
│   │   └── ...
│   ├── airplane/
│   └── ... (345 or 126 classes)
├── infograph/
├── painting/
├── quickdraw/
├── real/
└── sketch/
```

### Step 5: Generate Index File

The system requires an `index.json` mapping samples to domains and labels:

```bash
# Run index generation script (create if not exists)
python -c "
import os, json
from collections import defaultdict

data_root = 'datasets/domainnet'
domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
class_names = sorted(set(os.listdir(f'{data_root}/clipart')))  # Get class list from one domain

class_to_label = {name: i for i, name in enumerate(class_names)}
index = []

for domain in domains:
    domain_path = os.path.join(data_root, domain)
    for class_name in os.listdir(domain_path):
        class_path = os.path.join(domain_path, class_name)
        if not os.path.isdir(class_path):
            continue
        label = class_to_label.get(class_name)
        if label is None:
            continue  # Skip classes not in 126/345 subset
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.jpg', '.png')):
                rel_path = f'{domain}/{class_name}/{img_file}'
                index.append({'path': rel_path, 'label': label, 'domain': domain})

# Save index
with open(f'{data_root}/index.json', 'w') as f:
    json.dump(index, f)

print(f'Generated index with {len(index)} samples, {len(class_names)} classes')
"

# Expected output: Generated index with ~350000 samples, 126 classes (or 345)
```

---

## Configuration

### Step 6: Create Configuration File

Create `configs/default.yaml`:

```yaml
data:
  root: "./datasets/domainnet"
  split: "domainnet-126"  # or "domainnet-345"
  domains: ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
  num_clients_per_domain: 24
  dirichlet_alpha: 0.1
  unload_ratio: 0.2
  val_ratio: 0.1
  num_workers: 8

train:
  total_rounds: 200
  clients_participation: 0.2
  local_steps: 3
  batch_size: 64
  optimizer: "adamw"
  lr_theta: 3.0e-4
  lr_phi: 1.0e-3
  weight_decay: 1.0e-4
  cosine_lr: true

model:
  backbone: "resnet18"
  lora:
    enable: true
    rank: 16
    alpha: 32

selector:
  period_K: 5
  tau: 0.5
  w1: 1.0
  w2: 0.5
  w3: 0.1
  w4: 0.2
  drift_proj_dim: 64

logging:
  out_dir: "./outputs/exp1"
  eval_interval: 1
  ckpt_interval: 20

seed: 42
device: "cuda"
```

**Key parameters to adjust**:
- `data.root`: Path to DomainNet dataset
- `train.batch_size`: Reduce if GPU OOM (try 32 or 16)
- `data.num_workers`: Reduce if RAM constrained (try 4 or 2)
- `logging.out_dir`: Where to save logs and checkpoints
- `device`: Use "cpu" for testing without GPU (very slow)

---

## Running Experiments

### Step 7: Run Default Experiment

**Quick test** (2 domains, 3 rounds, smoke test):

```bash
python run_experiment.py --config configs/default.yaml --quick-test
# Expected: Completes in ~5 minutes, logs to outputs/exp1/train.log
```

**Full training** (6 domains, 200 rounds):

```bash
# Run in background with nohup (long-running)
nohup python run_experiment.py --config configs/default.yaml > train.out 2>&1 &

# Or use screen/tmux for interactive monitoring
screen -S fl-training
python run_experiment.py --config configs/default.yaml
# Detach: Ctrl+A, D
# Reattach: screen -r fl-training
```

**Expected runtime**: 24-48 hours on RTX 3090 for 200 rounds.

### Step 8: Monitor Progress

**Watch logs in real-time**:

```bash
tail -f outputs/exp1/train.log
```

**Expected log format**:
```
2025-11-03 10:15:32 - Round 1
2025-11-03 10:15:35 - Training 29 clients (clipart: 5, infograph: 4, ...)
2025-11-03 10:17:12 - Evaluation: AvgAcc=0.423, WorstAcc=0.312, Var=0.008
2025-11-03 10:17:15 - Round 5: Aggregator selection
2025-11-03 10:17:15 - L_map: {'clipart': 2.45, 'infograph': 2.89, ...}
2025-11-03 10:17:15 - Selected: infograph (p=0.35)
...
```

**Check GPU usage**:

```bash
watch -n 1 nvidia-smi
# Expected: GPU utilization 80-100%, memory ~14GB/16GB
```

---

## Outputs and Results

### Checkpoints

Saved to `outputs/exp1/checkpoints/`:

- `theta_global_r20.pt`, `theta_global_r40.pt`, ..., `theta_global_r200.pt` (global backbone)
- `phi_clipart_r20.pt`, `phi_real_r20.pt`, ... (domain-specific parameters per domain)
- `edge_manager_final.pt` (EMA losses, prototypes, final state)

**Loading checkpoints**:

```python
import torch

# Load global backbone
theta = torch.load('outputs/exp1/checkpoints/theta_global_r200.pt', map_location='cpu')
model.load_state_dict(theta['state_dict'], strict=False)

# Load domain-specific parameters
phi_clipart = torch.load('outputs/exp1/checkpoints/phi_clipart_r200.pt', map_location='cpu')
model.load_state_dict(phi_clipart['state_dict'], strict=False)
```

### Metrics

Logged per-round metrics include:

- `AvgAcc`: Average accuracy across all 6 domains
- `WorstAcc`: Minimum accuracy (fairness metric)
- `Var`: Variance of domain accuracies (lower = more fair)
- `L_map`: Per-domain EMA validation losses
- `drift_map`: Per-domain prototype drift scores
- `aggregator`: Selected aggregation point (every K rounds)

**Extract metrics for plotting**:

```bash
grep "AvgAcc" outputs/exp1/train.log | awk '{print $3}' > avg_acc.txt
grep "WorstAcc" outputs/exp1/train.log | awk '{print $3}' > worst_acc.txt
```

### Visualizations (Post-Training)

Generate visualizations using provided script:

```bash
python scripts/visualize_results.py --log outputs/exp1/train.log --output outputs/exp1/plots/

# Generates:
# - aggregator_timeline.png (heatmap of selected aggregators over rounds)
# - worst_domain_curve.png (worst-domain accuracy vs. round)
# - drift_heatmap.png (per-domain drift scores over time)
```

---

## Troubleshooting

### Common Errors

**1. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory. Tried to allocate 2.5GB (GPU 0; 15.75GB total)
```

**Solutions**:
- Reduce `batch_size` in config (try 32, then 16)
- Reduce `num_clients_per_domain` (try 12 instead of 24)
- Use gradient accumulation (not implemented in MVP, requires code modification)

**2. Missing Dataset Files**

```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/domainnet/clipart/...'
```

**Solutions**:
- Verify dataset downloaded completely (`du -sh datasets/domainnet` should show ~300GB)
- Check `data.root` path in config matches actual location
- Regenerate `index.json` (see Step 5)

**3. DataLoader Worker Errors**

```
RuntimeError: DataLoader worker (pid XXXX) is killed by signal: Bus error
```

**Solutions**:
- Reduce `num_workers` in config (try 4, 2, or 0)
- Increase shared memory if using Docker (`--shm-size=16g`)
- Check RAM availability (`free -h`)

**4. NaN Losses**

```
WARNING: Domain clipart validation loss is NaN
```

**Solutions**:
- Check learning rates (reduce `lr_theta` to 1e-4, `lr_phi` to 5e-4)
- Verify ImageNet normalization applied (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- Inspect data for corrupted images (`identify -verbose file.jpg`)

**5. Slow Training**

Expected: ~2-3 minutes per round. If much slower (>10 min/round):

**Solutions**:
- Verify GPU being used (`device: "cuda"` in config, `nvidia-smi` shows utilization)
- Reduce `local_steps` (try 1 instead of 3)
- Disable cosine_lr (`cosine_lr: false`) to skip scheduler overhead
- Check I/O bottleneck (use SSD for dataset, not HDD)

---

## Running Tests

**Unit tests** (fast, <1 minute):

```bash
pytest tests/ -v
# Expected: All tests pass (PASSED), ~15-20 tests total
```

**Integration test** (slow, ~2 minutes):

```bash
pytest tests/test_integration.py -v -s
# Expected: test_full_training_loop PASSED
```

**Test with coverage**:

```bash
pytest tests/ --cov=core --cov=data --cov=models --cov-report=html
# Coverage report saved to htmlcov/index.html
# Target: >80% coverage for core/, data/, models/
```

---

## Next Steps

1. **Baseline Experiments**: Run ablation studies by modifying config:
   - No aggregation point selection: Set `w1=w2=w3=w4=0` (uniform random)
   - No LoRA: Set `model.lora.enable=false` (full global model)
   - No offloading: Set `unload_ratio=0.0`

2. **Hyperparameter Tuning**: Grid search over:
   - `selector.tau` ∈ {0.3, 0.5, 1.0}
   - `selector.period_K` ∈ {1, 5, 10}
   - `data.unload_ratio` ∈ {0, 0.1, 0.2, 0.4}

3. **Multiple Seeds**: Run 3 times with different seeds (42, 123, 777) for statistical significance

4. **Visualization**: Analyze worst-domain trajectory, aggregator selection patterns, drift correlations

---

## Support and Documentation

- **Full Documentation**: See `doc.md` for architecture details
- **API Contracts**: See `contracts/` for interface specifications
- **Data Model**: See `data-model.md` for entity definitions
- **Issues**: Report bugs or ask questions in project issues

---

## Quick Reference

**Start training**:
```bash
source .venv/bin/activate
python run_experiment.py --config configs/default.yaml
```

**Monitor progress**:
```bash
tail -f outputs/exp1/train.log
watch -n 1 nvidia-smi
```

**Check results**:
```bash
grep "WorstAcc" outputs/exp1/train.log | tail -20  # Last 20 rounds
ls -lh outputs/exp1/checkpoints/  # Checkpoint files
```

**Run tests**:
```bash
pytest tests/ -v
```

---

**Status**: ✅ Quickstart guide complete. You should now be able to set up and run the FL-DomainNet-FAP-LoRA system end-to-end.
