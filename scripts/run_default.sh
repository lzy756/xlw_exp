#!/bin/bash
# Convenience script to run the FL-DomainNet experiment with default configuration

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the experiment
python run_experiment.py --config configs/default.yaml "$@"