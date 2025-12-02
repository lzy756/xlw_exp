"""Baseline FL core components."""

from baseline.core.trainer_fedavg import LocalTrainerFedAvg
from baseline.core.trainer_fedprox import LocalTrainerFedProx
from baseline.core.selector_fixed import FixedSelector

__all__ = ['LocalTrainerFedAvg', 'LocalTrainerFedProx', 'FixedSelector']
