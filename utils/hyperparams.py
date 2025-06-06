# hyperparams.py
"""Customizable model hyperparameter """
import torch.optim as optim
import torch.nn as nn

from .loss_functions import (
    mse_cc_loss,
    correlation_coefficient_loss,
)

# LSTM(10, 20, 2) -> input has 10 features, 20 hidden size and 2 layers.
# NOTE: Make sure to set batch_first=True. Optionally set bidirectional=True
RNN_CELL_FACTORY = {'lstm': nn.LSTM, 'gru': nn.GRU}

LOSS_FN_FACTORY = {
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss(),
    'mse_and_pearson': mse_cc_loss,
    'pearson': correlation_coefficient_loss,
    'binary_cross_entropy': nn.BCELoss(),
}

ACTIVATION_FN_FACTORY = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'tanh': nn.Tanh(),
    'lrelu': nn.LeakyReLU(),
    'elu': nn.ELU()
}
OPTIMIZER_FACTORY = {
    'adam': optim.Adam,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'gd': optim.SGD,
    'sparseadam': optim.SparseAdam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop
}

# ──────────────────────────────────────────────────────────────
# Learning-rate scheduler factory
# ──────────────────────────────────────────────────────────────
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR

SCHEDULER_FACTORY = {
    # plateau on validation loss (default in train.py)
    "plateau": lambda opt, **kw: ReduceLROnPlateau(
        opt, mode="min", factor=kw.get("factor", 0.3), patience=kw.get("patience", 3)
    ),
    "step": lambda opt, **kw: StepLR(
        opt, step_size=kw.get("step_size", 10), gamma=kw.get("gamma", 0.1)
    ),
    "exp": lambda opt, **kw: ExponentialLR(opt, gamma=kw.get("gamma", 0.95)),
}