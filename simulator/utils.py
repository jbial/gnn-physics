"""Miscellaneous utility functions
"""
import torch
import random
import numpy as np


def finite_difference(x, h):
    """
    Computes centered fourth order finite difference
    with left-sided second order finite differences on the boundary
    """
    dx = np.zeros_like(x)

    center_diff = x[2:]-x[:-2]

    dx[2:-2] = (x[:-4] - x[4:] + 8*center_diff[1:-1]) / (12*h)

    dx[1] = center_diff[0] / (2*h)
    dx[-2] = center_diff[-1] / (2*h)
    dx[-1] = (3*x[-1] - 4*x[-2] + x[-3]) / (2*h)

    return dx


def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    """Fix the random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)