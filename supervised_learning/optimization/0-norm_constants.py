#!/usr/bin/env python3
"""Module that calculates the normalization constants of a matrix"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants
    of a matrix

    Args:
        X: numpy.ndarray of shape (m, nx) to normalize
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
