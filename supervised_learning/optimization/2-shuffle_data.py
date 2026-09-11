#!/usr/bin/env python3
"""Module that shuffles the data points in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way

    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle
        Y: numpy.ndarray of shape (m, ny) to shuffle
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
