#!/usr/bin/env python3
"""Module that calculates the sensitivity for each class
in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where rows are correct labels and columns are predicted
    """
    true_positives = np.diagonal(confusion)
    actual_totals = np.sum(confusion, axis=1)
    return true_positives / actual_totals
