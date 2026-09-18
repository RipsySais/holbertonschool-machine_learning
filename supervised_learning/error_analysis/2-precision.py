#!/usr/bin/env python3
"""Module that calculates the precision for each class
in a confusion matrix"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where rows are correct labels and columns are predicted
    """
    true_positives = np.diagonal(confusion)
    predicted_totals = np.sum(confusion, axis=0)
    return true_positives / predicted_totals
