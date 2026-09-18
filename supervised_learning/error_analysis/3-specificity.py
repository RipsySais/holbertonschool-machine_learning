#!/usr/bin/env python3
"""Module that calculates the specificity for each class
in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
            where rows are correct labels and columns are predicted
    """
    total = np.sum(confusion)
    VP = np.diagonal(confusion)
    FP = np.sum(confusion, axis=0) - VP
    FN = np.sum(confusion, axis=1) - VP
    VN = total - (VP + FP + FN)

    return VN / (VN + FP)
