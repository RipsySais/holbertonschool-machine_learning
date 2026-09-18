#!/usr/bin/env python3
"""Module that creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes) with
            the correct labels for each data point
        logits: one-hot numpy.ndarray of shape (m, classes) with
            the predicted labels
    """
    classes = labels.shape[1]
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)

    confusion = np.zeros((classes, classes))
    np.add.at(confusion, (true_classes, predicted_classes), 1)

    return confusion
