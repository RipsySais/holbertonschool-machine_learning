#!/usr/bin/env python3
"""Module that converts a label vector into a one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix

    Args:
        labels: label vector to convert
        classes: number of classes (optional)
    """
    return K.utils.to_categorical(labels, num_classes=classes)
