#!/usr/bin/env python3
"""Module that saves and loads an entire Keras model"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model

    Args:
        network: the model to save
        filename: path of the file that the model should be saved to
    """
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model

    Args:
        filename: path of the file that the model should be loaded from
    """
    return K.models.load_model(filename)
