#!/usr/bin/env python3
"""Module that saves and loads a model's configuration in JSON format"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format

    Args:
        network: the model whose configuration should be saved
        filename: path of the file that the configuration
            should be saved to
    """
    json_config = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_config)
    return None


def load_config(filename):
    """Loads a model with a specific configuration

    Args:
        filename: path of the file containing the model's
            configuration in JSON format
    """
    with open(filename, 'r') as f:
        json_config = f.read()
    return K.models.model_from_json(json_config)
