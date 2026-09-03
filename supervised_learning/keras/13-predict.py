#!/usr/bin/env python3
"""Module that makes a prediction using a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network

    Args:
        network: the network model to make the prediction with
        data: the input data to make the prediction with
        verbose: boolean, determines if output should be printed
    """
    return network.predict(data, verbose=verbose)
