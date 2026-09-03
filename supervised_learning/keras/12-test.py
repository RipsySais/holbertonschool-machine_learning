#!/usr/bin/env python3
"""Module that tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network

    Args:
        network: the network model to test
        data: the input data to test the model with
        labels: correct one-hot labels of data
        verbose: boolean, determines if output should be printed
    """
    return network.evaluate(data, labels, verbose=verbose)
