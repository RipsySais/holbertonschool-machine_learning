#!/usr/bin/env python3
"""Module that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network using the Keras functional API

    Args:
        nx: number of input features to the network
        layers: list of number of nodes in each layer
        activations: list of activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    """
    regularizer = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))

    x = K.layers.Dense(
        layers[0], activation=activations[0],
        kernel_regularizer=regularizer)(inputs)

    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=regularizer)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
