#!/usr/bin/env python3
"""Module that sets up gradient descent with momentum in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Sets up the gradient descent with momentum
    optimization algorithm in TensorFlow

    Args:
        alpha: the learning rate
        beta1: the momentum weight
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
