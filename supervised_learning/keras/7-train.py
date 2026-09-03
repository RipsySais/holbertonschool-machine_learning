#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent, with optional
    validation data, early stopping, and learning rate decay

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes)
        batch_size: size of the batch for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: data to validate the model with, if not None
        early_stopping: boolean, whether to use early stopping
        patience: patience used for early stopping
        learning_rate_decay: boolean, whether to use learning rate decay
        alpha: the initial learning rate
        decay_rate: the decay rate
        verbose: boolean, determines if output should be printed
        shuffle: boolean, determines whether to shuffle the batches
    """
    callbacks = []

    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))

    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            """Calculates the learning rate using inverse time decay"""
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(
            K.callbacks.LearningRateScheduler(scheduler, verbose=1))

    history = network.fit(
        data, labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle)
    return history
