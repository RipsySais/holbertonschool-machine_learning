#!/usr/bin/env python3
"""Module that updates the learning rate using inverse time decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy

    Args:
        alpha: the original learning rate
        decay_rate: weight used to determine the decay rate
        global_step: number of passes of gradient descent elapsed
        decay_step: number of passes before alpha decays further
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
