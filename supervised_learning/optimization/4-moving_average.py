#!/usr/bin/env python3
"""Module that calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set,
    using bias correction

    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average
    """
    v = 0
    moving_averages = []
    for t in range(1, len(data) + 1):
        v = beta * v + (1 - beta) * data[t - 1]
        v_corrected = v / (1 - beta ** t)
        moving_averages.append(v_corrected)
    return moving_averages
