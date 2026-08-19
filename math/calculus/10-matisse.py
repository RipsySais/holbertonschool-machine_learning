#!/usr/bin/env python3
"""Module that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    derivative = [i * poly[i] for i in range(1, len(poly))]
    if all(c == 0 for c in derivative):
        return [0]
    return derivative
