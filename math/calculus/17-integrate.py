#!/usr/bin/env python3
"""Module that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) is not int:
        return None

    integral = [C]
    for i in range(len(poly)):
        coeff = poly[i] / (i + 1)
        if coeff == int(coeff):
            coeff = int(coeff)
        integral.append(coeff)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
