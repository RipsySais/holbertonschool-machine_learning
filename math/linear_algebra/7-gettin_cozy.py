#!/usr/bin/env python3
"""Module that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [list(row) for row in mat1] + [list(row) for row in mat2]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [list(r1) + list(r2) for r1, r2 in zip(mat1, mat2)]
    return None
