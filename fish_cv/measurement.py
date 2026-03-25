import numpy as np


def pixel_distance(p1, p2):
    """Returns the Euclidean distance in pixels between two points"""
    x1, y1 = p1
    x2, y2 = p2
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

def calculate_real_length(h_a, h_b, l_b, d_a, d_b):
    """Calculate the length of the fish"""
    return l_b * (h_a / h_b) * (d_a / d_b)
