import numpy as np


def pixel_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Return the Euclidean distance in pixels between two points."""
    x1, y1 = p1
    x2, y2 = p2
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def calculate_real_length(
    h_a: float,
    h_b: float,
    l_b: float,
    d_a: float,
    d_b: float,
) -> float:
    """Calculate the real length of the fish."""
    return float(l_b * (h_a / h_b) * (d_a / d_b))
