import numpy as np


def find_left_right_points(mask):
    """Returns the coordinates for the farthest point to the left and right of the fish"""
    ys, xs = np.nonzero(mask > 0)

    if len(xs) == 0:
        raise ValueError("No segmented object found")

    min_x = xs.min()
    left_y = ys[xs == min_x].min()

    max_x = xs.max()
    right_y = ys[xs == max_x].min()

    left = (int(min_x), int(left_y))
    right = (int(max_x), int(right_y))
    return left, right
