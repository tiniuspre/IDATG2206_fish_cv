import numpy as np
from numpy.typing import NDArray


def find_left_right_points(
    mask: NDArray[np.uint8],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the leftmost and rightmost points of the fish mask."""
    ys, xs = np.nonzero(mask > 0)

    if len(xs) == 0:
        msg = 'No segmented object found.'
        raise ValueError(msg)

    min_x = xs.min()
    left_y = ys[xs == min_x].min()

    max_x = xs.max()
    right_y = ys[xs == max_x].min()

    left = (int(min_x), int(left_y))
    right = (int(max_x), int(right_y))
    return left, right
