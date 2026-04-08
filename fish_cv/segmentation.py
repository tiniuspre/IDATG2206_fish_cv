import cv2
import numpy as np
from numpy.typing import NDArray


def load_image(path: str) -> NDArray[np.uint8]:
    """Load an image from disk."""
    img = cv2.imread(path)
    if img is None:
        msg = f'Could not load image: {path}'
        raise FileNotFoundError(msg)
    return img


def create_fish_mask(
    img: NDArray[np.uint8],
    kernel_size: int = 7,
) -> NDArray[np.uint8]:
    """Create a binary mask for the fish."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, fish_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    fish_mask = cv2.morphologyEx(fish_mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(fish_mask, cv2.MORPH_OPEN, kernel)


def keep_largest_contour(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Keep only the largest contour in the mask."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    clean_mask = np.zeros_like(mask)

    if not contours:
        msg = 'No contours found.'
        raise ValueError(msg)

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    return clean_mask


def apply_mask(
    img: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Apply a binary mask to an image."""
    return cv2.bitwise_and(img, img, mask=mask)
