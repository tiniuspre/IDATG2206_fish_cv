import logging

import cv2

from fish_cv.constants import D_A, D_B, H_B, INPUT_PATH, L_B, OUTPUT_PATH
from fish_cv.detect_img import is_whole_fish_in_image
from fish_cv.geometry import find_left_right_points
from fish_cv.measurement import calculate_real_length, pixel_distance
from fish_cv.segmentation import (
    apply_mask,
    create_fish_mask,
    keep_largest_contour,
    load_image,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """Main program."""
    img = load_image(INPUT_PATH)
    if not is_whole_fish_in_image(load_image(INPUT_PATH)):
        logger.warning(
            'The whole fish is not in the image.'
            'Please provide an image with the whole fish visible.'
        )
        return

    fish_mask = create_fish_mask(img)
    clean_mask = keep_largest_contour(fish_mask)

    left, right = find_left_right_points(clean_mask)
    result = apply_mask(img, clean_mask)

    h_a = pixel_distance(left, right)
    l_a = calculate_real_length(h_a, H_B, L_B, D_A, D_B)

    logger.info('Left pixel: %s', left)
    logger.info('Right pixel: %s', right)
    logger.info('Reference object in pixels = %.2f px', H_B)
    logger.info('Fish length in pixels = %.2f px', h_a)
    logger.info('Fish length in cm = %.2f cm', l_a)

    cv2.circle(result, left, 10, (0, 0, 255), -1)
    cv2.circle(result, right, 10, (255, 0, 0), -1)

    cv2.imwrite(OUTPUT_PATH, result)
    logger.info('Saved: %s', OUTPUT_PATH)


if __name__ == '__main__':
    main()
