import cv2

from fish_cv.constants import D_A, D_B, H_B, INPUT_PATH, L_B, OUTPUT_PATH
from fish_cv.geometry import find_left_right_points
from fish_cv.measurement import calculate_real_length, pixel_distance
from fish_cv.segmentation import (
    apply_mask,
    create_fish_mask,
    keep_largest_contour,
    load_image,
)


def main():
    """Main program"""
    img = load_image(INPUT_PATH)
    fish_mask = create_fish_mask(img)
    clean_mask = keep_largest_contour(fish_mask)

    left, right = find_left_right_points(clean_mask)
    result = apply_mask(img, clean_mask)

    h_a = pixel_distance(left, right)
    l_a = calculate_real_length(h_a, H_B, L_B, D_A, D_B)

    print(f"Left pixel: {left}")
    print(f"Right pixel: {right}")
    print(f"Reference object in pixels = {H_B:.2f} px")
    print(f"Fish length in pixels = {h_a:.2f} px")
    print(f"Fish length in cm = {l_a:.2f} cm")

    cv2.circle(result, left, 10, (0, 0, 255), -1)
    cv2.circle(result, right, 10, (255, 0, 0), -1)

    cv2.imwrite(OUTPUT_PATH, result)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
