import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def create_fish_mask(img, kernel_size=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, fish_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    fish_mask = cv2.morphologyEx(fish_mask, cv2.MORPH_CLOSE, kernel)
    fish_mask = cv2.morphologyEx(fish_mask, cv2.MORPH_OPEN, kernel)

    return fish_mask

def keep_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)

    if not contours:
        raise ValueError("No contours found")

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    return clean_mask

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)