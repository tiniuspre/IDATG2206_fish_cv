from time import sleep
from typing import Self

import cv2
import numpy as np


class CameraController:
    """A context manager for controlling the camera and capturing images."""

    def __init__(self) -> None:
        """Initialize the camera controller with no active camera."""
        self.camera: cv2.VideoCapture | None = None

    def take_image(self) -> np.ndarray:
        """Capture an image from the camera and return it as a grayscale numpy array."""
        _, image = self.camera.read()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def save_image(self, image: np.ndarray) -> None:
        """Save the captured image to a file."""
        cv2.imwrite('snapshot.jpg', image)

    def __enter__(self) -> Self:
        """Initialize the camera and wait until it is ready to capture images."""
        self.camera = cv2.VideoCapture(0)
        camera_not_ready = True
        while camera_not_ready:
            _, image = self.camera.read()
            if cv2.countNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) != 0:
                camera_not_ready = False
            sleep(0.01)

        return self

    def __exit__(self, *_: object) -> None:
        """Release the camera resource when exiting the context manager."""
        self.camera.release()
        del self.camera


if __name__ == '__main__':
    with CameraController() as camera:
        img = camera.take_image()
        camera.save_image(img)
