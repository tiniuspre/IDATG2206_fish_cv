import dataclasses
from statistics import mean, median

# This module provides a class for estimating fish size based on measurements from multiple images.
# It can use both mean and median to provide robust estimates, depending on the occurrence of outliers.
@dataclasses.dataclass
class ImageMeasurement:
    img_id: str
    length: float
    height: float

# The FishSizeEstimator class takes a list of ImageMeasurement instances and provides methods
# to estimate the average length and height of the fish.
class FishSizeEstimator:
    def __init__(self, measurements) -> None:
        self.measurements = measurements
        self._validate()

    def _validate(self) -> None:
        if not self.measurements:
            raise ValueError("No measurements provided")

    # Helper method to calculate quartiles for outlier detection
    @staticmethod
    def _quartiles(values: list[float]) -> tuple[float, float]:
        values = sorted(values)
        n = len(values)
        mid = n // 2 # index for splitting data

        if n % 2 == 0: # even number of values
            lower_half = values[:mid]
            upper_half = values[mid:]
        else:
            lower_half = values[:mid]
            upper_half = values[mid + 1:] # exclude median for odd count

        q1 = median(lower_half)
        q3 = median(upper_half)
        return q1, q3

    # Simple outlier detection using IQR method
    @classmethod
    def _has_outliers(cls, values: list[float]) -> bool:
        if len(values) < 4:
            return False

        q1, q3 = cls._quartiles(values)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return any(x < lower_bound or x > upper_bound for x in values)


    # If outliers are detected, use median; otherwise, use mean
    @classmethod
    def _robust_center(cls, values: list[float]) -> float:
        return median(values) if cls._has_outliers(values) else mean(values)

    def avg_length(self) -> float:
        return mean(m.length for m in self.measurements)

    def avg_height(self) -> float:
        return mean(m.height for m in self.measurements)

    def median_length(self) -> float:
        return median(m.length for m in self.measurements)

    def median_height(self) -> float:
        return median(m.height for m in self.measurements)

    def best_length(self) -> float:
        lengths = [m.length for m in self.measurements]
        return self._robust_center(lengths)

    def best_height(self) -> float:
        heights = [m.height for m in self.measurements]
        return self._robust_center(heights)

    # The relative_error method calculates the relative error between the estimated and measured values
    # which can be useful for evaluating the accuracy of the estimation.
    @staticmethod
    def relative_error(estimated: float, measured: float) -> float:
        if measured == 0:
            raise ValueError("Measured value cannot be zero for relative error calculation")
        return abs(estimated - measured) / abs(measured)

    # The result method provides a summary of the estimation process, including which method was used and the estimated values.
    def result(self) -> dict:
        lengths = [m.length for m in self.measurements]
        heights = [m.height for m in self.measurements]


        return {
            "length used": "median" if self._has_outliers(lengths) else "mean",
            "height used": "median" if self._has_outliers(heights) else "mean",
            "estimated_length": self.best_length(),
            "estimated_height": self.best_height(),
        }

# Example usage with test data
if __name__ == "__main__":
    img_measurements = [
        ImageMeasurement("img_1", 55.8, 12.2),
        ImageMeasurement("img_2", 60.3, 10.5),
        ImageMeasurement("img_3", 58.7, 11.0),
        ImageMeasurement("img_4", 57.2, 9.8),
        ImageMeasurement("img_5", 59.1, 11.5),
        ImageMeasurement("img_6", 56.5, 10.0),
        ImageMeasurement("img_7", 61.0, 12.8),
        ImageMeasurement("img_8", 58.0, 11.2),
        ImageMeasurement("img_9", 57.5, 10.7),
        ImageMeasurement("img_10", 59.5, 11.3)
    ]

    imgBatch = FishSizeEstimator(img_measurements)
    result = imgBatch.result()

    print(result)