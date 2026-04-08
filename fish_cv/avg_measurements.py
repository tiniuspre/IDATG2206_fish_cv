import dataclasses
from statistics import mean, median

MIN_VALUE_OUTLIER_CHECK = 4


@dataclasses.dataclass
class ImageMeasurement:
    """Stores fish measurements from a single image."""

    img_id: str
    length: float
    height: float


class FishSizeEstimator:
    """Estimates fish size from multiple image measurements."""

    def __init__(self, measurements: list[ImageMeasurement]) -> None:
        """Initializes the FishSizeEstimator."""
        self.measurements = measurements
        self._validate()

    def _validate(self) -> None:
        if not self.measurements:
            msg = 'No measurements provided'
            raise ValueError(msg)

    @staticmethod
    def _quartiles(values: list[float]) -> tuple[float, float]:
        values = sorted(values)
        n = len(values)
        mid = n // 2  # index for split data

        if n % 2 == 0:  # even number
            lower_half = values[:mid]
            upper_half = values[mid:]
        else:
            lower_half = values[:mid]
            upper_half = values[mid + 1 :]  # exclude median odd count

        q1 = median(lower_half)
        q3 = median(upper_half)
        return q1, q3

    @classmethod
    def _has_outliers(cls, values: list[float]) -> bool:
        """Detects outliers in the list of values."""
        if len(values) < MIN_VALUE_OUTLIER_CHECK:
            return False

        q1, q3 = cls._quartiles(values)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return any(x < lower_bound or x > upper_bound for x in values)

    @classmethod
    def _robust_center(cls, values: list[float]) -> float:
        """Returns the median if outliers are detected."""
        return median(values) if cls._has_outliers(values) else mean(values)

    def avg_length(self) -> float:
        """Calculates the average length from the measurements."""
        return mean(m.length for m in self.measurements)

    def avg_height(self) -> float:
        """Calculates the average height from the measurements."""
        return mean(m.height for m in self.measurements)

    def median_length(self) -> float:
        """Calculates the median length from the measurements."""
        return median(m.length for m in self.measurements)

    def median_height(self) -> float:
        """Calculates the median height from the measurements."""
        return median(m.height for m in self.measurements)

    def best_length(self) -> float:
        """Determines the best length estimate."""
        lengths = [m.length for m in self.measurements]
        return self._robust_center(lengths)

    def best_height(self) -> float:
        """Determines the best height estimate."""
        heights = [m.height for m in self.measurements]
        return self._robust_center(heights)

    @staticmethod
    def relative_error(estimated: float, measured: float) -> float:
        """Calculates the relative error between the estimated and measured values."""
        if measured == 0:
            msg = 'Measured value cannot be zero for relative error calculation'
            raise ValueError(msg)
        return abs(estimated - measured) / abs(measured)

    def result(self) -> dict:
        """Provides a summary of the estimation process."""
        lengths = [m.length for m in self.measurements]
        heights = [m.height for m in self.measurements]

        return {
            'length used': 'median' if self._has_outliers(lengths) else 'mean',
            'height used': 'median' if self._has_outliers(heights) else 'mean',
            'estimated_length': self.best_length(),
            'estimated_height': self.best_height(),
        }
