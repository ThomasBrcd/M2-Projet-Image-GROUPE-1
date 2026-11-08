from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class Face:
    bbox: np.ndarray
    landmarks: np.ndarray
    embedding: np.ndarray
    confidence: float = 1.0


class Recognizer(Protocol):
    @property
    def model_id(self) -> str: ...
    def detect(self, bgr: np.ndarray) -> list[Face]: ...


def intersection_over_union(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(0, hi - lo)))
    area_a = float(np.prod(np.maximum(0, np.asarray(a[2:]) - a[:2])))
    area_b = float(np.prod(np.maximum(0, np.asarray(b[2:]) - b[:2])))
    return intersection / max(area_a + area_b - intersection, 1e-6)
