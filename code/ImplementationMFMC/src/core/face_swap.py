from dataclasses import dataclass
from typing import Protocol
import numpy as np
from .detection import Face


@dataclass
class SwapPatch:
    bgr: np.ndarray
    alpha: np.ndarray
    matrix: np.ndarray  # image originale -> crop aligné
    warnings: list[str]


class FaceSwapper(Protocol):
    def swap(self, image: np.ndarray, face: Face, donor_image: np.ndarray,
             donor_face: Face, segmentation: bool = True) -> SwapPatch: ...
