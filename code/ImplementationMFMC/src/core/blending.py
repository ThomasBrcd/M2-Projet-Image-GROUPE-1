import cv2
import numpy as np
from .face_swap import SwapPatch


def geometric_mask(size):
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(mask, (size // 2, int(size * .53)), (int(size * .34), int(size * .40)),
                0, 0, 360, 1., -1)
    return cv2.GaussianBlur(mask, (0, 0), max(1, size * .025))


def compose(image, patch: SwapPatch, protected_faces=(), photometric=True):
    """Fusion dans une ROI ; les cadres des visages non sélectionnés sont exclus exactement."""
    h, w = image.shape[:2]
    size = patch.bgr.shape[0]
    inverse = cv2.invertAffineTransform(patch.matrix)
    corners = cv2.transform(np.float32([[[0, 0], [size, 0], [size, size], [0, size]]]), inverse)[0]
    x1, y1 = np.maximum(0, np.floor(corners.min(axis=0)).astype(int) - 2)
    x2, y2 = np.minimum([w, h], np.ceil(corners.max(axis=0)).astype(int) + 2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Visage aligné hors image")
    inverse[:, 2] -= [x1, y1]
    dims = (int(x2 - x1), int(y2 - y1))
    alpha = cv2.warpAffine(np.clip(patch.alpha, 0, 1), inverse, dims)
    generated = cv2.warpAffine(patch.bgr, inverse, dims).astype(np.float32)
    original = image[y1:y2, x1:x2].astype(np.float32)
    for face in protected_faces:
        a, b, c, d = np.asarray(face.bbox, dtype=int)
        alpha[max(0, b-y1):max(0, min(d, y2)-y1), max(0, a-x1):max(0, min(c, x2)-x1)] = 0
    if photometric:
        support = alpha > .7
        if support.sum() > 30:
            delta = np.clip(original[support].mean(0) - generated[support].mean(0), -12, 12)
            generated = np.clip(generated + delta, 0, 255)
    alpha[alpha < .005] = 0
    mixed = original * (1-alpha[..., None]) + generated * alpha[..., None]
    output = image.copy()
    output[y1:y2, x1:x2] = np.rint(mixed).clip(0, 255).astype(np.uint8)
    return output, (int(x1), int(y1), alpha)


def emergency_mask(image, faces, protected_faces=()):
    """Masquage opaque explicite, uniquement sur les visages sélectionnés."""
    result = image.copy()
    h, w = result.shape[:2]
    for face in faces:
        x1, y1, x2, y2 = np.asarray(face.bbox, dtype=int)
        result[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 0
    for face in protected_faces:
        x1, y1, x2, y2 = np.asarray(face.bbox, dtype=int)
        result[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    return result
