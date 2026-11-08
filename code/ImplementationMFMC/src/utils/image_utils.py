from pathlib import Path
import os
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_image(path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError("Formats acceptés : PNG, JPEG, WebP")
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image(path, bgr, protected_paths=()):
    """Export sans métadonnées biométriques/EXIF et sans écraser les entrées."""
    path = Path(path).resolve()
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError("Choisir PNG, JPEG ou WebP")
    for protected in protected_paths:
        protected = Path(protected).resolve()
        if path == protected or (path.exists() and protected.exists() and os.path.samefile(path, protected)):
            raise ValueError("L'image d'entrée ne peut pas être écrasée")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        options = {"quality": 96} if path.suffix.lower() != ".png" else {}
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            options["subsampling"] = 0
        Image.fromarray(rgb).save(temporary, **options)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def face_thumbnail(image, bbox, size=96):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=int)
    crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if not crop.size:
        raise ValueError("Cadre de visage hors image")
    return cv2.resize(crop, (size, size))
