import numpy as np
import pytest
from PIL import Image
from src.utils.image_utils import load_image, save_image


@pytest.mark.parametrize("extension", ["png", "jpg", "webp"])
def test_roundtrip_dimensions_and_bgr(tmp_path, extension):
    image = np.full((37, 51, 3), [5, 25, 220], np.uint8)
    path = tmp_path/f"épreuve.{extension}"
    save_image(path,image)
    output = load_image(path)
    assert output.shape == image.shape
    assert np.abs(output.astype(float).mean((0,1))-image.mean((0,1))).max() < 5


def test_exif_orientation_applied_before_detection(tmp_path):
    image = Image.new("RGB",(30,50),(200,30,10))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path/"oriented.jpg"
    image.save(path,exif=exif)
    assert load_image(path).shape[:2] == (30,50)


def test_save_cannot_overwrite_reference_or_source(tmp_path):
    path = tmp_path/"source.png"
    image = np.zeros((10,10,3),np.uint8)
    save_image(path,image)
    with pytest.raises(ValueError,match="écrasée"):
        save_image(path,image,[path])
