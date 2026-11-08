from unittest.mock import Mock
import numpy as np
from src.backends.insightface_backend import InsightFaceBackend


def test_close_portrait_fallback_preserves_original_coordinates():
    backend = InsightFaceBackend.__new__(InsightFaceBackend)
    backend.detector = Mock()
    # Image de 1000 pixels réduite à 640, puis entourée de marges de 320 pixels.
    box = np.float32([[384, 448, 704, 832, .9]])
    points = np.float32([[[448, 512], [640, 512], [544, 608], [480, 704], [608, 704]]])
    backend.detector.detect.side_effect = [(np.empty((0,5)), None), (box, points)]
    boxes, landmarks = backend._detect_boxes(np.zeros((1000,1000,3),np.uint8))
    np.testing.assert_allclose(boxes[0,:4], [100,200,600,800])
    np.testing.assert_allclose(landmarks[0,0], [200,300])
    assert backend.detector.detect.call_count == 2


def test_existing_detections_do_not_trigger_portrait_fallback():
    backend = InsightFaceBackend.__new__(InsightFaceBackend)
    backend.detector = Mock(return_value=None)
    expected = (np.float32([[10,10,50,50,.9]]), np.zeros((1,5,2),np.float32))
    backend.detector.detect.return_value = expected
    boxes, landmarks = backend._detect_boxes(np.zeros((100,100,3),np.uint8))
    assert boxes is expected[0] and landmarks is expected[1]
    assert backend.detector.detect.call_count == 1
