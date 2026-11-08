"""Tests unitaires du ciblage, de la fusion et de la validation avec backends simulés."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from src.core.detection import Face
from src.core.face_swap import SwapPatch
from src.core.identity_bank import Identity
from src.core.pipeline import Pipeline
from src.core.recognition import normalized
from src.core.errors import SelectionError, ModelError
from src.utils.config import Settings


def face(x=30, y=40, embedding=(1, 0, 0), size=70, angle=0):
    points = np.float32([[x+20, y+20], [x+50, y+20], [x+35, y+35], [x+23, y+52], [x+47, y+52]])
    points[:, 1] += np.float32([angle, -angle, 0, angle, -angle])
    return Face(np.float32([x, y, x+size, y+size]), points, normalized(embedding))


class Recognizer:
    model_id = "fake-model-unit-tests"

    def __init__(self, faces, refs=None, redetect=True):
        self.faces = faces
        self.refs = [face()] if refs is None else refs
        self.redetect = redetect

    def detect(self, image):
        if image.shape[:2] == (80, 80):
            return self.refs
        changed = bool(np.any(image[..., 1] > 100))
        if changed and not self.redetect:
            return []
        result = []
        for f in self.faces:
            x, y = ((f.bbox[:2]+f.bbox[2:])/2).astype(int)
            embedding = np.float32([0, 0, 1]) if image[y, x, 1] > 150 else f.embedding
            result.append(Face(f.bbox, f.landmarks, embedding))
        return result


class Swapper:
    def __init__(self):
        self.calls = []

    def swap(self, image, f, donor, donor_face, segmentation=True):
        self.calls.append(f)
        x, y, x2, y2 = f.bbox
        scale_x, scale_y = 64/(x2-x), 64/(y2-y)
        matrix = np.float32([[scale_x, 0, -x*scale_x], [0, scale_y, -y*scale_y]])
        crop = np.empty((64, 64, 3), np.uint8)
        crop[:] = donor[0, 0]
        return SwapPatch(crop, np.ones((64, 64), np.float32), matrix, [])


class Bank:
    def __init__(self, keys=("strong",)):
        self.entries = [Identity(key, Path("unused"), face(embedding=(0, 0, 1)), 100., "unused") for key in keys]

    def candidates(self, originals, maximum, manual_key=None, luminance=None):
        entries = [e for e in self.entries if manual_key is None or e.key == manual_key]
        return entries

    def image(self, identity):
        return np.full((64,64,3), [60,180 if identity.key != "weak" else 90,100], np.uint8)


def setup(faces, shape=(220, 360, 3), refs=None, bank=None, redetect=True):
    recognizer = Recognizer(faces, refs, redetect)
    swapper = Swapper()
    pipeline = Pipeline(recognizer, swapper, bank or Bank())
    reference = np.full((80,80,3), 11, np.uint8)
    image = np.full(shape, 33, np.uint8)
    return pipeline, swapper, reference, image


def test_no_faces_in_target():
    p, s, ref, img = setup([])
    analysis = p.analyze(ref, img, Settings())
    result = p.run(analysis, Settings())
    assert result.replaced == 0 and not s.calls
    assert np.array_equal(result.image, img)


def test_no_matching_person():
    p, s, ref, img = setup([face(embedding=(0, 1, 0))])
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert not result.success and not s.calls
    assert np.array_equal(result.image, img)


def test_single_person():
    p, s, ref, img = setup([face()])
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert result.success and result.replaced == 1
    assert len(s.calls) == 1
    assert result.reports[0].after < .3


def test_group_only_selected_face_changes():
    p, s, ref, img = setup([face(), face(150, embedding=(0, 1, 0))])
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert result.success and result.replaced == 1
    assert len(s.calls) == 1
    assert np.array_equal(result.image[40:110,150:220], img[40:110,150:220])
    assert np.array_equal(result.image[120:], img[120:])


def test_multiple_occurrences_same_fictional_identity():
    p, s, ref, img = setup([face(), face(150)])
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert result.success and result.replaced == 2 and len(s.calls) == 2
    assert all(r.accepted for r in result.reports)


def test_similar_faces_false_positive_threshold_and_manual_review():
    p, s, ref, img = setup([face(), face(150, embedding=(.44, .898, 0))])
    analysis = p.analyze(ref, img, Settings())
    assert analysis.selected == [0]
    # L'utilisateur peut désélectionner même une correspondance automatique.
    result = p.run(analysis, Settings(), selected=[])
    assert not result.success and not s.calls


def test_multiple_reference_faces_require_interactive_selection():
    p, s, ref, img = setup([face()], refs=[face(), face(embedding=(0,1,0))])
    with pytest.raises(SelectionError, match="Plusieurs"):
        p.analyze(ref, img, Settings())
    assert p.analyze(ref, img, Settings(), reference_index=1).selected == []


def test_no_reference_face():
    p, s, ref, img = setup([], refs=[])
    with pytest.raises(SelectionError, match="Aucun"):
        p.analyze(ref, img, Settings())


def test_sizes_and_orientations_pass_original_landmarks():
    faces = [face(size=40, angle=8), face(150, size=110, angle=-8)]
    p, s, ref, img = setup(faces)
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert result.success
    assert all(np.array_equal(a.landmarks, b.landmarks) for a,b in zip(s.calls, faces))


def test_high_resolution_preserves_framing():
    p, s, ref, img = setup([face()], shape=(2000, 3000, 3))
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert result.success and result.image.shape == img.shape
    assert np.array_equal(result.image[500:], img[500:])


def test_failed_validation_retries_another_identity():
    p, s, ref, img = setup([face()], bank=Bank(("weak", "strong")))
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert result.success and result.attempts == 2 and result.identity == "strong"


def test_failed_redetection_does_not_count_as_anonymization():
    p, s, ref, img = setup([face()], redetect=False)
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert not result.success and result.replaced == 0
    assert np.array_equal(result.image, img)


def test_all_candidates_fail_original_preserved():
    p, s, ref, img = setup([face()], bank=Bank(("weak",)))
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert not result.success and result.replaced == 0
    assert np.array_equal(result.image, img)


def test_cuda_error_is_explicit_and_preserves_input():
    p, s, ref, img = setup([face()])
    def fail(*args, **kwargs):
        raise ModelError("CUDA out of memory")
    s.swap = fail
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert not result.success and result.replaced == 0
    assert any("CUDA" in w for w in result.warnings)
    assert np.array_equal(result.image, img)


def test_best_valid_attempt_kept_if_later_attempt_has_gpu_failure():
    p, s, ref, img = setup([face()], bank=Bank(("strong", "second")))
    original_swap = s.swap
    def swap_or_fail(*args, **kwargs):
        if s.calls:
            raise ModelError("CUDA out of memory during second attempt")
        return original_swap(*args, **kwargs)
    s.swap = swap_or_fail
    result = p.run(p.analyze(ref, img, Settings()), Settings())
    assert result.success and result.identity == "strong"
    assert result.replaced == 1 and result.attempts == 2


def test_invalid_selection_refused():
    p, s, ref, img = setup([face()])
    with pytest.raises(SelectionError):
        p.run(p.analyze(ref,img,Settings()), Settings(), selected=[5])


def test_overlap_preserves_nonselected_face_pixels_exactly():
    p, s, ref, img = setup([face(size=100), face(85, y=40, embedding=(0,1,0))])
    result = p.run(p.analyze(ref,img,Settings()), Settings())
    assert np.array_equal(result.image[40:110,85:155], img[40:110,85:155])
