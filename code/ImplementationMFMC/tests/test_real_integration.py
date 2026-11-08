"""Test d'intégration avec les modèles préentraînés et des photographies locales.

FACEANON_INTEGRATION_CONFIG doit désigner un JSON basé sur integration.example.json.
Le test est ignoré lorsque la configuration n'est pas fournie.
"""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from src.utils.config import Settings
from src.utils.image_utils import load_image, save_image
from src.utils.model_manager import ModelManager
from src.core.service import ProcessingService


@pytest.mark.integration
@pytest.mark.parametrize("crop_size", [224, 512])
def test_real_group_anonymization(tmp_path, crop_size):
    config_path = os.environ.get("FACEANON_INTEGRATION_CONFIG")
    if not config_path:
        pytest.skip("Checkpoints et photographies réels requis : définir FACEANON_INTEGRATION_CONFIG")
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config.get("synthetic_donor_confirmed") is True, "Le donneur d'identité doit être attesté synthétique"
    def image_path(key):
        p = Path(config[key])
        return p if p.is_absolute() else config_path.parent/p
    settings = Settings(device=config.get("device", "auto"),
                        crop_size=crop_size,
                        match_threshold=config.get("match_threshold", .45),
                        validation_max_similarity=config.get("validation_max_similarity", .30),
                        independent_validation_max_similarity=config.get("independent_validation_max_similarity", .30),
                        candidate_max_similarity=config.get("candidate_max_similarity", .30),
                        max_attempts=config.get("max_attempts", 3),
                        segmentation=config.get("segmentation", True))
    pipeline = ProcessingService().pipeline(settings)
    from src.core.identity_bank import IdentityBank
    pipeline.bank = IdentityBank(tmp_path/"synthetic_bank", pipeline.recognizer)
    donor = load_image(image_path("synthetic_donor"))
    record = pipeline.bank._record_image(donor, provenance="Fixture d'intégration attestée synthétique")
    pipeline.bank._write({"version": 1, "identities": [record]})
    if config.get("independent_validation", False):
        from src.backends.insightface_backend import InsightFaceBackend
        pipeline.independent_recognizer = InsightFaceBackend(settings, ModelManager(settings).require("independent"))
        assert pipeline.independent_recognizer.model_id != pipeline.recognizer.model_id
    reference = load_image(image_path("reference"))
    image = load_image(image_path("group"))
    analysis = pipeline.analyze(reference, image, settings, config.get("reference_index"))
    assert len(analysis.faces) >= 2, "Utiliser une photographie de groupe pour vérifier les visages conservés"
    assert len(analysis.selected) == config["expected_matches"]
    assert 0 < len(analysis.selected) < len(analysis.faces)
    result = pipeline.run(analysis, settings)
    assert result.success, " ; ".join(result.warnings)
    assert result.replaced == config["expected_matches"]
    assert result.image.shape == image.shape
    assert np.any(result.image != image), "La photographie doit effectivement changer"
    for i, face in enumerate(analysis.faces):
        if i not in analysis.selected:
            x1, y1, x2, y2 = face.bbox.astype(int)
            assert np.array_equal(result.image[max(0,y1):y2,max(0,x1):x2], image[max(0,y1):y2,max(0,x1):x2])
    assert all(report.accepted and report.after <= settings.validation_max_similarity for report in result.reports)
    if settings.segmentation:
        assert pipeline.swapper.instance.parser is not None, "Ce test exige le vrai parsing BiSeNet"
        assert not any("géométrique" in message for message in result.warnings)
    output = tmp_path/"real_result.png"
    save_image(output, result.image, [image_path("group"), image_path("reference")])
    assert np.array_equal(load_image(output), result.image)

