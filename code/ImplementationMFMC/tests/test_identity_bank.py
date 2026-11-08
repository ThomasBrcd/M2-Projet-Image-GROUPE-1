from pathlib import Path
import json
import numpy as np
import pytest
from src.core.identity_bank import IdentityBank
from src.core.errors import BankError
from tests.test_pipeline import Recognizer, face


def manifest_entry(key, embedding, model="fake-model-unit-tests"):
    f = face(embedding=embedding)
    return {"key":key,"file":key+".png","synthetic":True,"model_id":model,"embedding":f.embedding.tolist(),
            "bbox":f.bbox.tolist(),"landmarks":f.landmarks.tolist(),"luminance":100.,"sha256":"unused"}


def make_bank(tmp_path, entries):
    for entry in entries:
        (tmp_path/entry["file"]).write_bytes(b"placeholder-not-read-by-selection-test")
    (tmp_path/"bank.json").write_text(json.dumps({"version":1,"identities":entries}),encoding="utf-8")
    return IdentityBank(tmp_path, Recognizer([]))


def test_too_similar_synthetic_identity_excluded(tmp_path):
    bank = make_bank(tmp_path, [manifest_entry("near",(1,0,0)), manifest_entry("far",(0,0,1))])
    assert [e.key for e in bank.candidates([np.array([1,0,0])], .3)] == ["far"]
    with pytest.raises(BankError):
        bank.candidates([np.array([1,0,0])], .3, manual_key="near")


def test_bank_embeddings_cannot_be_reused_with_another_model(tmp_path):
    bank = make_bank(tmp_path,[manifest_entry("a",(1,0,0),model="other-model")])
    with pytest.raises(BankError,match="incompatibles"):
        bank.load()


def test_import_requires_synthetic_attestation(tmp_path):
    with pytest.raises(BankError,match="Confirmer"):
        IdentityBank(tmp_path, Recognizer([])).import_folder(tmp_path)


def test_non_synthetic_entry_refused(tmp_path):
    entry = manifest_entry("a",(0,0,1))
    entry["synthetic"] = False
    bank = make_bank(tmp_path,[entry])
    with pytest.raises(BankError,match="synthétique"):
        bank.load()
