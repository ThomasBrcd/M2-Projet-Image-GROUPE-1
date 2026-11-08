"""Tests sans dépendance ML, exécutables aussi via unittest."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io
import zipfile
from src.utils.config import Settings
from src.utils.model_manager import ModelManager, SPECS
from src.core.errors import ModelError


class TestModelManager(unittest.TestCase):
    def setUp(self):
        folder = Settings().root/".cache/tests"
        folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=folder)
        self.manager = ModelManager(Settings(root=Path(self.temporary.name)))

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_checkpoint(self):
        with self.assertRaisesRegex(ModelError, "absent"):
            self.manager.require("simswap224")

    def test_truncated_checkpoint(self):
        path = self.manager.path("simswap224")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"bad")
        with self.assertRaisesRegex(ModelError, "tronqué"):
            self.manager.require("simswap224")

    def test_modified_checkpoint(self):
        source = Path(self.temporary.name)/"source.pth"
        source.write_bytes(b"x"*2048)
        self.manager.import_file("simswap224", source)
        self.manager.path("simswap224").write_bytes(b"y"*2048)
        with self.assertRaisesRegex(ModelError, "corrompu"):
            self.manager.require("simswap224")

    def test_pickle_requires_acknowledgment(self):
        source = Path(self.temporary.name)/"source.tar"
        source.write_bytes(b"x"*2048)
        with self.assertRaises(ModelError):
            self.manager.import_file("arcface", source)
        self.manager.import_file("arcface", source, trusted_pickle=True)
        self.assertTrue(self.manager.require("arcface").is_file())

    def test_pickle_copied_without_registry_is_refused(self):
        path = self.manager.path("arcface")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x"*2048)
        with self.assertRaisesRegex(ModelError, "pickle"):
            self.manager.require("arcface")

    def test_download_html_rejected(self):
        class Response(io.BytesIO):
            headers = {}
        with patch("urllib.request.urlopen", return_value=Response(b"<html>" + b"x"*2048)):
            with self.assertRaisesRegex(ModelError, "HTML"):
                self.manager.download("bisenet")
        self.assertFalse(self.manager.path("bisenet").exists())

    def test_zip_cannot_extract_other_members(self):
        class Response(io.BytesIO):
            headers = {}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipped:
            zipped.writestr("../../evil.py", "bad")
            zipped.writestr("models/det_10g.onnx", b"x"*2048)
        with patch("urllib.request.urlopen", return_value=Response(buffer.getvalue())):
            path = self.manager.download("detector")
        self.assertEqual(path.read_bytes(), b"x"*2048)
        self.assertFalse((Path(self.temporary.name)/"evil.py").exists())

    def test_invalid_configuration(self):
        for options in ({"crop_size": 128}, {"max_attempts": 0}, {"match_threshold": 2}, {"device": "mps"}):
            with self.assertRaises(ValueError):
                Settings(**options)


if __name__ == "__main__":
    unittest.main()
