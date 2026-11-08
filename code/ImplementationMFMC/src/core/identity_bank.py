from dataclasses import dataclass
from pathlib import Path
import json
import random
import uuid
import cv2
import numpy as np
from .detection import Face
from .errors import BankError
from .recognition import cosine, normalized
from ..utils.image_utils import load_image, save_image, EXTENSIONS
from ..utils.model_manager import sha256


@dataclass
class Identity:
    key: str
    path: Path
    face: Face
    luminance: float
    digest: str


class IdentityBank:
    """Seules les identités attestées synthétiques sont persistées, jamais les références réelles."""
    def __init__(self, path, recognizer):
        self.path = Path(path)
        self.recognizer = recognizer
        self.manifest = self.path / "bank.json"
        self.entries = []

    def _read(self):
        if not self.manifest.exists():
            return {"version": 1, "identities": []}
        try:
            data = json.loads(self.manifest.read_text(encoding="utf-8"))
            if data.get("version") != 1 or not isinstance(data.get("identities"), list):
                raise ValueError("Format de banque inconnu")
            return data
        except (ValueError, OSError) as exc:
            raise BankError(f"Banque illisible : {exc}") from exc

    def _write(self, data):
        self.path.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.manifest)

    def load(self):
        self.entries = []
        for item in self._read()["identities"]:
            if item.get("synthetic") is not True:
                raise BankError("Une entrée de banque n'est pas déclarée synthétique")
            if item["model_id"] != self.recognizer.model_id:
                raise BankError("Embeddings de banque incompatibles : utiliser bank-reindex avec le modèle courant")
            path = (self.path / item["file"]).resolve()
            if not path.is_relative_to(self.path.resolve()) or not path.is_file():
                raise BankError("Image synthétique manquante ou chemin invalide")
            face = Face(np.float32(item["bbox"]), np.float32(item["landmarks"]), normalized(item["embedding"]))
            self.entries.append(Identity(item["key"], path, face, item["luminance"], item["sha256"]))
        return self.entries

    def _record_image(self, image, *, provenance, seed=None):
        faces = self.recognizer.detect(image)
        if len(faces) != 1:
            raise BankError(f"Une identité doit contenir exactement un visage détectable ({len(faces)} détecté(s))")
        face = faces[0]
        key = uuid.uuid4().hex[:16]
        path = self.path / f"{key}.png"
        self.path.mkdir(parents=True, exist_ok=True)
        save_image(path, image)
        return {"key": key, "file": path.name, "synthetic": True, "provenance": provenance,
                "seed": seed, "sha256": sha256(path), "model_id": self.recognizer.model_id,
                "bbox": face.bbox.tolist(), "landmarks": face.landmarks.tolist(),
                "embedding": normalized(face.embedding).tolist(), "luminance": float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())}

    def import_folder(self, folder, *, synthetic_confirmed=False, progress=lambda p, m: None):
        if not synthetic_confirmed:
            raise BankError("Confirmer explicitement que les portraits importés sont fictifs")
        folder = Path(folder)
        if folder.resolve() == self.path.resolve():
            raise BankError("Choisir un dossier d'import différent de la banque gérée")
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise BankError("Aucune image PNG/JPEG/WebP dans le dossier")
        data = self._read()
        if data["identities"]:
            self.load()  # ne jamais mélanger deux espaces d'embeddings
        rejected = []
        count = 0
        for i, path in enumerate(files):
            try:
                record = self._record_image(load_image(path), provenance="Import attesté synthétique par l'utilisateur")
                data["identities"].append(record)
                self._write(data)
                count += 1
            except (BankError, ValueError, OSError) as exc:
                rejected.append(f"{path.name} : {exc}")
            progress(int((i+1)*100/len(files)), f"Import : {i+1}/{len(files)}")
        self.load()
        return count, rejected

    def generate(self, backend, seeds, progress=lambda p, m: None):
        data = self._read()
        if data["identities"]:
            self.load()
        seeds = list(seeds)
        rejected = []
        count = 0
        for i, seed in enumerate(seeds):
            try:
                image = backend.generate(seed)
                record = self._record_image(image, provenance="StyleGAN2-ADA FFHQ officiel", seed=seed)
                data["identities"].append(record)
                self._write(data)
                count += 1
            except BankError as exc:
                rejected.append(f"Seed {seed} : {exc}")
            progress(int((i+1)*100/max(1,len(seeds))), f"StyleGAN2 : {i+1}/{len(seeds)}")
        self.load()
        return count, rejected

    def reindex(self, progress=lambda p, m: None):
        data = self._read()
        updates = []
        for i, item in enumerate(data["identities"]):
            if item.get("synthetic") is not True:
                raise BankError("Banque non synthétique refusée")
            path = (self.path/item["file"]).resolve()
            if not path.is_relative_to(self.path.resolve()):
                raise BankError("Chemin de banque invalide")
            image = load_image(path)
            faces = self.recognizer.detect(image)
            if len(faces) != 1:
                raise BankError(f"{path.name} : un visage exactement est requis")
            face = faces[0]
            updates.append(dict(item, model_id=self.recognizer.model_id, embedding=normalized(face.embedding).tolist(),
                                bbox=face.bbox.tolist(), landmarks=face.landmarks.tolist(), sha256=sha256(path)))
            progress(int((i+1)*100/max(1,len(data["identities"]))), "Réindexation de la banque")
        data["identities"] = updates
        self._write(data)
        return self.load()

    def candidates(self, original_embeddings, maximum, manual_key=None, luminance=None, rng=None):
        self.load()
        candidates = [e for e in self.entries if all(cosine(e.face.embedding, original) <= maximum
                                                     for original in original_embeddings)]
        if manual_key:
            candidates = [e for e in candidates if e.key == manual_key]
        if not candidates:
            raise BankError("Aucune identité synthétique admissible : banque vide, choix manuel trop proche ou seuil trop strict")
        rng = rng or random.SystemRandom()
        rng.shuffle(candidates)
        if luminance is not None:
            # Pondération facultative sur la luminosité, sans inférer âge, genre ou identité.
            candidates.sort(key=lambda e: abs(e.luminance-luminance))
            top = candidates[:max(3, len(candidates)//2)]
            rng.shuffle(top)
            candidates = top + candidates[len(top):]
        return candidates

    def image(self, identity):
        if sha256(identity.path) != identity.digest:
            raise BankError("Portrait de banque modifié : réindexer avant utilisation")
        return load_image(identity.path)
