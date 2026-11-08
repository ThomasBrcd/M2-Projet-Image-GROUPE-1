"""Installation explicite uniquement. Aucun téléchargement depuis un backend d'inférence."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from .config import Settings
from ..core.errors import ModelError


@dataclass(frozen=True)
class ModelSpec:
    key: str
    relative_path: str
    source: str
    download: str | None = None
    archive_member: str | None = None
    pickle: bool = False
    minimum_bytes: int = 1024
    sha256: str | None = None  # Aucun hash éditeur inventé.


SIMSWAP_RELEASE = "https://github.com/neuralchen/SimSwap/releases/download/1.0/"
BUFFALO = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
SPECS = {
    "detector": ModelSpec("detector", "insightface/det_10g.onnx", BUFFALO, BUFFALO, "det_10g.onnx"),
    "recognizer": ModelSpec("recognizer", "insightface/w600k_r50.onnx", BUFFALO, BUFFALO, "w600k_r50.onnx"),
    "arcface": ModelSpec("arcface", "simswap/arcface_checkpoint.tar", SIMSWAP_RELEASE,
                         SIMSWAP_RELEASE + "arcface_checkpoint.tar", pickle=True),
    "simswap224": ModelSpec("simswap224", "simswap/224/latest_net_G.pth", SIMSWAP_RELEASE,
                            SIMSWAP_RELEASE + "checkpoints.zip", "latest_net_G.pth"),
    "simswap512": ModelSpec("simswap512", "simswap/512/550000_net_G.pth",
                            "https://github.com/neuralchen/SimSwap/releases/tag/512_beta",
                            "https://github.com/neuralchen/SimSwap/releases/download/512_beta/512.zip",
                            "550000_net_G.pth"),
    "bisenet": ModelSpec("bisenet", "parsing/79999_iter.pth", SIMSWAP_RELEASE,
                         SIMSWAP_RELEASE + "79999_iter.pth"),
    "stylegan": ModelSpec("stylegan", "stylegan/ffhq.pkl",
                          "https://github.com/NVlabs/stylegan2-ada-pytorch",
                          "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
                          pickle=True),
    "independent": ModelSpec("independent", "evaluation/recognizer.onnx",
                             "Modèle indépendant, prétraitement ArcFace ONNX documenté ; installation manuelle"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelManager:
    def __init__(self, settings=Settings()):
        self.settings = settings
        self.registry_path = settings.models / "provenance.json"

    def path(self, key):
        return self.settings.models / SPECS[key].relative_path

    def _registry(self):
        if not self.registry_path.exists():
            return {}
        try:
            registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if not isinstance(registry, dict) or any(not isinstance(record, dict) or "sha256" not in record
                                                     for record in registry.values()):
                raise ValueError("Format du registre incorrect")
            return registry
        except (ValueError, OSError) as exc:
            raise ModelError(f"Registre de provenance illisible : {exc}") from exc

    def _record(self, key, origin, trust_pickle=False):
        record = self._registry()
        record[key] = {"sha256": sha256(self.path(key)), "origin": origin,
                       "pickle_acknowledged": bool(trust_pickle)}
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.registry_path)

    def require(self, key):
        spec = SPECS[key]
        path = self.path(key)
        if not path.is_file():
            raise ModelError(f"Checkpoint absent : {path}. Gestionnaire des modèles → {key}.")
        if path.stat().st_size < spec.minimum_bytes:
            raise ModelError(f"Checkpoint vide ou tronqué : {path}")
        digest = sha256(path)
        if spec.sha256 and digest != spec.sha256:
            raise ModelError(f"Somme de contrôle éditeur incorrecte : {key}")
        record = self._registry().get(key)
        if record and record["sha256"] != digest:
            raise ModelError(f"Checkpoint modifié/corrompu depuis son installation : {key}")
        if spec.pickle and (not record or not record.get("pickle_acknowledged")):
            raise ModelError(f"{key} utilise pickle : importer depuis une source officielle et accepter explicitement son chargement.")
        return path

    def status(self):
        rows = []
        for key, spec in SPECS.items():
            try:
                self.require(key)
                state = "Présent — hash éditeur vérifié" if spec.sha256 else "Présent — pas de hash éditeur disponible"
            except ModelError as exc:
                state = str(exc)
            rows.append({"key": key, "path": str(self.path(key)), "status": state, "source": spec.source})
        return rows

    def import_file(self, key, source, *, trusted_pickle=False):
        spec = SPECS[key]
        if spec.pickle and not trusted_pickle:
            raise ModelError("Consentement au chargement pickle requis pour ce checkpoint officiel")
        source = Path(source).resolve()
        if not source.is_file() or source.stat().st_size < spec.minimum_bytes:
            raise ModelError("Checkpoint absent ou trop petit")
        destination = self.path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != source:
            shutil.copyfile(source, destination)
        self._record(key, f"Import manuel : {source.name}", trusted_pickle)
        return self.require(key)

    def download(self, key, progress=lambda percent, message: None, *, trusted_pickle=False):
        """Fichiers officiels uniquement ; extraction d'un seul membre, sans extractall."""
        spec = SPECS[key]
        if not spec.download:
            raise ModelError("Installation manuelle nécessaire pour ce modèle")
        if spec.pickle and not trusted_pickle:
            raise ModelError("Accepter explicitement le risque pickle du checkpoint officiel")
        destination = self.path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=self.settings.models) as folder:
            archive = Path(folder) / "payload"
            request = urllib.request.Request(spec.download, headers={"User-Agent": "FaceAnon/0.1"})
            progress(0, f"Téléchargement officiel : {key}")
            try:
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
                    total = int(response.headers.get("Content-Length", 0))
                    received = 0
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        received += len(block)
                        progress(min(95, int(received * 95 / total)) if total else 0,
                                 f"{key} : {received / 1024**2:.1f} Mo")
                payload = archive
                if spec.archive_member:
                    with zipfile.ZipFile(archive) as zipped:
                        matches = [n for n in zipped.namelist() if Path(n).name == spec.archive_member]
                        # L'archive peut contenir plusieurs variantes ; on exige le chemin people pour le 224.
                        if len(matches) > 1 and key == "simswap224":
                            matches = [n for n in matches if "people" in Path(n).parts]
                        if len(matches) != 1:
                            raise ModelError(f"Archive ambiguë : installer manuellement {spec.archive_member}")
                        payload = Path(folder) / "checkpoint"
                        with zipped.open(matches[0]) as source, payload.open("wb") as target:
                            shutil.copyfileobj(source, target)
                if payload.stat().st_size < spec.minimum_bytes:
                    raise ModelError("Le serveur n'a pas fourni un checkpoint valide")
                with payload.open("rb") as source:
                    prefix = source.read(100).lstrip().lower()
                if prefix.startswith((b"<!doctype", b"<html")):
                    raise ModelError("Une page HTML a été reçue à la place du modèle ; installation manuelle requise")
                if spec.sha256 and sha256(payload) != spec.sha256:
                    raise ModelError("Hash éditeur incorrect")
                # Un fichier déplacé depuis TemporaryDirectory conserve ses ACL restrictives
                # sous Windows. Créer le fichier final dans son dossier de destination pour
                # hériter des droits du projet, puis le publier atomiquement.
                staged = destination.with_name(destination.name + ".part")
                try:
                    shutil.copyfile(payload, staged)
                    os.replace(staged, destination)
                finally:
                    staged.unlink(missing_ok=True)
                self._record(key, spec.download, trusted_pickle)
                progress(100, f"{key} installé — compatibilité à vérifier par inférence")
            except Exception as exc:
                if isinstance(exc, ModelError):
                    raise
                raise ModelError(f"Téléchargement {key} impossible : {exc}. Source : {spec.source}") from exc
        return destination


def diagnostics():
    report = {}
    import sys
    report["python"] = sys.version
    report["executable"] = sys.executable
    for name in ("numpy", "torch", "torchvision", "PySide6", "insightface", "onnxruntime", "opencv-python-headless"):
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report[name] = "ABSENT"
    compatibility_warnings = []
    if sys.version_info[:2] != (3, 10):
        compatibility_warnings.append("Le profil fourni cible CPython 3.10")
    requirements = Settings().root / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            if "==" not in line or line.lstrip().startswith("#"):
                continue
            name, expected = line.strip().split("==", 1)
            try:
                actual = importlib.metadata.version(name).split("+", 1)[0]
                if actual != expected:
                    compatibility_warnings.append(f"{name} : {actual}, profil attendu {expected}")
            except importlib.metadata.PackageNotFoundError:
                compatibility_warnings.append(f"{name} absent")
    report["compatibility_warnings"] = compatibility_warnings
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            report["vram_free_MiB"] = free // 1024**2
            report["vram_total_MiB"] = total // 1024**2
            # Exécute réellement une opération CUDA, distincte du simple driver nvidia-smi.
            report["cuda_tensor_test"] = float((torch.ones(1, device="cuda") + 1).cpu()[0]) == 2
    except Exception as exc:
        report["torch_error"] = str(exc)
    return report
