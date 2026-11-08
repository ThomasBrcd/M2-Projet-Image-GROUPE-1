"""Inférence des réseaux OFFICIELS SimSwap, sans le lanceur historique CUDA obligatoire."""
import importlib
from collections.abc import Mapping
import re
import sys
import threading
from pathlib import Path
import cv2
import numpy as np
from ..core.errors import ModelError
from ..core.face_swap import SwapPatch
from ..core.blending import geometric_mask
from ..utils.model_manager import ModelManager

_LOAD_LOCK = threading.Lock()
_LEGACY_512_KEY = re.compile(
    r"BottleNeck\.[0-8]\.style[12]\."
    r"(?:conv_ada[12]\.(?:weight|bias)|"
    r"bn_ada1\.(?:weight|bias|running_mean|running_var|num_batches_tracked))"
)


def add_official_source(path):
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def choose_device(torch, requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ModelError("CUDA demandé mais indisponible pour PyTorch")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


def load_generator_weights(generator, state, crop_size):
    """Charge tous les poids actifs strictement, avec la compatibilité du checkpoint 512 beta.

    Le BaseModel officiel filtre les couches excédentaires à l'inférence. Ici, seuls les
    anciens conv_ada/bn_ada connus du 512 peuvent être ignorés : aucune couche active ne
    peut rester initialisée aléatoirement et les autres clés inattendues sont refusées.
    """
    if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
        raise ModelError(f"Checkpoint SimSwap {crop_size} invalide : dictionnaire de paramètres attendu")
    ignored = []
    if crop_size == 512:
        expected = generator.state_dict()
        ignored = sorted(set(state)-set(expected))
        unknown = [key for key in ignored if not _LEGACY_512_KEY.fullmatch(key)]
        if unknown:
            raise ModelError(f"Checkpoint SimSwap 512 incompatible : paramètres inconnus : {', '.join(unknown[:5])}")
        state = {key: value for key, value in state.items() if key in expected}
    try:
        generator.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ModelError(f"Checkpoint SimSwap {crop_size} incompatible avec le générateur : {exc}") from exc
    return ignored


class SimSwapBackend:
    def __init__(self, settings):
        self.settings = settings
        self.size = settings.crop_size
        self.notices = []
        self.parser = None
        manager = ModelManager(settings)
        generator_path = manager.require(f"simswap{self.size}")
        arcface_path = manager.require("arcface")
        if not (settings.simswap / "models/fs_networks.py").exists():
            raise ModelError("Sources officielles SimSwap absentes de vendor/SimSwap")
        try:
            import torch
            self.torch = torch
            self.device = choose_device(torch, settings.device)
            add_official_source(settings.simswap)
            module = importlib.import_module("models.fs_networks" if self.size == 224 else "models.fs_networks_512")
            align = importlib.import_module("insightface_func.utils.face_align_ffhqandnewarc")
            self.estimate_norm = align.estimate_norm
            # Architecture, noms de paramètres et sortie [0,1] identiques au fs_model officiel.
            self.generator = module.Generator_Adain_Upsample(3, 3, latent_size=512, n_blocks=9, deep=False)
            state = torch.load(generator_path, map_location="cpu", weights_only=True)
            ignored = load_generator_weights(self.generator, state, self.size)
            if ignored:
                self.notices.append(
                    f"SimSwap 512 : {len(ignored)} paramètres historiques hors du réseau d'inférence "
                    "écartés selon le chargement officiel ; tous les poids actifs sont chargés strictement."
                )
            self.generator = self.generator.eval().requires_grad_(False).to(self.device)
            # L'ArcFace historique est un module sérialisé, distinct du reconnaisseur ONNX.
            # Autorisation explicite enregistrée par ModelManager avant tout torch.load non restreint.
            self.arcface = torch.load(arcface_path, map_location="cpu", weights_only=False)
            if not isinstance(self.arcface, torch.nn.Module):
                raise ModelError("Le checkpoint ArcFace SimSwap doit contenir le module officiel sérialisé")
            self.arcface = self.arcface.eval().requires_grad_(False).to(self.device)
            self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)
            if settings.segmentation:
                self._load_parser(manager)
            self.notices.append(f"SimSwap {self.size} : {self.device}")
        except ModelError:
            raise
        except Exception as exc:
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                raise ModelError(f"Mémoire insuffisante au chargement de SimSwap {self.size} : {exc}. "
                                 "Libérer de la VRAM ou choisir le mode CPU.") from exc
            raise ModelError(f"Chargement SimSwap {self.size} impossible : {exc}") from exc

    def _load_parser(self, manager):
        try:
            path = manager.require("bisenet")
            # Resnet18.init_weight télécharge sinon des poids inutiles. Le checkpoint BiSeNet
            # complet les remplace strictement : aucun accès réseau n'est nécessaire en inférence.
            with _LOAD_LOCK:
                module = importlib.import_module("parsing_model.model")
                original = module.Resnet18.init_weight
                try:
                    module.Resnet18.init_weight = lambda self: None
                    parser = module.BiSeNet(n_classes=19)
                finally:
                    module.Resnet18.init_weight = original
            state = self.torch.load(path, map_location="cpu", weights_only=True)
            parser.load_state_dict(state, strict=True)
            self.parser = parser.eval().requires_grad_(False).to(self.device)
        except Exception as exc:
            self.parser = None
            self.notices.append(f"Segmentation indisponible : {exc}. Masque géométrique ; qualité réduite.")

    def _tensor(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(self.device) / 255

    def _align(self, bgr, face):
        matrix, _ = self.estimate_norm(face.landmarks, self.size, mode="None" if self.size == 224 else "ffhq")
        crop = cv2.warpAffine(bgr, matrix, (self.size, self.size), borderValue=0)
        return crop, matrix

    def swap(self, image, face, donor_image, donor_face, segmentation=True):
        torch = self.torch
        F = torch.nn.functional
        warnings = []
        try:
            target, matrix = self._align(image, face)
            donor, _ = self._align(donor_image, donor_face)
            with torch.inference_mode():
                source = self._tensor(donor)
                arc_input = F.interpolate((source-self.mean)/self.std, (112, 112), mode="nearest")
                identity = F.normalize(self.arcface(arc_input), p=2, dim=1)
                if identity.shape != (1, 512) or not torch.isfinite(identity).all():
                    raise ModelError("Embedding ArcFace SimSwap incompatible : attendu (1, 512)")
                target_tensor = self._tensor(target)
                generated = self.generator(target_tensor, identity)
                if generated.shape != target_tensor.shape or not torch.isfinite(generated).all():
                    raise ModelError("Sortie SimSwap invalide")
                # Le réseau officiel effectue déjà (tanh(x)+1)/2 : pas de seconde conversion.
                rgb = generated[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                bgr = np.rint(rgb[..., ::-1]*255).astype(np.uint8)
                mask = geometric_mask(self.size)
                if segmentation and self.parser is not None:
                    try:
                        inp = F.interpolate((target_tensor-self.mean)/self.std, (512, 512), mode="nearest")
                        labels = self.parser(inp)[0].argmax(1)[0].cpu().numpy()
                        # Classes du reverse2original officiel ; cheveux/cou/fond/occlusions exclus.
                        binary = np.isin(labels, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]).astype(np.float32)
                        if binary.sum() < 500:
                            raise ValueError("Région faciale trop petite")
                        binary = cv2.resize(binary, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                        eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
                        soft = cv2.GaussianBlur(eroded, (0, 0), self.size * .015)
                        # Aucune diffusion dans les pixels étiquetés cheveux/fond/occlusion.
                        mask = mask * soft * binary
                    except Exception as exc:
                        warnings.append(f"Échec du parsing : {exc}. Masque géométrique ; qualité réduite.")
                elif segmentation:
                    warnings.append("BiSeNet absent : masque géométrique ; qualité réduite.")
                return SwapPatch(bgr, mask.astype(np.float32), matrix, warnings)
        except ModelError:
            raise
        except Exception as exc:
            hint = " Libérer la VRAM ou relancer avec --device cpu." if "memory" in str(exc).lower() else ""
            raise ModelError(f"Échec d'inférence SimSwap/CUDA : {exc}.{hint}") from exc
