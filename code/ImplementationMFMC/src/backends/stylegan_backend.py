"""Génération indépendante de SimSwap, avec le réseau FFHQ officiel."""
import numpy as np
from ..core.errors import ModelError
from ..utils.model_manager import ModelManager
from .simswap_backend import add_official_source, choose_device


class StyleGANBackend:
    def __init__(self, settings):
        path = ModelManager(settings).require("stylegan")
        if not (settings.stylegan / "legacy.py").exists():
            raise ModelError("Sources officielles StyleGAN2-ADA absentes de vendor/stylegan2-ada-pytorch")
        try:
            import torch
            self.torch = torch
            self.device = choose_device(torch, settings.device)
            add_official_source(settings.stylegan)
            import legacy
            # Pickle autorisé explicitement par le gestionnaire. Aucun URL dans load_network_pkl.
            with path.open("rb") as stream:
                self.generator = legacy.load_network_pkl(stream)["G_ema"].eval().requires_grad_(False).to(self.device)
        except Exception as exc:
            raise ModelError(f"Chargement StyleGAN2 impossible : {exc}") from exc

    def generate(self, seed, truncation=.7):
        try:
            torch = self.torch
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.generator.z_dim)).float().to(self.device)
            with torch.inference_mode():
                rgb = self.generator(z, None, truncation_psi=truncation, noise_mode="const", force_fp32=True)
            rgb = (rgb[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            return rgb[..., ::-1].copy()
        except Exception as exc:
            raise ModelError(f"Génération StyleGAN2 impossible (CUDA Toolkit/MSVC requis sur GPU) : {exc}") from exc
