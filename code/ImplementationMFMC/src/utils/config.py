from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    device: str = "auto"
    crop_size: int = 224
    match_threshold: float = 0.45
    candidate_max_similarity: float = 0.30
    validation_max_similarity: float = 0.30
    independent_validation_max_similarity: float = 0.30
    max_attempts: int = 3
    segmentation: bool = True
    photometric: bool = True
    identity_mode: str = "random"
    manual_identity: str | None = None
    attribute_matching: bool = False

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device : auto, cpu ou cuda")
        if self.crop_size not in {224, 512}:
            raise ValueError("SimSwap : 224 ou 512")
        if self.identity_mode not in {"random", "session", "manual"}:
            raise ValueError("Mode d'identité inconnu")
        for x in (self.match_threshold, self.candidate_max_similarity, self.validation_max_similarity,
                  self.independent_validation_max_similarity):
            if not -1 <= x <= 1:
                raise ValueError("Les seuils cosinus doivent être compris entre -1 et 1")
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("Entre 1 et 10 tentatives")

    @property
    def models(self):
        return self.root / "models"

    @property
    def bank(self):
        return self.root / "identity_bank"

    @property
    def simswap(self):
        return self.root / "vendor" / "SimSwap"

    @property
    def stylegan(self):
        return self.root / "vendor" / "stylegan2-ada-pytorch"
