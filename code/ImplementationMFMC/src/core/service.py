"""Instances réutilisées, initialisation tardive ; aucune dépendance graphique."""
from .pipeline import Pipeline
from .identity_bank import IdentityBank


class LazySwapper:
    def __init__(self, settings):
        self.settings = settings
        self.instance = None

    @property
    def notices(self):
        return self.instance.notices if self.instance else []

    def swap(self, *args, **kwargs):
        if self.instance is None:
            from ..backends.simswap_backend import SimSwapBackend
            self.instance = SimSwapBackend(self.settings)
        return self.instance.swap(*args, **kwargs)


class ProcessingService:
    def __init__(self):
        self._recognizer = None
        self._key = None
        self._pipeline = None

    def pipeline(self, settings):
        key = (settings.root, settings.device, settings.crop_size, settings.segmentation)
        if self._pipeline is None or key != self._key:
            from ..backends.insightface_backend import InsightFaceBackend
            # Un seul générateur actif à la fois, pour la VRAM de la RTX 3060.
            old = self._pipeline
            session = old._session_identities if old else []
            self._pipeline = None
            if old:
                old.swapper = None
            if self._recognizer is None or self._key[:2] != key[:2]:
                self._recognizer = InsightFaceBackend(settings)
            self._pipeline = Pipeline(self._recognizer, LazySwapper(settings), IdentityBank(settings.bank, self._recognizer))
            self._pipeline._session_identities = session
            self._key = key
        return self._pipeline

    def release_swapper(self):
        if self._pipeline:
            self._pipeline.swapper = LazySwapper(self._pipeline.swapper.settings)

    def clear_session(self):
        if self._pipeline:
            self._pipeline.clear_session()
