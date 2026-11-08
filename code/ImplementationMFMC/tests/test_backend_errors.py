"""Les poids incompatibles arrêtent le backend avant toute génération."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from src.backends.simswap_backend import SimSwapBackend
from src.core.errors import ModelError
from src.utils.config import Settings


def test_incompatible_generator_checkpoint_never_reaches_forward():
    class Generator:
        def __init__(self, *args, **kwargs):
            pass

        def load_state_dict(self, state, strict):
            assert strict is True
            raise RuntimeError("size mismatch in generator checkpoint")

        def forward(self, *args):
            pytest.fail("Un checkpoint incompatible ne doit jamais être utilisé")

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), load=lambda *a, **k: {})
    module = SimpleNamespace(Generator_Adain_Upsample=Generator)
    align = SimpleNamespace(estimate_norm=lambda *a: None)
    with patch.dict("sys.modules", {"torch": torch}), \
         patch("src.backends.simswap_backend.ModelManager") as manager, \
         patch("src.backends.simswap_backend.importlib.import_module", side_effect=[module, align]):
        manager.return_value.require.return_value = "placeholder"
        with pytest.raises(ModelError, match="size mismatch"):
            SimSwapBackend(Settings(device="cpu"))
