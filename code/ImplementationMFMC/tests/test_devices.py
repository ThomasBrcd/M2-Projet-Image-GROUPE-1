import pytest
from src.backends.simswap_backend import choose_device
from src.core.errors import ModelError


class NoCuda:
    class cuda:
        @staticmethod
        def is_available():
            return False


def test_missing_cuda_has_clear_error():
    with pytest.raises(ModelError, match="CUDA"):
        choose_device(NoCuda, "cuda")
    assert choose_device(NoCuda, "auto") == "cpu"
    assert choose_device(NoCuda, "cpu") == "cpu"
