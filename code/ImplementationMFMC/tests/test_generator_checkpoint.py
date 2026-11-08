"""Compatibilité des checkpoints, sans accepter de poids actifs manquants."""
from collections import OrderedDict
import pytest
import torch
from src.backends.simswap_backend import load_generator_weights
from src.core.errors import ModelError


def network_and_state():
    generator = torch.nn.Linear(2, 3)
    # Poids déterministes distincts de l'initialisation, pour vérifier leur chargement effectif.
    state = OrderedDict(weight=torch.full((3, 2), .25), bias=torch.full((3,), .5))
    return generator, state


def legacy_parameters():
    state = {}
    for block in range(9):
        for style in (1, 2):
            prefix = f"BottleNeck.{block}.style{style}."
            for layer in ("conv_ada1", "conv_ada2", "bn_ada1"):
                for suffix in ("weight", "bias"):
                    state[prefix+layer+"."+suffix] = torch.ones(1)
            for suffix in ("running_mean", "running_var", "num_batches_tracked"):
                state[prefix+"bn_ada1."+suffix] = torch.ones(1)
    return state


def test_512_discards_only_known_unused_layers_and_loads_active_weights():
    generator, state = network_and_state()
    extras = legacy_parameters()
    state.update(extras)
    ignored = load_generator_weights(generator, state, 512)
    assert len(ignored) == 162
    assert set(ignored) == set(extras)
    torch.testing.assert_close(generator.weight, torch.full((3, 2), .25))
    torch.testing.assert_close(generator.bias, torch.full((3,), .5))
    assert set(extras) <= set(state)  # Le dictionnaire du checkpoint n'est pas modifié.


@pytest.mark.parametrize("crop_size", [224, 512])
def test_complete_checkpoint_without_extras_loads_strictly(crop_size):
    generator, state = network_and_state()
    assert load_generator_weights(generator, state, crop_size) == []
    torch.testing.assert_close(generator.weight, state["weight"])


def test_512_does_not_tolerate_missing_active_weights_even_with_legacy_extras():
    generator, state = network_and_state()
    state.pop("weight")
    state.update(legacy_parameters())
    with pytest.raises(ModelError, match="Missing key"):
        load_generator_weights(generator, state, 512)


def test_512_does_not_tolerate_incompatible_active_shapes():
    generator, state = network_and_state()
    state["weight"] = torch.ones(4, 2)
    state.update(legacy_parameters())
    with pytest.raises(ModelError, match="size mismatch"):
        load_generator_weights(generator, state, 512)


@pytest.mark.parametrize("unknown", ["other.weight", "BottleNeck.0.style1.linear.typo",
                                    "BottleNeck.9.style1.conv_ada1.weight",
                                    "BottleNeck.0.style3.bn_ada1.weight"])
def test_512_refuses_other_unknown_layers(unknown):
    generator, state = network_and_state()
    state[unknown] = torch.ones(1)
    with pytest.raises(ModelError, match="paramètres inconnus"):
        load_generator_weights(generator, state, 512)


def test_224_does_not_apply_512_compatibility_filter():
    generator, state = network_and_state()
    state.update(legacy_parameters())
    with pytest.raises(ModelError, match="Unexpected key"):
        load_generator_weights(generator, state, 224)


@pytest.mark.parametrize("state", [None, [], {1: torch.ones(1)}])
def test_invalid_checkpoint_format_is_refused(state):
    with pytest.raises(ModelError, match="dictionnaire"):
        load_generator_weights(torch.nn.Linear(2, 3), state, 512)
