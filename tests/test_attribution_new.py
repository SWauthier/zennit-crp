"""Tests for ConditionalGradient covering parallel and sequential conditioning.

These complement test_attribution.py with multi-branch and sequential
conditioning scenarios on the same branching SimpleModel fixture.
"""

import pytest
import torch
import torch.nn as nn
from zennit.composites import EpsilonPlus

from zennit_crp.attribution import ConditionalGradient
from zennit_crp.helper import get_layer_names


class SimpleModel(nn.Module):
    """Branching model: two parallel linear layers merged via concatenation.

    Architecture::

        x -> layer1 -> y1 -|
        x -> layer2 -> y2 -+-> cat(y1,y2) -> layer3 -> out
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2, False)
        self.layer2 = nn.Linear(2, 2, False)
        self.layer3 = nn.Linear(4, 1, False)
        self.layer1.weight = nn.Parameter(torch.tensor([[1, 2], [0, 1]], dtype=torch.float32))
        self.layer2.weight = nn.Parameter(torch.tensor([[2, 3], [0, 0]], dtype=torch.float32))
        self.layer3.weight = nn.Parameter(torch.tensor([[1, 4, 5, 0]], dtype=torch.float32))

    def forward(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(x)
        return self.layer3(torch.cat([y1, y2], dim=1))


@pytest.fixture
def simple_model():
    return SimpleModel()


def test_parallel_attribution(simple_model):
    """Conditioning on one parallel branch leaves the other zeroed out."""
    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    composite = EpsilonPlus()
    layer_names = get_layer_names(simple_model, [nn.Linear])
    init_rel = torch.eye(1, dtype=torch.float32)[[0]]

    # Mask layer1 to channel 0 only, mask layer2 to no channels (empty list).
    conditions = [{"y": [0], "layer1": [0], "layer2": []}]
    with ConditionalGradient(simple_model, composite=composite) as attributor:
        attr = attributor(
            inp,
            conditions,
            record_layers=layer_names,
            init_rel=init_rel,
            heatmap_fn=lambda g: g.squeeze(0),
            exclude_parallel=False,
        )

    assert torch.allclose(attr.heatmap, torch.tensor([-0.1, 0.2]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([[0.1, 0.0]]), atol=1e-6)
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([[0.0, 0.0]]), atol=1e-6)

    # With exclude_parallel=True and only layer1 masked, layer2 path is cut.
    conditions_excl = [{"y": [0], "layer1": [0]}]
    with ConditionalGradient(simple_model, composite=composite) as attributor:
        attr_excl = attributor(
            inp,
            conditions_excl,
            record_layers=layer_names,
            init_rel=init_rel,
            heatmap_fn=lambda g: g.squeeze(0),
            exclude_parallel=True,
        )

    assert torch.allclose(attr_excl.heatmap, torch.tensor([-0.1, 0.2]))
    assert torch.allclose(attr_excl.relevances["layer1"], torch.tensor([[0.1, 0.0]]), atol=1e-6)
    assert torch.allclose(attr_excl.relevances["layer2"], torch.tensor([[0.0, 0.0]]), atol=1e-6)


def test_parallel_cond_attribution(simple_model):
    """Conditioning on channel 0 in both parallel branches simultaneously."""
    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    composite = EpsilonPlus()
    layer_names = get_layer_names(simple_model, [nn.Linear])
    init_rel = torch.eye(1, dtype=torch.float32)[[0]]
    conditions = [{"y": [0], "layer2": [0], "layer1": [0]}]

    with ConditionalGradient(simple_model, composite=composite) as attributor:
        attr = attributor(
            inp,
            conditions,
            record_layers=layer_names,
            init_rel=init_rel,
            heatmap_fn=lambda g: g.squeeze(0),
            exclude_parallel=False,
        )

    assert torch.allclose(attr.heatmap, torch.tensor([-1.1, 1.7]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([[0.1, 0.0]]), atol=1e-6)
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([[0.5, 0.0]]), atol=1e-6)


def test_seq_cond_attribution(simple_model):
    """Sequential conditioning: mask layer3 channel 0 then layer1 channel 0."""
    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    composite = EpsilonPlus()
    layer_names = get_layer_names(simple_model, [nn.Linear])
    init_rel = torch.eye(1, dtype=torch.float32)[[0]]
    conditions = [{"y": [0], "layer3": [0], "layer1": [0]}]

    with ConditionalGradient(simple_model, composite=composite) as attributor:
        attr = attributor(
            inp,
            conditions,
            record_layers=layer_names,
            init_rel=init_rel,
            heatmap_fn=lambda g: g.squeeze(0),
            exclude_parallel=True,
        )

    assert torch.allclose(attr.heatmap, torch.tensor([-0.1, 0.2]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([[0.1, 0.0]]), atol=1e-6)
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([[0.0, 0.0]]), atol=1e-6)
