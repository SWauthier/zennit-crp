import pytest
import torch
import torch.nn as nn
from zennit.composites import EpsilonPlus

from zennit_crp.attribution import ConditionalGradient


class SimpleModel(nn.Module):
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
        y3 = torch.cat([y1, y2], dim=1)
        return self.layer3(y3)


@pytest.fixture
def simple_model():
    return SimpleModel()


def test_simple_attribution(simple_model):
    """Test basic conditional attribution with EpsilonPlus."""
    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0]}]
    composite = EpsilonPlus()

    with ConditionalGradient(simple_model, composite=composite) as attributor:
        attr = attributor(
            inp,
            conditions,
            record_layers=["layer1", "layer2", "layer3"],
            init_rel=torch.eye(1, dtype=torch.float32)[[0]],
            heatmap_fn=lambda g: g.squeeze(0),
        )

    assert torch.allclose(attr.heatmap, torch.tensor([-1.1, 2.1]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([[0.1, 0.4]]), atol=1e-6)
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([[0.5, 0.0]]), atol=1e-6)


def test_parallel_attribution(simple_model):
    """Test conditional masking on parallel layers."""
    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0], "layer1": [0], "layer2": []}]
    composite = EpsilonPlus()

    with ConditionalGradient(simple_model, composite=composite) as attributor:
        attr = attributor(
            inp,
            conditions,
            record_layers=["layer1", "layer2", "layer3"],
            init_rel=torch.eye(1, dtype=torch.float32)[[0]],
            heatmap_fn=lambda g: g.squeeze(0),
            exclude_parallel=False,
        )

    assert torch.allclose(attr.heatmap, torch.tensor([-0.1, 0.2]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([[0.1, 0.0]]), atol=1e-6)
    assert torch.allclose(attr.relevances["layer2"], torch.tensor([[0.0, 0.0]]), atol=1e-6)


def test_exclude_parallel(simple_model):
    """Test exclude_parallel restricts gradient flow."""
    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0], "layer1": [0]}]
    composite = EpsilonPlus()

    with ConditionalGradient(simple_model, composite=composite) as attributor:
        attr = attributor(
            inp,
            conditions,
            record_layers=["layer1", "layer2", "layer3"],
            init_rel=torch.eye(1, dtype=torch.float32)[[0]],
            heatmap_fn=lambda g: g.squeeze(0),
            exclude_parallel=True,
        )

    assert torch.allclose(attr.heatmap, torch.tensor([-0.1, 0.2]))
    assert torch.allclose(attr.relevances["layer1"], torch.tensor([[0.1, 0.0]]), atol=1e-6)


def test_broadcast_single_input(simple_model):
    """Test that a single input is broadcast to match multiple conditions."""
    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0]}, {"y": [0]}]
    composite = EpsilonPlus()

    with ConditionalGradient(simple_model, composite=composite) as attributor:
        attr = attributor(
            inp,
            conditions,
            init_rel=torch.eye(1, dtype=torch.float32)[[0]],
            heatmap_fn=lambda g: g.squeeze(1),
        )

    assert attr.heatmap.shape[0] == 2


def test_conditions_module():
    """Test condition utility functions."""
    from zennit_crp.conditions import conditioned_layer_names, partition_conditions, split_output_conditions

    conditions = [{"conv1": [0], "y": [1]}, {"conv1": [1], "y": [2]}, {"conv2": [0]}]

    partitions = partition_conditions(conditions)
    assert len(partitions) == 2

    y_targets, layer_conds = split_output_conditions(conditions)
    assert y_targets == [[1], [2], None]
    assert all("y" not in c for c in layer_conds)

    names = conditioned_layer_names(conditions)
    assert set(names) == {"conv1", "conv2"}


def test_concepts():
    """Test ChannelConcept attribute and reference_sampling."""
    from zennit_crp.concepts import ChannelConcept

    rel = torch.randn(4, 8, 3, 3)

    attr = ChannelConcept.attribute(rel)
    assert attr.shape == (4, 8)
    # abs_norm divides by sum of absolute values; absolute values must sum to 1
    assert torch.allclose(attr.abs().sum(dim=-1), torch.ones(4), atol=1e-4)

    b_sorted, rel_sorted, rf_sorted = ChannelConcept.reference_sampling(rel)
    assert b_sorted.shape == (4, 8)
    assert rel_sorted.shape == (4, 8)
    assert rf_sorted.shape == (4, 8)


def test_mask_hook():
    """Test MaskHook applies gradient masks."""
    from zennit_crp.hooks import MaskHook

    def mask_fn(grad):
        grad = grad.clone()
        grad[:, 1:] = 0
        return grad

    hook = MaskHook([mask_fn])
    grad = torch.ones(2, 4)
    result = hook.pre_backward(None, None, (grad,))
    assert result[0][:, 0].sum() == 2.0
    assert result[0][:, 1:].sum() == 0.0
