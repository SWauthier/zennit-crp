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


def test_recording_hook_inherits_hook():
    """Test RecordingHook is a proper zennit Hook subclass."""
    from zennit.core import Hook, RemovableHandleList

    from zennit_crp.hooks import RecordingHook

    hook = RecordingHook()
    assert isinstance(hook, Hook)
    assert hasattr(hook, "stored_tensors")
    assert hasattr(hook, "active")
    assert hasattr(hook, "tensor_handles")

    # register() returns RemovableHandleList (zennit's convention)
    model = nn.Linear(2, 2)
    handles = hook.register(model)
    assert isinstance(handles, RemovableHandleList)

    # forward recording works
    inp = torch.randn(1, 2, requires_grad=True)
    out = model(inp)
    assert hook.output is not None
    assert torch.allclose(out, hook.output)

    handles.remove()


def test_auto_registration(simple_model):
    """Test ConditionalGradient auto-registers composite without context manager."""
    inp = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    conditions = [{"y": [0]}]
    composite = EpsilonPlus()

    # Call without `with` block — composite should be auto-registered and removed
    attributor = ConditionalGradient(simple_model, composite=composite)
    attr = attributor(
        inp,
        conditions,
        record_layers=["layer1", "layer2", "layer3"],
        init_rel=torch.eye(1, dtype=torch.float32)[[0]],
        heatmap_fn=lambda g: g.squeeze(0),
    )

    assert torch.allclose(attr.heatmap, torch.tensor([-1.1, 2.1]))
    # Composite should be removed after the call
    assert not composite.handles


def test_find_files(tmp_path):
    """Test find_files discovers analysis result directories."""
    from zennit_crp.helper import find_files

    # Create mock result directories
    (tmp_path / "RelMax_sum_normed").mkdir()
    (tmp_path / "ActMax_sum_normed").mkdir()
    (tmp_path / "RelStats_sum_normed").mkdir()
    (tmp_path / "ActStats_sum_normed").mkdir()
    (tmp_path / "ReField").mkdir()
    (tmp_path / "unrelated").mkdir()

    r_max, a_max, r_stats, a_stats, rf = find_files(tmp_path)
    assert len(r_max) == 1 and "RelMax" in r_max[0]
    assert len(a_max) == 1 and "ActMax" in a_max[0]
    assert len(r_stats) == 1 and "RelStats" in r_stats[0]
    assert len(a_stats) == 1 and "ActStats" in a_stats[0]
    assert len(rf) == 1 and "ReField" in rf[0]


def test_dump_pytorch_graph(capsys):
    """Test dump_pytorch_graph prints graph nodes."""
    from zennit_crp.graph import dump_pytorch_graph

    model = nn.Linear(2, 2)
    traced = torch.jit.trace(model, torch.randn(1, 2))
    dump_pytorch_graph(traced.inlined_graph)

    captured = capsys.readouterr()
    assert "kind" in captured.out
    assert "scopeName" in captured.out


def test_channel_concept_get_rf_indices():
    """Test ChannelConcept.get_rf_indices for conv and linear layers."""
    from zennit_crp.concepts import ChannelConcept

    # Linear layer output shape (1-D)
    assert ChannelConcept.get_rf_indices(torch.Size([64])) == [0]

    # Conv layer output shape (C, H, W)
    rf = ChannelConcept.get_rf_indices(torch.Size([16, 4, 4]))
    assert len(rf) == 16  # 4 * 4
    assert rf[0] == 0
    assert rf[-1] == 15


def test_attribution_graph_set_layer_map(simple_model):
    """Test AttributionGraph.set_layer_map rebuilds mask_map."""
    from zennit_crp.attribution import AttributionGraph
    from zennit_crp.concepts import ChannelConcept

    layer_map = {"layer1": ChannelConcept()}
    ag = AttributionGraph(
        ConditionalGradient(simple_model),
        graph=None,
        layer_map=layer_map,
    )
    assert "layer1" in ag.mask_map

    new_layer_map = {"layer2": ChannelConcept()}
    ag.set_layer_map(new_layer_map)
    assert "layer2" in ag.mask_map
    assert "layer1" not in ag.mask_map


def test_all_exports():
    """Test all public names are accessible from the top-level package."""
    import zennit_crp

    for name in zennit_crp.__all__:
        assert hasattr(zennit_crp, name), f"{name} not accessible from zennit_crp"
