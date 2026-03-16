from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import FashionMNIST
from zennit.composites import EpsilonPlus
from zennit.layer import Sum
from zennit.torchvision import SequentialMergeBatchNorm

from zennit_crp.attribution import ConditionalGradient


class FashionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.parallel = nn.Sequential(nn.Conv2d(1, 16, 5, stride=2, bias=False), nn.BatchNorm2d(16))
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()
        self.sum = Sum()
        self.maxpooling = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 10)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.parallel is not None:
            identity = self.parallel(identity)

        out = torch.stack([identity, out], dim=-1)
        out = self.sum(out)
        out = self.relu(out)
        out = self.maxpooling(out)
        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu(out)

        return self.linear2(out)


class SplitFashionMNIST(FashionMNIST):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, False, transform, target_transform, False)

    def _check_exists(self):
        return True

    def _load_data(self):
        data = np.load(Path(self.root, "fashion_val.npz"))
        return torch.from_numpy(data["val_set"]), torch.from_numpy(data["targets"])


@pytest.fixture
def fashion_model_data():
    model = FashionModel()
    model.load_state_dict(torch.load("tests/data/fashion_acc=89.ckpt", weights_only=False)["state_dict"])
    model.eval()

    val_set = SplitFashionMNIST(root="tests/data", transform=T.Compose([T.ToTensor()]))
    return model, val_set


def test_fashion_conditional_attribution(fashion_model_data):
    """Test conditional attribution on FashionMNIST model matches saved references."""
    model, dataset = fashion_model_data

    composite = EpsilonPlus(canonizers=[SequentialMergeBatchNorm()])
    test_sample, target = dataset[0]
    test_sample = test_sample.unsqueeze(0).requires_grad_()

    conditions = [
        {"y": [target]},
        {"y": [target], "conv2": []},
        {"y": [target], "conv2": [2]},
        {"y": [target], "parallel.0": []},
        {"y": [target], "parallel.0": [], "conv2": [3, 2, 1]},
    ]

    with ConditionalGradient(model, composite=composite) as attributor:
        attr = attributor(
            test_sample,
            conditions,
            record_layers=["conv1", "conv2"],
            init_rel=abs,
            exclude_parallel=False,
        )

    heatmaps_ref = np.load("tests/data/heatmaps.npz")["heatmaps"]
    conv1_ref = np.load("tests/data/conv1_relevances.npz")["conv1_relevances"]

    assert np.allclose(heatmaps_ref, attr.heatmap.numpy(), atol=1e-5)
    assert np.allclose(conv1_ref, attr.relevances["conv1"].numpy(), atol=1e-5)


def test_fashion_exclude_parallel(fashion_model_data):
    """Test exclude_parallel produces same results for single-path conditions."""
    model, dataset = fashion_model_data

    composite = EpsilonPlus(canonizers=[SequentialMergeBatchNorm()])
    test_sample, target = dataset[0]
    test_sample = test_sample.unsqueeze(0).requires_grad_()

    conditions = [{"y": [target], "conv2": [3, 2, 1]}]

    with ConditionalGradient(model, composite=composite) as attributor:
        attr_p = attributor(
            test_sample,
            conditions,
            record_layers=["conv1", "conv2"],
            init_rel=abs,
            exclude_parallel=True,
        )

    heatmaps_ref = np.load("tests/data/heatmaps.npz")["heatmaps"]
    conv1_ref = np.load("tests/data/conv1_relevances.npz")["conv1_relevances"]

    assert np.allclose(heatmaps_ref[-1], attr_p.heatmap.numpy()[-1], atol=1e-5)
    assert np.allclose(conv1_ref[-1], attr_p.relevances["conv1"].numpy()[-1], atol=1e-5)


def test_fashion_generator(fashion_model_data):
    """Test generator-based batch attribution matches saved references."""
    model, dataset = fashion_model_data

    composite = EpsilonPlus(canonizers=[SequentialMergeBatchNorm()])
    test_sample, target = dataset[0]
    test_sample = test_sample.unsqueeze(0).requires_grad_()

    conditions = [{"y": [target], "parallel.0": [], "conv2": [i]} for i in range(16)]

    heatmaps, relevances = [], []
    with ConditionalGradient(model, composite=composite) as attributor:
        for attr in attributor.generate(
            test_sample,
            conditions,
            record_layers=["conv1"],
            init_rel=abs,
            batch_size=5,
            verbose=False,
            exclude_parallel=False,
        ):
            heatmaps.append(attr.heatmap)
            relevances.append(attr.relevances["conv1"])

    heatmaps = torch.cat(heatmaps)
    relevances = torch.cat(relevances)

    gen_heatmaps = np.load("tests/data/gen_heatmaps.npz")["heatmaps"]
    gen_conv1 = np.load("tests/data/gen_conv1_relevances.npz")["conv1_relevances"]

    assert np.allclose(gen_heatmaps, heatmaps.numpy(), atol=1e-5)
    assert np.allclose(gen_conv1, relevances.numpy(), atol=1e-5)
