"""Helper utilities for zennit-crp.

Provides functions for layer introspection, normalization, and loading
precomputed feature visualization results from disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def get_layer_names(model: torch.nn.Module, types: list[type]) -> list[str]:
    """Retrieve names of all layers matching the given types.

    Parameters
    ----------
    model : torch.nn.Module
        The model to inspect.
    types : list[type]
        Layer types to match, e.g. ``[torch.nn.Conv2d, torch.nn.Linear]``.

    Returns
    -------
    list[str]
        Names of matching layers, in module traversal order.
    """
    return [name for name, layer in model.named_modules() if any(isinstance(layer, t) for t in types)]


def abs_norm(rel: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize a tensor by the sum of its absolute values.

    Parameters
    ----------
    rel : torch.Tensor
        Input tensor.
    eps : float, optional
        Stabilizer to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    return rel / (torch.sum(torch.abs(rel)) + eps)


def max_norm(rel: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Normalize a tensor by its maximum value.

    Parameters
    ----------
    rel : torch.Tensor
        Input tensor.
    eps : float, optional
        Stabilizer to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    return rel / (rel.max() + eps)


def get_output_shapes(
    model: torch.nn.Module,
    sample: torch.Tensor,
    record_layers: list[str],
) -> dict[str, torch.Size]:
    """Compute output shapes of specified layers via a forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        The model to inspect.
    sample : torch.Tensor
        An example input tensor (single sample).
    record_layers : list[str]
        Layer names for which to record output shapes.

    Returns
    -------
    dict[str, torch.Size]
        Mapping from layer names to their output shapes (excluding the batch dimension).
    """
    output_shapes: dict[str, torch.Size] = {}

    def make_hook(name: str):
        def hook(module, input, output):
            output_shapes[name] = output.shape[1:]

        return hook

    handles = [
        layer.register_forward_hook(make_hook(name)) for name, layer in model.named_modules() if name in record_layers
    ]

    with torch.no_grad():
        model(sample)

    for h in handles:
        h.remove()

    return output_shapes


def load_maximization(path_folder: str | Path, layer_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed maximization results from disk.

    Parameters
    ----------
    path_folder : str or Path
        Directory containing the result files.
    layer_name : str
        Name of the layer.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Data indices, relevance values, and receptive field indices (memory-mapped).
    """
    folder = Path(path_folder)
    prefix = f"{layer_name}_"
    return (
        np.load(folder / f"{prefix}data.npy", mmap_mode="r"),
        np.load(folder / f"{prefix}rel.npy", mmap_mode="r"),
        np.load(folder / f"{prefix}rf.npy", mmap_mode="r"),
    )


def load_stat_targets(path_folder: str | Path) -> np.ndarray:
    """Load target array for statistics.

    Parameters
    ----------
    path_folder : str or Path
        Directory containing ``targets.npy``.

    Returns
    -------
    np.ndarray
        Integer array of targets.
    """
    return np.load(Path(path_folder) / "targets.npy").astype(np.int32)


def load_statistics(path_folder: str | Path, layer_name: str, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed statistics results from disk.

    Parameters
    ----------
    path_folder : str or Path
        Directory containing the result files.
    layer_name : str
        Name of the layer.
    target : int
        Target class index.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Data indices, relevance values, and receptive field indices (memory-mapped).
    """
    folder = Path(path_folder) / layer_name
    prefix = f"{target}_"
    return (
        np.load(folder / f"{prefix}data.npy", mmap_mode="r"),
        np.load(folder / f"{prefix}rel.npy", mmap_mode="r"),
        np.load(folder / f"{prefix}rf.npy", mmap_mode="r"),
    )


def load_receptive_field(path_folder: str | Path, layer_name: str) -> np.ndarray:
    """Load a precomputed receptive field array.

    Parameters
    ----------
    path_folder : str or Path
        Directory containing the receptive field file.
    layer_name : str
        Name of the layer.

    Returns
    -------
    np.ndarray
        Receptive field data.
    """
    return np.load(Path(path_folder) / f"{layer_name}.npy")
