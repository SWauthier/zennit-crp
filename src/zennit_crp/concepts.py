"""Concept definitions for Concept Relevance Propagation.

Concepts define how relevance is attributed to interpretable units within a layer.
The default implementation treats each channel as a separate concept.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Concept(Protocol):
    """Protocol for concept implementations.

    A concept defines how to mask gradients for conditional attribution,
    how to attribute relevance to individual concepts, and how to sample
    reference examples.
    """

    @staticmethod
    def mask(batch_id: int, concept_ids: list[int]) -> Callable:
        """Create a gradient mask function for the given concepts.

        Parameters
        ----------
        batch_id : int
            Index in the batch dimension.
        concept_ids : list[int]
            Channel/neuron indices to condition on.

        Returns
        -------
        callable
            Function ``(grad: torch.Tensor) -> torch.Tensor`` that masks the gradient.
        """
        ...

    def attribute(self, relevance: torch.Tensor, abs_norm: bool = True) -> torch.Tensor:
        """Compute per-concept attribution from a relevance tensor.

        Parameters
        ----------
        relevance : torch.Tensor
            Relevance tensor of shape ``(batch, channels, *spatial)``.
        abs_norm : bool, optional
            Whether to normalize by absolute sum. Default is ``True``.

        Returns
        -------
        torch.Tensor
            Per-concept attribution of shape ``(batch, channels)``.
        """
        ...

    def reference_sampling(
        self, relevance: torch.Tensor, max_target: str = "sum", abs_norm: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample references based on relevance.

        Parameters
        ----------
        relevance : torch.Tensor
            Relevance tensor of shape ``(batch, channels, *spatial)``.
        max_target : str, optional
            Aggregation mode, either ``"sum"`` or ``"max"``. Default is ``"sum"``.
        abs_norm : bool, optional
            Whether to normalize by absolute sum. Default is ``True``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Batch-sorted indices, relevance values, and receptive field neuron indices.
        """
        ...


class ChannelConcept:
    """Concept implementation treating each channel as a separate concept.

    Suitable for :py:class:`torch.nn.Conv2d` and :py:class:`torch.nn.Linear` layers.
    Each output channel (or neuron in linear layers) is considered an individual concept.
    """

    @staticmethod
    def mask(batch_id: int, concept_ids: list[int]) -> Callable:
        """Create a gradient mask that zeros out all but the specified channels.

        Parameters
        ----------
        batch_id : int
            Index in the batch dimension.
        concept_ids : list[int]
            Channel indices to keep. All other channels are zeroed.

        Returns
        -------
        callable
            Mask function ``(grad: torch.Tensor) -> torch.Tensor``.
        """

        def mask_fn(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(grad[batch_id])
            mask[concept_ids] = 1
            masked = grad.clone()
            masked[batch_id] = mask * masked[batch_id]
            return masked

        return mask_fn

    @staticmethod
    def mask_rf(batch_id: int, channel_neuron_map: dict[int, list[int]]) -> Callable:
        """Create a gradient mask for specific neurons within channels (receptive field).

        Parameters
        ----------
        batch_id : int
            Index in the batch dimension.
        channel_neuron_map : dict[int, list[int]]
            Mapping from channel indices to lists of spatial neuron indices.
            Neuron indices are linearized, e.g. shape ``(C, H, W)`` -> ``(C, H*W)``.

        Returns
        -------
        callable
            Mask function ``(grad: torch.Tensor) -> torch.Tensor``.
        """

        def mask_fn(grad: torch.Tensor) -> torch.Tensor:
            original_shape = grad.shape
            grad = grad.view(*original_shape[:2], -1)
            mask = torch.zeros_like(grad[batch_id])
            for channel, neurons in channel_neuron_map.items():
                mask[channel, neurons] = 1
            grad[batch_id] = grad[batch_id] * mask
            return grad.view(original_shape)

        return mask_fn

    @staticmethod
    def attribute(
        relevance: torch.Tensor,
        mask: torch.Tensor | None = None,
        abs_norm: bool = True,
    ) -> torch.Tensor:
        """Compute per-channel relevance attribution.

        Sums relevance over spatial dimensions and optionally normalizes.

        Parameters
        ----------
        relevance : torch.Tensor
            Relevance tensor of shape ``(batch, channels, *spatial)``.
        mask : torch.Tensor, optional
            Optional mask to apply before attribution.
        abs_norm : bool, optional
            If ``True``, normalize by the sum of absolute relevances. Default is ``True``.

        Returns
        -------
        torch.Tensor
            Per-channel attribution of shape ``(batch, channels)``.
        """
        if mask is not None:
            relevance = relevance * mask

        # Sum over spatial dimensions: (batch, channels, *spatial) -> (batch, channels)
        rel_c = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        if abs_norm:
            rel_c = rel_c / (torch.abs(rel_c).sum(-1, keepdim=True) + 1e-10)

        return rel_c

    @staticmethod
    def get_rf_indices(output_shape: torch.Size) -> list[int] | np.ndarray:
        """Return receptive field neuron indices for a given output shape.

        For linear layers (1-D output), returns ``[0]``.  For convolutional
        layers, returns all spatial indices ``[0, ..., H*W - 1]``.

        Parameters
        ----------
        output_shape : torch.Size
            Shape of the layer output *excluding* the batch dimension,
            e.g. ``(C, H, W)``.

        Returns
        -------
        list[int] or np.ndarray
            Receptive field neuron indices.
        """
        if len(output_shape) == 1:
            return [0]
        return np.arange(0, int(np.prod(output_shape[1:])))

    @staticmethod
    def reference_sampling(
        relevance: torch.Tensor,
        max_target: str = "sum",
        abs_norm: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find the most relevant samples and neurons per channel.

        Parameters
        ----------
        relevance : torch.Tensor
            Relevance tensor of shape ``(batch, channels, *spatial)``.
        max_target : str, optional
            ``"sum"`` to rank by total channel relevance,
            ``"max"`` to rank by peak neuron relevance. Default is ``"sum"``.
        abs_norm : bool, optional
            If ``True``, normalize by absolute sum. Default is ``True``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - ``b_sorted``: Batch indices sorted by relevance per channel.
            - ``rel_sorted``: Corresponding relevance values.
            - ``rf_sorted``: Receptive field neuron indices (argmax within each channel).
        """
        # Flatten spatial dims: (batch, channels, *spatial) -> (batch, channels, neurons)
        rel_flat = relevance.view(*relevance.shape[:2], -1)

        # Receptive field: most relevant neuron per channel
        rf_neuron = torch.argmax(rel_flat, dim=-1)

        # Per-channel relevance
        match max_target:
            case "sum":
                rel_c = torch.sum(rel_flat, dim=-1)
            case "max":
                rel_c = torch.max(rel_flat, dim=-1).values
            case _:
                raise ValueError(f"max_target must be 'sum' or 'max', got '{max_target}'")

        if abs_norm:
            rel_c = rel_c / (torch.abs(rel_c).sum(-1, keepdim=True) + 1e-10)

        # Sort batch indices by relevance for each channel
        b_sorted = torch.argsort(rel_c, dim=0, descending=True)

        rel_sorted = torch.gather(rel_c, 0, b_sorted)
        rf_sorted = torch.gather(rf_neuron, 0, b_sorted)

        return b_sorted, rel_sorted, rf_sorted
