"""Composites for Concept Relevance Propagation.

Provides :py:class:`MaskComposite`, which builds gradient-masking hooks from
condition dictionaries and registers them via zennit's composite system.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from zennit.composites import NameMapComposite

from zennit_crp.conditions import MODEL_OUTPUT_NAME
from zennit_crp.hooks import MaskHook


class MaskComposite(NameMapComposite):
    """Composite that masks gradients at specified layers based on conditions.

    Each condition in the list corresponds to one sample in the batch.
    For each conditioned layer, a :py:class:`~zennit_crp.hooks.MaskHook` is
    created with mask functions derived from the conditions.

    Parameters
    ----------
    conditions : list[dict[str, list[int]]]
        Each dictionary maps layer names to lists of concept (channel) indices.
        The special key ``"y"`` references the model output and is ignored here.
    mask_fn : callable, optional
        Function with signature ``(batch_id: int, concept_ids: list[int]) -> callable``
        that returns a gradient mask function. If ``None``, uses
        :py:meth:`default_mask_fn`.
    name_map : list[tuple[list[str], Hook]], optional
        Additional name-to-hook mappings prepended to those generated from conditions.
    canonizers : list[Canonizer], optional
        List of canonizer instances to apply before hooks.

    Examples
    --------
    >>> conditions = [{"conv1": [0, 1], "y": [3]}, {"conv1": [2], "y": [5]}]
    >>> composite = MaskComposite(conditions)
    """

    def __init__(
        self,
        conditions: list[dict[str, list[int]]],
        mask_fn: Callable[[int, list[int]], Callable] | dict[str, Callable] | None = None,
        name_map: list | None = None,
        canonizers: list | None = None,
    ):
        # Build hook map: layer_name -> MaskHook with accumulated masks
        hook_map: dict[str, MaskHook] = {}
        for batch_id, condition in enumerate(conditions):
            for layer_name, concept_ids in condition.items():
                if layer_name == MODEL_OUTPUT_NAME:
                    continue
                if layer_name not in hook_map:
                    hook_map[layer_name] = MaskHook([])
                fn = self._resolve_mask_fn(mask_fn, layer_name)
                hook_map[layer_name].masks.append(fn(batch_id, concept_ids))

        # Convert to name_map format expected by NameMapComposite
        mask_name_map = [([name], hook) for name, hook in hook_map.items()]
        combined = (name_map or []) + mask_name_map

        super().__init__(name_map=combined, canonizers=canonizers)

    @staticmethod
    def _resolve_mask_fn(
        mask_fn: Callable | dict[str, Callable] | None,
        layer_name: str,
    ) -> Callable:
        """Resolve a mask function for a specific layer.

        Supports a single callable (used for all layers), a dict mapping
        layer names to callables, or ``None`` for the default channel mask.

        Parameters
        ----------
        mask_fn : callable, dict, or None
            Mask function specification.
        layer_name : str
            Name of the target layer.

        Returns
        -------
        callable
            A mask function ``(batch_id, concept_ids) -> mask_callable``.
        """
        if mask_fn is None:
            return MaskComposite.default_mask_fn
        if isinstance(mask_fn, dict):
            return mask_fn.get(layer_name, MaskComposite.default_mask_fn)
        return mask_fn

    @staticmethod
    def default_mask_fn(batch_id: int, concept_ids: list[int]) -> Callable:
        """Create a default channel mask function.

        Zeros out all channels except those in ``concept_ids`` for the
        specified batch element.

        Parameters
        ----------
        batch_id : int
            Index into the batch dimension.
        concept_ids : list[int]
            Channel indices to preserve.

        Returns
        -------
        callable
            Function ``(grad: torch.Tensor) -> torch.Tensor``.
        """

        def mask(grad: torch.Tensor) -> torch.Tensor:
            result = grad.clone()
            channel_mask = torch.zeros_like(grad[batch_id])
            channel_mask[concept_ids] = 1
            result[batch_id] = channel_mask * result[batch_id]
            return result

        return mask
