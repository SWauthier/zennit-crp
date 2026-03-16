"""Attributors for Concept Relevance Propagation.

Provides :py:class:`ConditionalGradient`, a zennit-compatible attributor that
computes conditional attributions by masking relevance flows during the backward
pass. Also provides :py:class:`AttributionGraph` for decomposing higher-level
concepts into their constituent lower-level concepts.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from zennit.attribution import Gradient
from zennit.core import RemovableHandleList

from zennit_crp.composites import MaskComposite
from zennit_crp.concepts import ChannelConcept
from zennit_crp.conditions import (
    MODEL_OUTPUT_NAME,
    conditioned_layer_names,
    partition_conditions,
    split_output_conditions,
)
from zennit_crp.hooks import MaskHook, RecordingHook


@dataclass(frozen=True)
class AttributionResult:
    """Result of a conditional attribution computation.

    Attributes
    ----------
    heatmap : torch.Tensor
        Attribution heatmap, typically summed over the channel dimension.
    activations : dict[str, torch.Tensor]
        Recorded layer activations, keyed by layer name.
    relevances : dict[str, torch.Tensor]
        Recorded layer relevances (gradients under LRP), keyed by layer name.
    prediction : torch.Tensor
        Model output (logits).
    """

    heatmap: torch.Tensor
    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    relevances: dict[str, torch.Tensor] = field(default_factory=dict)
    prediction: torch.Tensor | None = None


@dataclass(frozen=True)
class GraphResult:
    """Result of an attribution graph decomposition.

    Attributes
    ----------
    nodes : list[tuple[str, int]]
        All concept nodes as ``(layer_name, concept_id)`` pairs.
    connections : dict[tuple[str, int], list[tuple[str, int, float]]]
        Edges of the graph. Each key ``(layer, concept)`` maps to a list
        of ``(child_layer, child_concept, relevance)`` tuples.
    """

    nodes: list[tuple[str, int]]
    connections: dict[tuple[str, int], list[tuple[str, int, float]]]


def _broadcast(
    data: torch.Tensor, conditions: list[dict[str, list[int]]]
) -> tuple[torch.Tensor, list[dict[str, list[int]]]]:
    """Broadcast data and conditions to compatible batch sizes.

    If there is one data sample and multiple conditions, the sample is
    repeated. If there are multiple samples and one condition, the condition
    is repeated. Otherwise they must have the same length.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor.
    conditions : list[dict[str, list[int]]]
        Condition dictionaries.

    Returns
    -------
    tuple[torch.Tensor, list[dict[str, list[int]]]]
        Broadcast data and conditions.
    """
    n_data, n_cond = len(data), len(conditions)

    if n_data == n_cond:
        return data, conditions

    if n_cond > 1 and n_data == 1:
        data = data.repeat(n_cond, *([1] * (data.ndim - 1)))
    elif n_data > 1 and n_cond == 1:
        conditions = conditions * n_data
    else:
        raise ValueError(f"Cannot broadcast {n_data} samples with {n_cond} conditions.")

    return data, conditions


def _init_relevance(
    prediction: torch.Tensor,
    y_targets: list[list[int] | None],
    init_rel: torch.Tensor | int | Callable | None = None,
) -> torch.Tensor:
    """Initialize the relevance signal for the backward pass.

    Parameters
    ----------
    prediction : torch.Tensor
        Model output logits.
    y_targets : list[list[int] | None]
        Per-condition output neuron indices.
    init_rel : torch.Tensor, int, callable, or None
        Relevance initializer. If callable, receives prediction. If ``None``,
        uses the prediction values at specified targets.

    Returns
    -------
    torch.Tensor
        Relevance initialization tensor with the same shape as ``prediction``.
    """
    if callable(init_rel):
        output_selection = init_rel(prediction)
    elif isinstance(init_rel, torch.Tensor):
        output_selection = init_rel.expand_as(prediction)
    elif isinstance(init_rel, int):
        output_selection = torch.full(prediction.shape, init_rel, dtype=prediction.dtype)
    else:
        output_selection = prediction

    # Apply target mask if any targets are specified
    has_targets = any(t is not None for t in y_targets)
    if has_targets:
        mask = torch.zeros_like(output_selection)
        for i, targets in enumerate(y_targets):
            if targets is not None:
                mask[i, targets] = output_selection[i, targets]
        output_selection = mask

    return output_selection.to(prediction)


def _default_heatmap_fn(gradient: torch.Tensor) -> torch.Tensor:
    """Default heatmap: sum gradient over channel dimension.

    Parameters
    ----------
    gradient : torch.Tensor
        Input gradient tensor.

    Returns
    -------
    torch.Tensor
        Heatmap with channels summed.
    """
    return torch.sum(gradient, dim=1)


class ConditionalGradient(Gradient):
    """Gradient attributor for Concept Relevance Propagation.

    Extends zennit's :py:class:`~zennit.attribution.Gradient` with conditional
    masking support. When called with conditions, gradient masks are applied to
    specified layers, effectively implementing CRP's conditional backpropagation.

    Combined with a zennit composite (e.g. ``EpsilonPlus``), this computes
    CRP attribution values.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which attributions are computed. If ``composite`` is
        provided, it will be registered to this model.
    composite : Composite, optional
        Zennit composite describing the LRP relevance distribution rules.
    attr_output : torch.Tensor or callable, optional
        Default output attribution.
    create_graph : bool, optional
        Whether to create the computation graph for higher-order gradients.
    retain_graph : bool, optional
        Whether to retain the computation graph after backward.

    Examples
    --------
    >>> from zennit.composites import EpsilonPlus
    >>> from zennit.canonizers import SequentialMergeBatchNorm
    >>> composite = EpsilonPlus(canonizers=[SequentialMergeBatchNorm()])
    >>> with ConditionalGradient(model, composite=composite) as attributor:
    ...     result = attributor(data, [{"conv1": [0, 1], "y": [3]}], record_layers=["conv1"])
    ...     result.heatmap, result.activations, result.relevances
    """

    def __call__(
        self,
        input: torch.Tensor,
        conditions: list[dict[str, list[int]]],
        *,
        record_layers: list[str] | None = None,
        mask_fn: Callable | dict[str, Callable] | None = None,
        init_rel: torch.Tensor | int | Callable | None = None,
        start_layer: str | None = None,
        on_device: str | torch.device | None = None,
        exclude_parallel: bool = False,
        heatmap_fn: Callable | None = None,
    ) -> AttributionResult:
        """Compute conditional attributions.

        Parameters
        ----------
        input : torch.Tensor
            Input sample for which a conditional heatmap is computed.
            Must have ``requires_grad=True``.
        conditions : list[dict[str, list[int]]]
            Each dict maps layer names to concept (channel) indices.
            Use ``"y"`` for the model output target selection.
        record_layers : list[str], optional
            Layer names at which to record activations and relevances.
        mask_fn : callable, dict[str, callable], or None, optional
            Custom mask function ``(batch_id, concept_ids) -> mask_callable``.
            Can also be a dict mapping layer names to per-layer mask functions.
        init_rel : torch.Tensor, int, callable, or None, optional
            Relevance initialization. If callable, receives prediction.
        start_layer : str, optional
            Start the backward pass from this intermediate layer.
        on_device : str or torch.device, optional
            Device for storing intermediate results.
        exclude_parallel : bool, optional
            If ``True``, restricts gradient flow to avoid parallel connections.
            Requires layer names to be ordered from last to first in the model.
        heatmap_fn : callable, optional
            Function to transform input gradient into heatmap.
            Default sums over the channel dimension.

        Returns
        -------
        AttributionResult
            Contains heatmap, activations, relevances, and prediction.
        """
        if record_layers is None:
            record_layers = []
        if heatmap_fn is None:
            heatmap_fn = _default_heatmap_fn

        # Partition conditions for exclude_parallel handling
        if exclude_parallel:
            return self._partitioned_call(
                input,
                conditions,
                record_layers=record_layers,
                mask_fn=mask_fn,
                init_rel=init_rel,
                start_layer=start_layer,
                on_device=on_device,
                heatmap_fn=heatmap_fn,
            )

        return self._attribute(
            input,
            conditions,
            record_layers=record_layers,
            mask_fn=mask_fn,
            init_rel=init_rel,
            start_layer=start_layer,
            on_device=on_device,
            exclude_parallel=False,
            heatmap_fn=heatmap_fn,
        )

    def _partitioned_call(
        self,
        input: torch.Tensor,
        conditions: list[dict[str, list[int]]],
        **kwargs,
    ) -> AttributionResult:
        """Handle exclude_parallel by partitioning conditions with different layer sets.

        When ``exclude_parallel`` is enabled, conditions must share the same
        set of layer names. This method partitions conditions and processes
        each group separately, concatenating the results.
        """
        partitions = partition_conditions(conditions)

        all_heatmaps, all_predictions = [], []
        all_activations: dict[str, list[torch.Tensor]] = {}
        all_relevances: dict[str, list[torch.Tensor]] = {}

        for partition in partitions.values():
            result = self._attribute(input, partition, exclude_parallel=True, **kwargs)

            all_heatmaps.append(result.heatmap)
            all_predictions.append(result.prediction)
            for name in result.activations:
                all_activations.setdefault(name, []).append(result.activations[name])
            for name in result.relevances:
                all_relevances.setdefault(name, []).append(result.relevances[name])

        return AttributionResult(
            heatmap=torch.cat(all_heatmaps),
            activations={n: torch.cat(v) for n, v in all_activations.items()},
            relevances={n: torch.cat(v) for n, v in all_relevances.items()},
            prediction=torch.cat(all_predictions),
        )

    def _attribute(
        self,
        input: torch.Tensor,
        conditions: list[dict[str, list[int]]],
        *,
        record_layers: list[str],
        mask_fn: Callable | dict[str, Callable] | None,
        init_rel,
        start_layer: str | None,
        on_device,
        exclude_parallel: bool,
        heatmap_fn: Callable,
    ) -> AttributionResult:
        """Core attribution logic.

        Broadcasts input to conditions, builds mask hooks, registers recording
        hooks, and performs the forward and backward passes.
        """
        # Broadcast input to match conditions
        data, conditions = _broadcast(input, conditions)
        if not data.requires_grad:
            data = data.detach().requires_grad_(True)
        data.retain_grad()

        # Separate output targets from layer conditions
        y_targets, _ = split_output_conditions(conditions)
        cond_names = conditioned_layer_names(conditions)

        # Determine which layers need recording hooks
        layers_to_record = set(record_layers) | set(cond_names)
        if start_layer:
            layers_to_record.add(start_layer)

        # Register recording hooks
        recording_hooks: dict[str, RecordingHook] = {}
        rec_handles = RemovableHandleList()
        for name, module in self.model.named_modules():
            if name == MODEL_OUTPUT_NAME:
                raise ValueError(
                    f"Layer name '{name}' conflicts with the model output identifier. "
                    "Please rename the layer or change MODEL_OUTPUT_NAME."
                )
            if name in layers_to_record:
                hook = RecordingHook()
                recording_hooks[name] = hook
                rec_handles.append(hook.register(module))
                layers_to_record.discard(name)

        if start_layer and start_layer not in recording_hooks:
            raise KeyError(f"start_layer '{start_layer}' not found in model.")
        if layers_to_record:
            warnings.warn(f"Layer names not found in model: {layers_to_record}", stacklevel=3)

        # Build mask composite from conditions
        mask_composite = MaskComposite(conditions, mask_fn=mask_fn)

        # Forward and backward within composite contexts
        with mask_composite.context(self.model):
            if start_layer:
                self.model(data)
                prediction = recording_hooks[start_layer].output
                relevance_init = _init_relevance(
                    prediction.detach().clone(),
                    [None] * len(conditions),
                    init_rel,
                )

                # Remove start_layer from segmented backward if present
                seg_names = [n for n in cond_names if n != start_layer]
                self._backward(
                    prediction,
                    data,
                    relevance_init,
                    exclude_parallel,
                    seg_names,
                    recording_hooks,
                )
            else:
                prediction = self.model(data)
                relevance_init = _init_relevance(prediction.detach().clone(), y_targets, init_rel)
                self._backward(
                    prediction,
                    data,
                    relevance_init,
                    exclude_parallel,
                    cond_names,
                    recording_hooks,
                )

            # Compute heatmap from input gradient
            heatmap = heatmap_fn(data.grad.detach())
            if on_device:
                heatmap = heatmap.to(on_device)

            # Collect recorded activations and relevances
            activations: dict[str, torch.Tensor] = {}
            relevances: dict[str, torch.Tensor] = {}
            for name, hook in recording_hooks.items():
                hook.collect(on_device=on_device)
                if name in record_layers:
                    activations[name] = hook.activation
                    relevances[name] = hook.relevance

        rec_handles.remove()

        return AttributionResult(
            heatmap=heatmap,
            activations=activations,
            relevances=relevances,
            prediction=prediction,
        )

    def _backward(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        grad_output: torch.Tensor,
        exclude_parallel: bool,
        cond_names: list[str],
        recording_hooks: dict[str, RecordingHook],
    ):
        """Perform the backward pass, optionally segmented for exclude_parallel.

        When ``exclude_parallel`` is enabled, the backward pass proceeds
        through conditioned layers one at a time using
        :py:func:`torch.autograd.grad`, preventing gradient flow through
        parallel (shortcut) connections.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor to differentiate from.
        input : torch.Tensor
            Input tensor to differentiate to.
        grad_output : torch.Tensor
            Initial relevance signal.
        exclude_parallel : bool
            Whether to segment the backward pass.
        cond_names : list[str]
            Ordered list of conditioned layer names.
        recording_hooks : dict[str, RecordingHook]
            Recording hooks for intermediate layers.
        """
        if exclude_parallel and cond_names:
            # Segmented backward: layer by layer from output to input
            current_output = output
            current_grad = grad_output

            for name in cond_names:
                intermediate = recording_hooks[name].output
                try:
                    (gradient,) = torch.autograd.grad(
                        (current_output,),
                        (intermediate,),
                        grad_outputs=(current_grad,),
                        retain_graph=True,
                        create_graph=self.create_graph,
                    )
                except RuntimeError as e:
                    if "allow_unused=True" not in str(e):
                        raise
                    raise RuntimeError(
                        "Layer names must be ordered from last to first in the model when "
                        "'exclude_parallel' is True. Parallel layers cannot appear in one "
                        "condition."
                    ) from e

                intermediate.grad = None
                current_output = intermediate
                current_grad = gradient

            # Final segment: from first conditioned layer to input
            (gradient,) = torch.autograd.grad(
                (current_output,),
                (input,),
                grad_outputs=(current_grad,),
                retain_graph=self.retain_graph,
                create_graph=self.create_graph,
            )
            input.grad = gradient
        else:
            # Standard backward pass
            torch.autograd.backward(output, grad_output.to(output))

    def generate(
        self,
        input: torch.Tensor,
        conditions: list[dict[str, list[int]]],
        *,
        record_layers: list[str] | None = None,
        mask_fn: Callable | dict[str, Callable] | None = None,
        init_rel=None,
        start_layer: str | None = None,
        batch_size: int = 10,
        on_device: str | torch.device | None = None,
        exclude_parallel: bool = False,
        heatmap_fn: Callable | None = None,
        verbose: bool = True,
    ) -> Iterator[AttributionResult]:
        """Generate conditional attributions for many conditions efficiently.

        Reuses the forward pass across batches of conditions. The computation
        graph is retained until the last batch, reducing memory overhead for
        large condition sets.

        Parameters
        ----------
        input : torch.Tensor
            Single input sample (will be broadcast to batch_size).
        conditions : list[dict[str, list[int]]]
            All conditions to process.
        record_layers : list[str], optional
            Layers to record.
        mask_fn : callable, optional
            Custom mask function.
        init_rel : optional
            Relevance initialization.
        start_layer : str, optional
            Start backward from this layer.
        batch_size : int, optional
            Number of conditions per batch.
        on_device : str or torch.device, optional
            Device for results.
        exclude_parallel : bool, optional
            Restrict gradient flow through parallel connections.
        heatmap_fn : callable, optional
            Heatmap transformation function.
        verbose : bool, optional
            Whether to display a progress bar.

        Yields
        ------
        AttributionResult
            Result for each batch of conditions.
        """
        if record_layers is None:
            record_layers = []
        if heatmap_fn is None:
            heatmap_fn = _default_heatmap_fn

        # Collect all conditioned layer names across all conditions
        all_cond_names = conditioned_layer_names(conditions)

        # Determine layers to record
        layers_to_record = set(record_layers) | set(all_cond_names)
        if start_layer:
            layers_to_record.add(start_layer)

        # Register empty mask hooks for all conditioned layers
        hook_map: dict[str, MaskHook] = {}
        for name in all_cond_names:
            hook_map[name] = MaskHook([])

        # Register recording hooks
        recording_hooks: dict[str, RecordingHook] = {}
        rec_handles = RemovableHandleList()
        for name, module in self.model.named_modules():
            if name in layers_to_record:
                rec_hook = RecordingHook()
                recording_hooks[name] = rec_hook
                rec_handles.append(rec_hook.register(module))

        # Build the mask composite from pre-allocated empty hooks
        from zennit.composites import NameMapComposite

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        # Calculate batching
        n_conditions = len(conditions)
        n_batches = max(1, math.ceil(n_conditions / batch_size))
        actual_batch_size = min(batch_size, n_conditions)

        # Broadcast input to batch size
        data_batch = torch.repeat_interleave(input, actual_batch_size, dim=0)
        data_batch = data_batch.detach().requires_grad_(True)
        data_batch.retain_grad()

        seg_names = [n for n in all_cond_names if n != start_layer] if start_layer else all_cond_names

        with mask_composite.context(self.model):
            # Single forward pass
            if start_layer:
                self.model(data_batch)
                prediction = recording_hooks[start_layer].output
            else:
                prediction = self.model(data_batch)

            progress = tqdm(total=n_batches, dynamic_ncols=True) if verbose else None

            for b in range(n_batches):
                if progress:
                    progress.update(1)

                cond_batch = conditions[b * actual_batch_size : (b + 1) * actual_batch_size]
                is_last = b == n_batches - 1
                current_batch_size = len(cond_batch)

                # Build masks for this batch
                y_targets: list[list[int] | None] = []
                for i, cond in enumerate(cond_batch):
                    for l_name, indices in cond.items():
                        if l_name == MODEL_OUTPUT_NAME:
                            y_targets.append(indices)
                        elif l_name in hook_map:
                            mask_func = MaskComposite._resolve_mask_fn(mask_fn, l_name)
                            hook_map[l_name].masks.append(mask_func(i, indices))

                # Pad y_targets for consistency
                while len(y_targets) < current_batch_size:
                    y_targets.append(None)

                # Initialize relevance
                relevance_init = _init_relevance(
                    prediction.detach().clone(),
                    y_targets,
                    init_rel,
                )

                # Backward pass (retain graph unless last batch)
                retain = not is_last
                if exclude_parallel and seg_names:
                    current_out, current_grad = prediction, relevance_init
                    for name in seg_names:
                        intermediate = recording_hooks[name].output
                        (gradient,) = torch.autograd.grad(
                            (current_out,),
                            (intermediate,),
                            grad_outputs=(current_grad,),
                            retain_graph=True,
                            create_graph=self.create_graph,
                        )
                        intermediate.grad = None
                        current_out, current_grad = intermediate, gradient

                    (gradient,) = torch.autograd.grad(
                        (current_out,),
                        (data_batch,),
                        grad_outputs=(current_grad,),
                        retain_graph=retain,
                        create_graph=self.create_graph,
                    )
                    data_batch.grad = gradient
                else:
                    torch.autograd.backward(
                        prediction,
                        relevance_init.to(prediction),
                        retain_graph=retain,
                    )

                # Collect results
                heatmap = heatmap_fn(data_batch.grad.detach()[:current_batch_size])
                if on_device:
                    heatmap = heatmap.to(on_device)

                activations: dict[str, torch.Tensor] = {}
                relevances_dict: dict[str, torch.Tensor] = {}
                for name, hook in recording_hooks.items():
                    hook.collect(on_device=on_device, length=current_batch_size)
                    if name in record_layers:
                        activations[name] = hook.activation
                        relevances_dict[name] = hook.relevance

                yield AttributionResult(
                    heatmap=heatmap,
                    activations=activations,
                    relevances=relevances_dict,
                    prediction=prediction[:current_batch_size],
                )

                # Reset for next batch
                self._reset_gradients(data_batch)
                for hook in hook_map.values():
                    hook.masks.clear()

            if progress:
                progress.close()

        rec_handles.remove()

    def _reset_gradients(self, data: torch.Tensor):
        """Clear gradients on model parameters and input data."""
        for p in self.model.parameters():
            p.grad = None
        data.grad = None


class AttributionGraph:
    """Decompose a concept into its constituent lower-level concepts.

    Uses the model's computational graph and conditional attribution to trace
    relevance flows from a higher-level concept through successive layers,
    building a hierarchical explanation graph.

    Parameters
    ----------
    attribution : ConditionalGradient
        Attributor instance to use for computing conditional attributions.
    graph : ModelGraph
        Graph describing the model's layer connectivity.
    layer_map : dict[str, ChannelConcept]
        Mapping from layer names to concept instances for attribution.

    Examples
    --------
    >>> ag = AttributionGraph(attributor, model_graph, layer_map)
    >>> result = ag(sample, concept_id=5, layer_name="conv2")
    >>> result.nodes, result.connections
    """

    def __init__(
        self,
        attribution: ConditionalGradient,
        graph,
        layer_map: dict[str, ChannelConcept],
    ):
        self.attribution = attribution
        self.graph = graph
        self.layer_map = layer_map
        self.mask_map: dict[str, Callable] = {l_name: concept.mask for l_name, concept in layer_map.items()}

    def __call__(
        self,
        sample: torch.Tensor,
        concept_id: int,
        layer_name: str,
        *,
        target: int | None = None,
        width: list[int] | None = None,
        parent_c_id: int | None = None,
        parent_layer: str | None = None,
        abs_norm: bool = True,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> GraphResult:
        """Decompose a concept into lower-level concepts.

        Parameters
        ----------
        sample : torch.Tensor
            Input sample.
        concept_id : int
            Index of the higher-level concept to decompose.
        layer_name : str
            Layer where the concept resides.
        target : int, optional
            If provided, conditions the decomposition on this output target.
        width : list[int], optional
            Number of lower-level concepts to retrieve per layer.
            Length determines the depth of decomposition. Default: ``[4, 2]``.
        parent_c_id : int, optional
            Original higher-level concept index for nested decomposition.
        parent_layer : str, optional
            Layer of the parent concept.
        abs_norm : bool, optional
            Normalize relevances by absolute sum.
        batch_size : int, optional
            Batch size for the generate method.
        verbose : bool, optional
            Show progress bar.

        Returns
        -------
        GraphResult
            Graph of concept nodes and their relevance connections.
        """
        if width is None:
            width = [4, 2]

        nodes: list[tuple[str, int]] = [(layer_name, concept_id)]
        connections: dict[tuple[str, int], list[tuple[str, int, float]]] = {}

        # Determine the starting layer and parent condition
        start_layer = None if target is not None else (parent_layer or layer_name)

        parent_cond: dict[str, list[int]] = {}
        if parent_c_id is not None and parent_layer:
            parent_cond[parent_layer] = [parent_c_id]
        else:
            parent_cond[layer_name] = [concept_id]

        if target is not None:
            parent_cond[MODEL_OUTPUT_NAME] = [target]

        cond_tuples: list[tuple[str, int]] = [(layer_name, concept_id)]

        # Process each depth level
        for w in width:
            conditions: list[dict[str, list[int]]] = []
            input_layers: list[str] = []

            for l_name, c_id in cond_tuples:
                cond = {l_name: [c_id], **parent_cond}
                conditions.append(cond)

                for inp_name in self.graph.find_input_layers(l_name):
                    if inp_name not in input_layers:
                        input_layers.append(inp_name)

            # Compute attributions for all conditions
            next_cond_tuples: list[tuple[str, int]] = []

            for batch_idx, attr in enumerate(
                self.attribution.generate(
                    sample,
                    conditions,
                    record_layers=input_layers,
                    mask_fn=self.mask_map,
                    start_layer=start_layer,
                    batch_size=batch_size,
                    verbose=verbose,
                    exclude_parallel=False,
                )
            ):
                self._decompose_batch(
                    cond_tuples[batch_idx * batch_size : (batch_idx + 1) * batch_size],
                    attr.relevances,
                    w,
                    nodes,
                    connections,
                    next_cond_tuples,
                    abs_norm,
                )

            cond_tuples = next_cond_tuples

        return GraphResult(nodes, connections)

    def _decompose_batch(
        self,
        cond_tuples: list[tuple[str, int]],
        relevances: dict[str, torch.Tensor],
        width: int,
        nodes: list[tuple[str, int]],
        connections: dict[tuple[str, int], list[tuple[str, int, float]]],
        next_cond_tuples: list[tuple[str, int]],
        abs_norm: bool,
    ):
        """Decompose a batch of concept-layer pairs into lower-level concepts."""
        for i, (l_name, c_id) in enumerate(cond_tuples):
            input_layers = self.graph.find_input_layers(l_name)

            for inp_l in input_layers:
                rel = relevances[inp_l][[i]]
                rel_c = self.layer_map[inp_l].attribute(rel, abs_norm=abs_norm)[0]

                top_ids = torch.argsort(rel_c, descending=True)[:width].tolist()
                nodes.extend((inp_l, cid) for cid in top_ids)
                next_cond_tuples.extend((inp_l, cid) for cid in top_ids)

                connections.setdefault((l_name, c_id), []).extend((inp_l, cid, rel_c[cid].item()) for cid in top_ids)
