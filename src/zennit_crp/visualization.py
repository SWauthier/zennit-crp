"""Feature visualization using CRP reference sampling.

Provides :py:class:`FeatureVisualization` which computes relevance and
activation maximization statistics, and retrieves reference samples for
interpreting individual concepts.
"""

from __future__ import annotations

import concurrent.futures
import functools
import inspect
import math
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from zennit.composites import Composite, NameMapComposite

from zennit_crp.attribution import ConditionalGradient
from zennit_crp.cache import Cache
from zennit_crp.concepts import ChannelConcept
from zennit_crp.conditions import MODEL_OUTPUT_NAME
from zennit_crp.helper import load_maximization, load_stat_targets, load_statistics
from zennit_crp.hooks import FeatVisHook
from zennit_crp.image import vis_img_heatmap, vis_opaque_img
from zennit_crp.maximization import Maximization
from zennit_crp.statistics import Statistics


def _cache_reference(func: Callable) -> Callable:
    """Decorator for caching reference images in reference retrieval methods.

    If a :py:class:`~zennit_crp.cache.Cache` is available on the
    ``FeatureVisualization`` instance, previously computed reference images
    are loaded from disk. Missing entries are computed, saved, and merged
    into the result. Accepts an ``overwrite`` keyword argument to force
    recomputation.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        overwrite = kwargs.pop("overwrite", False)

        # Bind arguments to get named parameters
        sig = inspect.signature(func)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)

        plot_fn = params.get("plot_fn")
        if self.Cache is None or plot_fn is None:
            return func(**params)

        r_range = params["r_range"]
        mode = params["mode"]
        layer_name = params["layer_name"]
        rf = params["rf"]
        composite = params["composite"]
        f_name = func.__name__
        plot_name = plot_fn.__name__

        # Determine indices depending on which method is decorated
        if f_name == "get_max_reference":
            concept_ids = params["concept_ids"]
            if not isinstance(concept_ids, Iterable):
                concept_ids = [concept_ids]
            indices = list(concept_ids)
        else:
            concept_id = params["concept_id"]
            targets = params["targets"]
            if not isinstance(targets, Iterable):
                targets = [targets]
            indices = [f"{concept_id}:{t}" for t in targets]

        # Try to load from cache
        if overwrite:
            not_found = {idx: r_range for idx in indices}
            ref_c: dict = {}
        else:
            ref_c, not_found = self.Cache.load(
                indices,
                layer_name,
                mode,
                r_range,
                composite,
                rf,
                f_name,
                plot_name,
            )

        # Compute and cache missing entries
        if not_found:
            for idx in not_found:
                params["r_range"] = not_found[idx]
                if f_name == "get_max_reference":
                    params["concept_ids"] = idx
                else:
                    params["targets"] = int(str(idx).split(":")[-1])

                ref_partial = func(**params)
                self.Cache.save(
                    ref_partial,
                    layer_name,
                    mode,
                    not_found[idx],
                    composite,
                    rf,
                    f_name,
                    plot_name,
                )
                ref_c = self.Cache.extend_dict(ref_c, ref_partial)

        return ref_c

    return wrapper


class FeatureVisualization:
    """Compute and retrieve reference samples for concept-level explanations.

    Runs the dataset through the model, recording per-layer activations and
    relevances. These are aggregated by :py:class:`Maximization` and
    :py:class:`Statistics` to identify the most relevant/active samples per
    concept and per target class.

    Parameters
    ----------
    attribution : ConditionalGradient
        Attributor for conditional heatmap computation on reference samples.
    dataset : torch.utils.data.Dataset
        Dataset providing ``(data, target)`` pairs.
    layer_map : dict[str, ChannelConcept]
        Mapping from layer names to concept instances.
    preprocess_fn : callable, optional
        Preprocessing function applied to data before attribution.
    max_target : str
        Aggregation mode for reference sampling (``"sum"`` or ``"max"``).
    abs_norm : bool
        Whether to normalize relevances by absolute sum.
    path : str
        Directory for saving analysis results.
    device : str or torch.device, optional
        Device for computation. Defaults to the attributor's device.
    cache : Cache, optional
        Cache for storing/loading rendered reference images.
    """

    def __init__(
        self,
        attribution: ConditionalGradient,
        dataset,
        layer_map: dict[str, ChannelConcept],
        preprocess_fn: Callable | None = None,
        max_target: str = "sum",
        abs_norm: bool = True,
        path: str = "FeatureVisualization",
        device: str | torch.device | None = None,
        cache: Cache | None = None,
    ):
        self.dataset = dataset
        self.layer_map = layer_map
        self.preprocess_fn = preprocess_fn
        self.attribution = attribution
        self.device = device or next(attribution.model.parameters()).device

        self.RelMax = Maximization("relevance", max_target, abs_norm, path)
        self.ActMax = Maximization("activation", max_target, abs_norm, path)
        self.RelStats = Statistics("relevance", max_target, abs_norm, path)
        self.ActStats = Statistics("activation", max_target, abs_norm, path)
        self.Cache = cache

    def preprocess_data(self, data: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to input data."""
        if callable(self.preprocess_fn):
            return self.preprocess_fn(data)
        return data

    def get_data_sample(self, index: int, preprocessing: bool = True) -> tuple[torch.Tensor, int]:
        """Load a single sample from the dataset.

        Parameters
        ----------
        index : int
            Dataset index.
        preprocessing : bool
            Whether to apply :py:meth:`preprocess_data`.

        Returns
        -------
        tuple[torch.Tensor, int]
            ``(data, target)`` with data on device.
        """
        data, target = self.dataset[index]
        data = data.to(self.device).unsqueeze(0)
        if preprocessing:
            data = self.preprocess_data(data)
        data.requires_grad = True
        return data, target

    def multitarget_to_single(self, multi_target):
        """Convert a multi-target label to single targets.

        Override this method for multi-label datasets. By default raises
        ``NotImplementedError``, causing the run loop to skip conversion.
        """
        raise NotImplementedError

    def run(
        self,
        composite: Composite,
        data_start: int,
        data_end: int,
        batch_size: int = 32,
        checkpoint: int = 500,
        on_device: str | torch.device | None = None,
    ) -> dict[str, list[str]]:
        """Run the full analysis pipeline.

        Parameters
        ----------
        composite : Composite
            Zennit composite for LRP rules.
        data_start : int
            Start index in the dataset (inclusive).
        data_end : int
            End index in the dataset (exclusive).
        batch_size : int
            Batch size for processing.
        checkpoint : int
            Save intermediate results every ``checkpoint`` batches.
        on_device : str or torch.device, optional
            Device for intermediate results.

        Returns
        -------
        dict[str, list[str]]
            Paths to saved result files.
        """
        print("Running Analysis...")
        saved_checkpoints = self._run_batched(
            composite,
            data_start,
            data_end,
            batch_size,
            checkpoint,
            on_device,
        )

        print("Collecting results...")
        return self.collect_results(saved_checkpoints)

    def _run_batched(
        self,
        composite: Composite,
        data_start: int,
        data_end: int,
        batch_size: int = 16,
        checkpoint: int = 500,
        on_device: str | torch.device | None = None,
    ) -> dict[str, list[str]]:
        """Run analysis in batches with checkpointing."""
        saved_checkpoints: dict[str, list[str]] = {
            "r_max": [],
            "a_max": [],
            "r_stats": [],
            "a_stats": [],
        }
        last_checkpoint = 0

        n_samples = data_end - data_start
        samples = np.arange(start=data_start, stop=data_end)
        batches = max(1, math.ceil(n_samples / batch_size))
        batch_size = min(batch_size, n_samples)

        # Register FeatVisHooks for all layers
        name_map = []
        context: dict[str, Any] = {}
        for l_name, concept in self.layer_map.items():
            hook = FeatVisHook(self, concept, l_name, context, on_device)
            name_map.append(([l_name], hook))
        fv_composite = NameMapComposite(name_map)

        if composite:
            composite.register(self.attribution.model)
        fv_composite.register(self.attribution.model)

        pbar = tqdm(total=batches, dynamic_ncols=True)

        for b in range(batches):
            pbar.update(1)

            samples_batch = samples[b * batch_size : (b + 1) * batch_size]
            data_batch, targets_samples = self._get_data_concurrently(
                samples_batch,
                preprocessing=True,
            )

            targets_samples = np.array(targets_samples)

            # Handle multi-target datasets
            data_broadcast, targets, sample_indices = self._broadcast_targets(
                data_batch,
                targets_samples,
                samples_batch,
            )
            if data_broadcast is None:
                continue

            conditions = [{MODEL_OUTPUT_NAME: [int(t)]} for t in targets]
            context["sample_indices"] = sample_indices
            context["targets"] = targets

            # Composites are already registered
            self.attribution(data_broadcast, conditions)

            if b % checkpoint == checkpoint - 1:
                self._save_results(saved_checkpoints, (last_checkpoint, sample_indices[-1] + 1))
                last_checkpoint = sample_indices[-1] + 1

        self._save_results(saved_checkpoints, (last_checkpoint, sample_indices[-1] + 1))

        if composite:
            composite.remove()
        fv_composite.remove()
        pbar.close()

        return saved_checkpoints

    def _broadcast_targets(
        self,
        data_batch: torch.Tensor,
        targets_samples: np.ndarray,
        samples_batch: np.ndarray,
    ) -> tuple[torch.Tensor | None, np.ndarray, np.ndarray]:
        """Convert multi-target to single-target if needed."""
        try:
            data_broadcast, targets, sample_indices = [], [], []
            for i_t, target in enumerate(targets_samples):
                single_targets = self.multitarget_to_single(target)
                for st in single_targets:
                    targets.append(st)
                    data_broadcast.append(data_batch[i_t])
                    sample_indices.append(samples_batch[i_t])
            if not data_broadcast:
                return None, np.array([]), np.array([])
            return (
                torch.stack(data_broadcast),
                np.array(targets),
                np.array(sample_indices),
            )
        except NotImplementedError:
            return data_batch, targets_samples, samples_batch

    @torch.no_grad()
    def analyze_relevance(
        self,
        rel: torch.Tensor,
        layer_name: str,
        concept: ChannelConcept,
        data_indices: np.ndarray,
        targets: np.ndarray,
    ):
        """Analyze relevance for a layer (called from FeatVisHook)."""
        d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted = self.RelMax.analyze_layer(
            rel,
            concept,
            layer_name,
            data_indices,
            targets,
        )
        self.RelStats.analyze_layer(
            d_c_sorted,
            rel_c_sorted,
            rf_c_sorted,
            t_c_sorted,
            layer_name,
        )

    @torch.no_grad()
    def analyze_activation(
        self,
        act: torch.Tensor,
        layer_name: str,
        concept: ChannelConcept,
        data_indices: np.ndarray,
        targets: np.ndarray,
    ):
        """Analyze activations for a layer (called from FeatVisHook)."""
        unique_indices = np.unique(data_indices, return_index=True)[1]
        data_indices = data_indices[unique_indices]
        act = act[unique_indices]
        targets = targets[unique_indices]

        d_c_sorted, act_c_sorted, rf_c_sorted, t_c_sorted = self.ActMax.analyze_layer(
            act,
            concept,
            layer_name,
            data_indices,
            targets,
        )
        self.ActStats.analyze_layer(
            d_c_sorted,
            act_c_sorted,
            rf_c_sorted,
            t_c_sorted,
            layer_name,
        )

    def _save_results(
        self,
        checkpoints: dict[str, list[str]],
        d_index: tuple[int, int] | None = None,
    ):
        checkpoints["r_max"].extend(self.RelMax._save_results(d_index))
        checkpoints["a_max"].extend(self.ActMax._save_results(d_index))
        checkpoints["r_stats"].extend(self.RelStats._save_results(d_index))
        checkpoints["a_stats"].extend(self.ActStats._save_results(d_index))

    def collect_results(
        self,
        checkpoints: dict[str, list[str]],
        d_index: tuple[int, int] | None = None,
    ) -> dict[str, list[str]]:
        """Collect and merge checkpoint results."""
        return {
            "r_max": self.RelMax.collect_results(checkpoints["r_max"], d_index),
            "a_max": self.ActMax.collect_results(checkpoints["a_max"], d_index),
            "r_stats": self.RelStats.collect_results(checkpoints["r_stats"], d_index),
            "a_stats": self.ActStats.collect_results(checkpoints["a_stats"], d_index),
        }

    def _get_data_concurrently(
        self,
        indices: np.ndarray | list,
        preprocessing: bool = False,
    ) -> tuple[torch.Tensor, list]:
        """Load multiple samples concurrently using threads."""
        if len(indices) == 1:
            data, label = self.get_data_sample(indices[0], preprocessing)
            return data, [label]

        data_list, labels = [], []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_data_sample, idx, preprocessing) for idx in indices]
            for future in futures:
                sample, label = future.result()
                data_list.append(sample)
                labels.append(label)

        return torch.cat(data_list), labels

    # --- Reference retrieval ---

    @_cache_reference
    def get_max_reference(
        self,
        concept_ids: int | list[int],
        layer_name: str,
        mode: str = "relevance",
        r_range: tuple[int, int] = (0, 8),
        composite: Composite | None = None,
        rf: bool = False,
        plot_fn: Callable | None = vis_img_heatmap,
        batch_size: int = 32,
    ) -> dict:
        """Retrieve reference samples for concepts maximizing relevance or activation.

        Parameters
        ----------
        concept_ids : int or list[int]
            Concept (channel) indices.
        layer_name : str
            Layer name.
        mode : str
            ``"relevance"`` or ``"activation"``.
        r_range : tuple[int, int]
            Range of top-N reference samples.
        composite : Composite, optional
            Composite for computing conditional heatmaps on references.
        rf : bool
            Whether to crop to receptive field.
        plot_fn : callable, optional
            Visualization function. If ``None``, returns raw tensors.
        batch_size : int
            Batch size for heatmap computation.

        Returns
        -------
        dict
            Keyed by concept index. Values depend on ``plot_fn``.
        """
        ref_c: dict = {}
        if not isinstance(concept_ids, Iterable):
            concept_ids = [concept_ids]

        if mode == "relevance":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.RelMax.PATH, layer_name)
        elif mode == "activation":
            d_c_sorted, _, rf_c_sorted = load_maximization(self.ActMax.PATH, layer_name)
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        if rf and not composite:
            warnings.warn(
                "The receptive field is only computed if 'composite' is provided.",
                stacklevel=2,
            )

        for c_id in concept_ids:
            d_indices = d_c_sorted[r_range[0] : r_range[1], c_id]
            n_indices = rf_c_sorted[r_range[0] : r_range[1], c_id]
            ref_c[c_id] = self._load_ref_and_attribute(
                d_indices,
                c_id,
                n_indices,
                layer_name,
                composite,
                rf,
                plot_fn,
                batch_size,
            )

        return ref_c

    @_cache_reference
    def get_stats_reference(
        self,
        concept_id: int,
        layer_name: str,
        targets: int | list[int],
        mode: str = "relevance",
        r_range: tuple[int, int] = (0, 8),
        composite: Composite | None = None,
        rf: bool = False,
        plot_fn: Callable | None = vis_img_heatmap,
        batch_size: int = 32,
    ) -> dict:
        """Retrieve reference samples for a concept across different targets.

        Parameters
        ----------
        concept_id : int
            Concept index.
        layer_name : str
            Layer name.
        targets : int or list[int]
            Target class indices.
        mode : str
            ``"relevance"`` or ``"activation"``.
        r_range : tuple[int, int]
            Range of top-N reference samples.
        composite : Composite, optional
            Composite for conditional heatmaps.
        rf : bool
            Crop to receptive field.
        plot_fn : callable, optional
            Visualization function.
        batch_size : int
            Batch size for heatmap computation.

        Returns
        -------
        dict
            Keyed by ``"concept_id:target"``.
        """
        ref_t: dict = {}
        if not isinstance(targets, Iterable):
            targets = [targets]

        if mode == "relevance":
            path = self.RelStats.PATH
        elif mode == "activation":
            path = self.ActStats.PATH
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        if rf and not composite:
            warnings.warn(
                "The receptive field is only computed if 'composite' is provided.",
                stacklevel=2,
            )

        for t in targets:
            d_c_sorted, _, rf_c_sorted = load_statistics(path, layer_name, t)
            d_indices = d_c_sorted[r_range[0] : r_range[1], concept_id]
            n_indices = rf_c_sorted[r_range[0] : r_range[1], concept_id]

            ref_t[f"{concept_id}:{t}"] = self._load_ref_and_attribute(
                d_indices,
                concept_id,
                n_indices,
                layer_name,
                composite,
                rf,
                plot_fn,
                batch_size,
            )

        return ref_t

    def _load_ref_and_attribute(
        self,
        d_indices: torch.Tensor,
        c_id: int,
        n_indices: torch.Tensor,
        layer_name: str,
        composite: Composite | None,
        rf: bool,
        plot_fn: Callable | None,
        batch_size: int,
    ):
        """Load reference samples and optionally compute conditional heatmaps."""
        data_batch, _ = self._get_data_concurrently(d_indices, preprocessing=False)

        if composite:
            data_p = self.preprocess_data(data_batch)
            heatmaps = self._attribution_on_reference(
                data_p,
                c_id,
                layer_name,
                composite,
                rf,
                n_indices,
                batch_size,
            )
            if callable(plot_fn):
                return plot_fn(data_batch.detach(), heatmaps.detach(), rf)
            return data_batch.detach().cpu(), heatmaps.detach().cpu()

        return data_batch.detach().cpu()

    def _attribution_on_reference(
        self,
        data: torch.Tensor,
        concept_id: int,
        layer_name: str,
        composite: Composite,
        rf: bool = False,
        neuron_ids: torch.Tensor | list | None = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Compute conditional heatmaps for reference samples."""
        n_samples = len(data)
        batches = max(1, math.ceil(n_samples / batch_size))
        batch_size = min(batch_size, n_samples)

        if rf and (neuron_ids is None or len(neuron_ids) != n_samples):
            raise ValueError("'neuron_ids' must have same length as 'data' when rf=True")

        heatmaps = []
        with ConditionalGradient(self.attribution.model, composite=composite) as attributor:
            for b in range(batches):
                data_batch = data[b * batch_size : (b + 1) * batch_size].detach().requires_grad_()

                if rf:
                    batch_neuron_ids = neuron_ids[b * batch_size : (b + 1) * batch_size]
                    conditions = [{layer_name: {concept_id: int(n_idx)}} for n_idx in batch_neuron_ids]
                    attr = attributor(
                        data_batch,
                        conditions,
                        mask_fn=ChannelConcept.mask_rf,
                        start_layer=layer_name,
                        on_device=self.device,
                    )
                else:
                    conditions = [{layer_name: [concept_id]}]
                    attr = attributor(
                        data_batch,
                        conditions,
                        start_layer=layer_name,
                        on_device=self.device,
                    )

                heatmaps.append(attr.heatmap)

        return torch.cat(heatmaps)

    def compute_stats(
        self,
        concept_id: int,
        layer_name: str,
        mode: str = "relevance",
        top_N: int = 5,
        mean_N: int = 10,
        norm: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Compute statistics about which targets a concept is most relevant for.

        Parameters
        ----------
        concept_id : int
            Concept index.
        layer_name : str
            Layer name.
        mode : str
            ``"relevance"`` or ``"activation"``.
        top_N : int
            Number of top targets to return.
        mean_N : int
            Number of top samples to average for importance.
        norm : bool
            Normalize values relative to the top target.

        Returns
        -------
        tuple[np.ndarray, torch.Tensor]
            ``(sorted_targets, sorted_values)``.
        """
        if mode == "relevance":
            path = self.RelStats.PATH
        elif mode == "activation":
            path = self.ActStats.PATH
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        targets = load_stat_targets(path)
        rel_target = torch.zeros(len(targets))

        for i, t in enumerate(targets):
            _, rel_c_sorted, _ = load_statistics(path, layer_name, t)
            rel_target[i] = float(rel_c_sorted[:mean_N, concept_id].mean())

        args = torch.argsort(rel_target, descending=True)[:top_N]
        sorted_t = targets[args]
        sorted_val = rel_target[args]

        if norm:
            sorted_val = sorted_val / sorted_val[0]

        return sorted_t, sorted_val

    def precompute_ref(
        self,
        layer_c_ind: dict[str, list[int]],
        composite: Composite,
        rf: bool = True,
        stats: bool = False,
        top_N: int = 4,
        mean_N: int = 10,
        mode: str = "relevance",
        r_range: tuple[int, int] = (0, 8),
        plot_list: list[Callable] | None = None,
        batch_size: int = 32,
    ):
        """Precompute and save reference images for the given concepts and layers.

        Calls :py:meth:`get_max_reference` (and optionally
        :py:meth:`get_stats_reference`) for every concept in ``layer_c_ind``
        and writes the results to the :py:class:`~zennit_crp.cache.Cache`.

        Parameters
        ----------
        layer_c_ind : dict[str, list[int]]
            Mapping from layer names to lists of concept indices to precompute.
        composite : Composite
            Zennit composite for LRP rules.
        rf : bool, optional
            Crop reference images to the receptive field. Default is ``True``.
        stats : bool, optional
            If ``True``, also precompute statistics-based references via
            :py:meth:`get_stats_reference`. Default is ``False``.
        top_N : int, optional
            Number of top target classes for statistics. Default is ``4``.
        mean_N : int, optional
            Number of samples used to compute per-target importance. Default is ``10``.
        mode : str, optional
            ``"relevance"`` or ``"activation"``. Default is ``"relevance"``.
        r_range : tuple[int, int], optional
            Range of top-N reference samples. Default is ``(0, 8)``.
        plot_list : list[callable], optional
            Visualization functions. Default is ``[vis_opaque_img]``.
        batch_size : int, optional
            Batch size for heatmap computation. Default is ``32``.

        Raises
        ------
        ValueError
            If :py:attr:`Cache` or ``composite`` is ``None``.
        """
        if self.Cache is None:
            raise ValueError("A Cache must be provided to precompute reference images.")
        if composite is None:
            raise ValueError("A zennit Composite must be provided to precompute reference images.")

        if plot_list is None:
            plot_list = [vis_opaque_img]

        for l_name, c_indices in layer_c_ind.items():
            print("Layer:", l_name)
            pbar = tqdm(total=len(c_indices), dynamic_ncols=True)

            for c_id in c_indices:
                # Compute reference images with each plot function
                s_tensor, h_tensor = self.get_max_reference(
                    c_id,
                    l_name,
                    mode,
                    r_range,
                    composite,
                    rf,
                    None,
                    batch_size,
                )[c_id]

                for plot_fn in plot_list:
                    ref = {c_id: plot_fn(s_tensor, h_tensor, rf)}
                    self.Cache.save(
                        ref,
                        l_name,
                        mode,
                        r_range,
                        composite,
                        rf,
                        "get_max_reference",
                        plot_fn.__name__,
                    )

                # Optionally precompute per-target statistics references
                if stats:
                    targets, _ = self.compute_stats(c_id, l_name, mode, top_N, mean_N)
                    for t in targets:
                        stat_index = f"{c_id}:{t}"
                        s_tensor, h_tensor = self.get_stats_reference(
                            c_id,
                            l_name,
                            t,
                            mode,
                            r_range,
                            composite,
                            rf,
                            None,
                            batch_size,
                        )[stat_index]

                        for plot_fn in plot_list:
                            ref = {stat_index: plot_fn(s_tensor, h_tensor, rf)}
                            self.Cache.save(
                                ref,
                                l_name,
                                mode,
                                r_range,
                                composite,
                                rf,
                                "get_stats_reference",
                                plot_fn.__name__,
                            )

                pbar.update(1)

            pbar.close()
