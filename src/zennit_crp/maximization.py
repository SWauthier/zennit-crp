"""Relevance and activation maximization for reference sampling.

Tracks the top-N most relevant or active samples per concept across
dataset batches, supporting incremental analysis with checkpointing.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from zennit_crp.concepts import ChannelConcept


class Maximization:
    """Track top reference samples per concept by relevance or activation.

    Parameters
    ----------
    mode : str
        ``"relevance"`` or ``"activation"``.
    max_target : str
        Aggregation mode (``"sum"`` or ``"max"``).
    abs_norm : bool
        Whether to normalize by absolute sum.
    path : str or Path, optional
        Root directory for saving results.
    """

    SAMPLE_SIZE = 40

    def __init__(
        self,
        mode: str = "relevance",
        max_target: str = "sum",
        abs_norm: bool = False,
        path: str | Path | None = None,
    ):
        self.d_c_sorted: dict[str, torch.Tensor] = {}
        self.rel_c_sorted: dict[str, torch.Tensor] = {}
        self.rf_c_sorted: dict[str, torch.Tensor] = {}

        self.max_target = max_target
        self.abs_norm = abs_norm

        norm_str = "normed" if abs_norm else "unnormed"
        prefix = "RelMax" if mode == "relevance" else "ActMax"
        if mode not in ("relevance", "activation"):
            raise ValueError("'mode' must be 'relevance' or 'activation'.")

        self.sub_folder = Path(f"{prefix}_{max_target}_{norm_str}")
        self.PATH = Path(path) / self.sub_folder if path else self.sub_folder
        self.PATH.mkdir(parents=True, exist_ok=True)

    def analyze_layer(
        self,
        rel: torch.Tensor,
        concept: ChannelConcept,
        layer_name: str,
        data_indices: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analyze a batch of relevances/activations for a layer.

        Parameters
        ----------
        rel : torch.Tensor
            Relevance or activation tensor.
        concept : ChannelConcept
            Concept instance for reference sampling.
        layer_name : str
            Name of the layer.
        data_indices : np.ndarray
            Dataset indices for this batch.
        targets : np.ndarray
            Target labels for this batch.

        Returns
        -------
        tuple[torch.Tensor, ...]
            ``(d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted)``.
        """
        b_c_sorted, rel_c_sorted, rf_c_sorted = concept.reference_sampling(
            rel,
            self.max_target,
            self.abs_norm,
        )

        data_indices_t = torch.from_numpy(data_indices).to(b_c_sorted)
        d_c_sorted = torch.take(data_indices_t, b_c_sorted)

        targets_t = torch.tensor(targets).to(b_c_sorted)
        t_c_sorted = torch.take(targets_t, b_c_sorted)

        sz = self.SAMPLE_SIZE
        self._concatenate(layer_name, d_c_sorted[:sz], rel_c_sorted[:sz], rf_c_sorted[:sz])
        self._sort(layer_name)

        return d_c_sorted, rel_c_sorted, rf_c_sorted, t_c_sorted

    def delete_result_arrays(self):
        """Clear all stored result arrays."""
        self.d_c_sorted.clear()
        self.rel_c_sorted.clear()
        self.rf_c_sorted.clear()
        gc.collect()

    def _concatenate(self, layer_name: str, d_c: torch.Tensor, rel_c: torch.Tensor, rf_c: torch.Tensor):
        if layer_name not in self.d_c_sorted:
            self.d_c_sorted[layer_name] = d_c
            self.rel_c_sorted[layer_name] = rel_c
            self.rf_c_sorted[layer_name] = rf_c
        else:
            self.d_c_sorted[layer_name] = torch.cat([d_c, self.d_c_sorted[layer_name]])
            self.rel_c_sorted[layer_name] = torch.cat([rel_c, self.rel_c_sorted[layer_name]])
            self.rf_c_sorted[layer_name] = torch.cat([rf_c, self.rf_c_sorted[layer_name]])

    def _sort(self, layer_name: str):
        args = torch.argsort(self.rel_c_sorted[layer_name], dim=0, descending=True)
        args = args[: self.SAMPLE_SIZE]

        self.rel_c_sorted[layer_name] = torch.gather(self.rel_c_sorted[layer_name], 0, args)
        self.rf_c_sorted[layer_name] = torch.gather(self.rf_c_sorted[layer_name], 0, args)
        self.d_c_sorted[layer_name] = torch.gather(self.d_c_sorted[layer_name], 0, args)

    def _save_results(self, d_index: tuple[int, int] | None = None) -> list[str]:
        saved = []
        for layer_name in self.d_c_sorted:
            prefix = f"{layer_name}_{d_index[0]}_{d_index[1]}_" if d_index else f"{layer_name}_"
            np.save(self.PATH / f"{prefix}data.npy", self.d_c_sorted[layer_name].cpu().numpy())
            np.save(self.PATH / f"{prefix}rf.npy", self.rf_c_sorted[layer_name].cpu().numpy())
            np.save(self.PATH / f"{prefix}rel.npy", self.rel_c_sorted[layer_name].cpu().numpy())
            saved.append(str(self.PATH / prefix))

        self.delete_result_arrays()
        return saved

    def collect_results(self, path_list: list[str], d_index: tuple[int, int] | None = None) -> list[str]:
        """Collect and merge results from checkpoint files.

        Parameters
        ----------
        path_list : list[str]
            Paths to checkpoint result files (without suffix).
        d_index : tuple[int, int], optional
            Dataset index range for the final save.

        Returns
        -------
        list[str]
            Paths to the final merged result files.
        """
        self.delete_result_arrays()
        pbar = tqdm(total=len(path_list), dynamic_ncols=True)

        for path in path_list:
            pbar.update(1)
            filename = path.replace("\\", "/").split("/")[-1]
            l_name = re.split(r"_[0-9]+_[0-9]+_\b", filename)[0]

            d_c, rf_c, rel_c = (
                torch.from_numpy(np.load(path + "data.npy")),
                torch.from_numpy(np.load(path + "rf.npy")),
                torch.from_numpy(np.load(path + "rel.npy")),
            )

            self._concatenate(l_name, d_c, rel_c, rf_c)
            self._sort(l_name)

        pbar.close()
        return self._save_results(d_index)
