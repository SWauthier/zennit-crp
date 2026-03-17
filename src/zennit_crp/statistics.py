"""Per-target statistics for concept analysis.

Tracks the most relevant/active samples for each concept, grouped by
target class, enabling target-specific concept explanations.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class Statistics:
    """Track per-target concept statistics (relevance and activation).

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
        self.d_c_sorted: dict[int, dict[str, torch.Tensor]] = {}
        self.rel_c_sorted: dict[int, dict[str, torch.Tensor]] = {}
        self.rf_c_sorted: dict[int, dict[str, torch.Tensor]] = {}

        norm_str = "normed" if abs_norm else "unnormed"
        prefix = "RelStats" if mode == "relevance" else "ActStats"
        if mode not in ("relevance", "activation"):
            raise ValueError("'mode' must be 'relevance' or 'activation'.")

        self.sub_folder = Path(f"{prefix}_{max_target}_{norm_str}")
        self.PATH = Path(path) / self.sub_folder if path else self.sub_folder
        self.PATH.mkdir(parents=True, exist_ok=True)

    def analyze_layer(
        self,
        d_c_sorted: torch.Tensor,
        rel_c_sorted: torch.Tensor,
        rf_c_sorted: torch.Tensor,
        t_c_sorted: torch.Tensor,
        layer_name: str,
    ):
        """Group results by target and merge into running statistics.

        Parameters
        ----------
        d_c_sorted : torch.Tensor
            Dataset indices sorted by relevance.
        rel_c_sorted : torch.Tensor
            Sorted relevance values.
        rf_c_sorted : torch.Tensor
            Sorted receptive field neuron indices.
        t_c_sorted : torch.Tensor
            Target labels (sorted same as above).
        layer_name : str
            Name of the layer.
        """
        for t in torch.unique(t_c_sorted):
            t_indices = t_c_sorted.t() == t
            n_concepts = t_c_sorted.shape[1]

            d_c_t = d_c_sorted.t()[t_indices].view(n_concepts, -1).t()
            rel_c_t = rel_c_sorted.t()[t_indices].view(n_concepts, -1).t()
            rf_c_t = rf_c_sorted.t()[t_indices].view(n_concepts, -1).t()

            target = t.item()
            self._concatenate(layer_name, target, d_c_t, rel_c_t, rf_c_t)
            self._sort(layer_name, target)

    def delete_result_arrays(self):
        """Clear all stored result arrays."""
        self.d_c_sorted.clear()
        self.rel_c_sorted.clear()
        self.rf_c_sorted.clear()
        gc.collect()

    def _concatenate(
        self,
        layer_name: str,
        target: int,
        d_c: torch.Tensor,
        rel_c: torch.Tensor,
        rf_c: torch.Tensor,
    ):
        if target not in self.d_c_sorted:
            self.d_c_sorted[target] = {}
            self.rel_c_sorted[target] = {}
            self.rf_c_sorted[target] = {}

        if layer_name not in self.d_c_sorted[target]:
            self.d_c_sorted[target][layer_name] = d_c
            self.rel_c_sorted[target][layer_name] = rel_c
            self.rf_c_sorted[target][layer_name] = rf_c
        else:
            self.d_c_sorted[target][layer_name] = torch.cat([d_c, self.d_c_sorted[target][layer_name]])
            self.rel_c_sorted[target][layer_name] = torch.cat([rel_c, self.rel_c_sorted[target][layer_name]])
            self.rf_c_sorted[target][layer_name] = torch.cat([rf_c, self.rf_c_sorted[target][layer_name]])

    def _sort(self, layer_name: str, target: int):
        args = torch.argsort(self.rel_c_sorted[target][layer_name], dim=0, descending=True)
        args = args[: self.SAMPLE_SIZE]

        self.rel_c_sorted[target][layer_name] = torch.gather(self.rel_c_sorted[target][layer_name], 0, args)
        self.rf_c_sorted[target][layer_name] = torch.gather(self.rf_c_sorted[target][layer_name], 0, args)
        self.d_c_sorted[target][layer_name] = torch.gather(self.d_c_sorted[target][layer_name], 0, args)

    def _save_results(self, d_index: tuple[int, int] | None = None) -> list[str]:
        saved = []

        for target in self.d_c_sorted:
            for layer_name in self.d_c_sorted[target]:
                prefix = f"{target}_{d_index[0]}_{d_index[1]}_" if d_index else f"{target}_"
                p_path = self.PATH / layer_name
                p_path.mkdir(parents=True, exist_ok=True)

                np.save(p_path / f"{prefix}data.npy", self.d_c_sorted[target][layer_name].cpu().numpy())
                np.save(p_path / f"{prefix}rf.npy", self.rf_c_sorted[target][layer_name].cpu().numpy())
                np.save(p_path / f"{prefix}rel.npy", self.rel_c_sorted[target][layer_name].cpu().numpy())
                saved.append(str(p_path / prefix))

        if d_index is None:
            np.save(self.PATH / "targets.npy", np.array(list(self.d_c_sorted.keys())))

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
            l_name, filename = path.replace("\\", "/").split("/")[-2:]
            target = int(filename.split("_")[0])

            d_c = torch.from_numpy(np.load(path + "data.npy"))
            rf_c = torch.from_numpy(np.load(path + "rf.npy"))
            rel_c = torch.from_numpy(np.load(path + "rel.npy"))

            self._concatenate(l_name, target, d_c, rel_c, rf_c)
            self._sort(l_name, target)

        # Clean up checkpoint files
        for path in path_list:
            for suffix in ("data.npy", "rf.npy", "rel.npy"):
                checkpoint_file = Path(path + suffix)
                checkpoint_file.unlink(missing_ok=True)

        pbar.close()
        return self._save_results(d_index)
