"""Caching system for reference images.

Provides :py:class:`Cache` (abstract base) and :py:class:`ImageCache` for
storing and loading rendered reference images as PIL files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


class Cache:
    """Abstract base for reference image caching.

    Parameters
    ----------
    path : str or Path
        Root directory for cached files.
    """

    def __init__(self, path: str | Path = "cache"):
        self.path = Path(path)

    def save(self, ref_c, layer_name, mode, r_range, composite, rf, f_name, plot_name, **kwargs):
        raise NotImplementedError

    def load(self, concept_ids, layer_name, mode, r_range, composite, rf, f_name, plot_name, **kwargs):
        raise NotImplementedError

    def extend_dict(self, ref_original, rf_addition):
        raise NotImplementedError


class ImageCache(Cache):
    """Cache that saves PIL Image files to disk.

    Stores lists or tuples of lists of :py:class:`PIL.Image.Image` objects
    in a directory structure organized by mode, composite, and layer.

    Parameters
    ----------
    path : str or Path
        Root directory for cached images.
    """

    def _create_path(self, layer_name: str, mode: str, composite, rf: bool, func_name: str, plot_name: str) -> Path:
        folder_name = f"{mode}_{composite.__class__.__name__}"
        if rf:
            folder_name += "_rf"

        path = self.path / func_name / plot_name / folder_name / layer_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_img_list(self, img_list: list, id_val, tuple_index: int, r_range: tuple, path: Path):
        for img, r in zip(img_list, range(*r_range), strict=False):
            if not isinstance(img, Image.Image):
                raise TypeError(f"ImageCache can only save PIL.Image objects, got {type(img)}")
            img.save(path / f"{id_val}_{tuple_index}_{r}.png", optimize=True)

    def save(
        self,
        ref_dict: dict,
        layer_name: str,
        mode: str,
        r_range: tuple[int, int],
        composite,
        rf: bool,
        func_name: str,
        plot_name: str,
    ):
        """Save reference images to disk.

        Parameters
        ----------
        ref_dict : dict
            Dictionary of reference images.
        layer_name : str
            Layer name.
        mode : str
            ``"relevance"`` or ``"activation"``.
        r_range : tuple[int, int]
            Sample index range.
        composite : Composite
            Zennit composite (used for naming).
        rf : bool
            Whether receptive field was used.
        func_name : str
            Name of the calling function.
        plot_name : str
            Name of the plot function.
        """
        path = self._create_path(layer_name, mode, composite, rf, func_name, plot_name)

        for id_key, value in ref_dict.items():
            if isinstance(value, tuple):
                self._save_img_list(value[0], id_key, 0, r_range, path)
                self._save_img_list(value[1], id_key, 1, r_range, path)
            elif isinstance(value[0], Image.Image):
                self._save_img_list(value, id_key, 0, r_range, path)

    def _load_img_list(
        self,
        id_val,
        tuple_index: int,
        r_range: tuple[int, int],
        path: Path,
    ) -> tuple[list, tuple[int, int] | None]:
        """Load a list of images for a given ID and tuple index.

        Returns
        -------
        tuple[list, tuple[int, int] | None]
            ``(images, not_found_range)`` where ``not_found_range`` is the
            remaining range if any image is missing.
        """
        imgs: list = []
        not_found = None
        for r in range(*r_range):
            img_path = path / f"{id_val}_{tuple_index}_{r}.png"
            if img_path.exists():
                imgs.append(Image.open(img_path))
            else:
                not_found = (r, r_range[-1])
                break
        return imgs, not_found

    def load(
        self,
        concept_ids: list,
        layer_name: str,
        mode: str,
        r_range: tuple[int, int],
        composite,
        rf: bool,
        func_name: str,
        plot_name: str,
    ) -> tuple[dict, dict]:
        """Load cached reference images from disk.

        Supports both single-list and tuple-of-lists formats (e.g. images
        returned by :py:func:`~zennit_crp.image.vis_img_heatmap`).

        Parameters
        ----------
        concept_ids : list
            Concept indices to load.
        layer_name : str
            Layer name.
        mode : str
            ``"relevance"`` or ``"activation"``.
        r_range : tuple[int, int]
            Sample index range.
        composite : Composite
            Zennit composite.
        rf : bool
            Whether receptive field was used.
        func_name : str
            Name of the calling function.
        plot_name : str
            Name of the plot function.

        Returns
        -------
        tuple[dict, dict]
            ``(found, not_found)`` where ``found`` contains loaded images
            and ``not_found`` maps missing IDs to their ``r_range``.
        """
        path = self._create_path(layer_name, mode, composite, rf, func_name, plot_name)

        found: dict[Any, Any] = {}
        not_found: dict[Any, tuple[int, int]] = {}

        for c_id in concept_ids:
            imgs_0, not_found_0 = self._load_img_list(c_id, 0, r_range, path)
            imgs_1, _ = self._load_img_list(c_id, 1, r_range, path)

            if imgs_0:
                if imgs_1:
                    found[c_id] = (imgs_0, imgs_1)
                else:
                    found[c_id] = imgs_0

            if not_found_0:
                not_found[c_id] = not_found_0

        return found, not_found

    def extend_dict(self, ref_original: dict, ref_addition: dict) -> dict:
        """Merge two reference dictionaries.

        For tuple-of-lists values (e.g. from :py:func:`vis_img_heatmap`),
        extends each sub-list independently.
        """
        for key, value in ref_addition.items():
            if key in ref_original:
                if isinstance(value, tuple):
                    ref_original[key][0].extend(value[0])
                    ref_original[key][1].extend(value[1])
                elif isinstance(value, list):
                    ref_original[key].extend(value)
                else:
                    raise TypeError("Values must be lists or tuples of lists.")
            else:
                ref_original[key] = value
        return ref_original
