"""Image utilities for CRP visualization.

Provides functions for rendering reference images with opacity masks,
conditional heatmaps, and grid plotting.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import zennit.image as zimage
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from zennit_crp.helper import max_norm


def get_crop_range(heatmap: torch.Tensor, crop_th: float) -> tuple[int, int, int, int]:
    """Return row/column indices to crop a heatmap to its relevant region.

    Parameters
    ----------
    heatmap : torch.Tensor
        2D heatmap tensor.
    crop_th : float
        Threshold in ``[0, 1)`` relative to max relevance.

    Returns
    -------
    tuple[int, int, int, int]
        ``(row_start, row_end, col_start, col_end)``.
    """
    crop_mask = heatmap > crop_th
    rows, columns = torch.where(crop_mask)

    if len(rows) == 0 or len(columns) == 0:
        return 0, -1, 0, -1

    row1, row2 = rows.min().item(), rows.max().item()
    col1, col2 = columns.min().item(), columns.max().item()

    if row1 >= row2 and col1 >= col2:
        return 0, -1, 0, -1

    return row1, row2, col1, col2


@torch.no_grad()
def vis_opaque_img(
    data_batch: torch.Tensor,
    heatmaps: torch.Tensor,
    rf: bool = False,
    alpha: float = 0.3,
    vis_th: float = 0.2,
    crop_th: float = 0.1,
    kernel_size: int = 19,
) -> list[Image.Image]:
    """Render reference images with opacity based on relevance.

    Lowers opacity in regions where relevance is below ``vis_th * max(relevance)``.
    Optionally crops to the receptive field region.

    Parameters
    ----------
    data_batch : torch.Tensor
        Original images (before preprocessing).
    heatmaps : torch.Tensor
        Attribution heatmaps.
    rf : bool
        If ``True``, crop images to the receptive field.
    alpha : float
        Transparency for low-relevance regions, in ``[0, 1]``.
    vis_th : float
        Visibility threshold in ``[0, 1)``.
    crop_th : float
        Cropping threshold in ``[0, 1)``.
    kernel_size : int
        Gaussian blur kernel size for smoothing heatmaps.

    Returns
    -------
    list[Image.Image]
        Rendered images.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("'alpha' must be between [0, 1]")
    if not 0 <= vis_th < 1:
        raise ValueError("'vis_th' must be between [0, 1)")
    if not 0 <= crop_th < 1:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):
        img = data_batch[i]
        filtered_heat = max_norm(gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0])
        vis_mask = filtered_heat > vis_th

        if rf:
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)
            img_t = img[..., row1:row2, col1:col2]
            vis_mask_t = vis_mask[row1:row2, col1:col2]

            if img_t.sum() != 0 and vis_mask_t.sum() != 0:
                img = img_t
                vis_mask = vis_mask_t

        inv_mask = ~vis_mask
        img = img * vis_mask + img * inv_mask * alpha
        imgs.append(zimage.imgify(img.detach().cpu()))

    return imgs


@torch.no_grad()
def vis_img_heatmap(
    data_batch: torch.Tensor,
    heatmaps: torch.Tensor,
    rf: bool = False,
    crop_th: float = 0.1,
    kernel_size: int = 19,
    cmap: str = "bwr",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = True,
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Render reference images alongside their conditional heatmaps.

    Parameters
    ----------
    data_batch : torch.Tensor
        Original images (before preprocessing).
    heatmaps : torch.Tensor
        Attribution heatmaps.
    rf : bool
        If ``True``, crop to the receptive field.
    crop_th : float
        Cropping threshold in ``[0, 1)``.
    kernel_size : int
        Gaussian blur kernel size.
    cmap : str
        Colormap for heatmaps.
    vmin, vmax : float, optional
        Value range for colormap normalization.
    symmetric : bool
        Whether to use symmetric color normalization.

    Returns
    -------
    tuple[list[Image.Image], list[Image.Image]]
        ``(images, heatmap_images)``.
    """
    img_list, heat_list = [], []

    for i in range(len(data_batch)):
        img = data_batch[i]
        heat = heatmaps[i]

        if rf:
            filtered_heat = max_norm(gaussian_blur(heat.unsqueeze(0), kernel_size=kernel_size)[0])
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)
            img_t = img[..., row1:row2, col1:col2]
            heat_t = heat[row1:row2, col1:col2]

            if img_t.sum() != 0 and heat_t.sum() != 0:
                img = img_t
                heat = heat_t

        heat_list.append(imgify(heat, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric))
        img_list.append(imgify(img))

    return img_list, heat_list


def imgify(
    image: Image.Image | torch.Tensor | np.ndarray,
    cmap: str = "bwr",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    level: float = 1.0,
    grid: bool = False,
    gridfill: float | None = None,
    resize: int | None = None,
    padding: bool = False,
) -> Image.Image:
    """Convert a tensor, array, or PIL Image to a displayable PIL Image.

    Wrapper around :py:func:`zennit.image.imgify` with optional resizing
    (preserving aspect ratio) and square padding.

    Parameters
    ----------
    image : Image.Image, torch.Tensor, or np.ndarray
        Input image.
    cmap : str
        Colormap for single-channel images.
    vmin, vmax : float, optional
        Value range for normalization.
    symmetric : bool
        Symmetric color normalization.
    level : float
        Colormap level.
    grid : bool
        Whether to add grid lines.
    gridfill : float, optional
        Grid fill value.
    resize : int, optional
        Maximum dimension for resizing.
    padding : bool
        If ``True``, pad to a square shape.

    Returns
    -------
    Image.Image
        Rendered image.
    """
    if isinstance(image, torch.Tensor):
        img = zimage.imgify(
            image.detach().cpu(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            symmetric=symmetric,
            level=level,
            grid=grid,
            gridfill=gridfill,
        )
    elif isinstance(image, np.ndarray):
        img = zimage.imgify(
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            symmetric=symmetric,
            level=level,
            grid=grid,
            gridfill=gridfill,
        )
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if resize:
        ratio = resize / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.NEAREST)

    if padding:
        max_size = resize if resize else max(img.size)
        new_im = Image.new("RGBA", (max_size, max_size))
        new_im.putalpha(0)
        new_im.paste(img, ((max_size - img.size[0]) // 2, (max_size - img.size[1]) // 2))
        img = new_im

    return img


def plot_grid(
    ref_c: dict[int, Any],
    cmap_dim: int = 1,
    cmap: str = "bwr",
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = True,
    resize: int | None = None,
    padding: bool = True,
    figsize: tuple[float, float] = (6, 6),
):
    """Plot a grid of reference images.

    Parameters
    ----------
    ref_c : dict[int, Any]
        Dictionary of reference images keyed by concept index.
        Values can be lists of images or tuples of lists (e.g., from
        :py:func:`vis_img_heatmap`).
    cmap_dim : int
        Which tuple element to apply the colormap to (1 or 2).
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Value range.
    symmetric : bool
        Symmetric normalization.
    resize : int, optional
        Maximum dimension for resizing.
    padding : bool
        Pad images to squares.
    figsize : tuple[float, float]
        Figure size.
    """
    keys = list(ref_c.keys())
    nrows = len(keys)
    value = next(iter(ref_c.values()))

    if cmap_dim not in (1, 2):
        raise ValueError("'cmap_dim' must be 1 or 2.")

    if isinstance(value, tuple) and isinstance(value[0], Iterable):
        nsubrows = len(value)
        ncols = len(value[0])
    elif isinstance(value, list):
        nsubrows = 1
        ncols = len(value)
    else:
        raise TypeError("Values must be lists or tuples of lists.")

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(nrows, 1, wspace=0.025, hspace=0.025)

    for row_idx, key in enumerate(keys):
        inner = gridspec.GridSpecFromSubplotSpec(
            nsubrows,
            ncols,
            subplot_spec=outer[row_idx],
            wspace=0.025,
            hspace=0.025,
        )
        imgs = ref_c[key]

        for sr in range(nsubrows):
            sub_imgs = imgs[sr] if nsubrows > 1 else imgs
            for col in range(ncols):
                ax = plt.Subplot(fig, inner[sr, col])
                img = sub_imgs[col]

                if sr + 1 == cmap_dim:
                    img = imgify(
                        img, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, resize=resize, padding=padding
                    )
                else:
                    img = imgify(img, resize=resize, padding=padding)

                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

    plt.show()
