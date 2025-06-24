"""Utilities for MRI image analysis using ANTsPyNet."""

import ants
import streamlit as st
from antspynet.utilities import brain_extraction
from pathlib import Path
from typing import Tuple
import numpy as np

_ALLOWED_EXTS = {".nii", ".nii.gz", ".png", ".jpg", ".jpeg"}

def is_supported_file(path: str) -> bool:
    """Return ``True`` if ``path`` has a supported medical image extension."""
    ext = Path(path).suffix.lower()
    if ext == ".gz":
        ext = Path(path).with_suffix("").suffix.lower() + ".gz"
    return ext in _ALLOWED_EXTS


@st.cache_resource
def _get_brain_extractor():
    """Return the ANTsPyNet ``brain_extraction`` function.

    Caching avoids re-loading heavy model weights on every rerun.
    """
    return brain_extraction


def extract_brain(image_path: str) -> Tuple[ants.ANTsImage, ants.ANTsImage] | None:
    """Run brain extraction and return image and mask.

    Parameters
    ----------
    image_path : str
        Path to a NIfTI image on disk.

    Returns
    -------
    Tuple[ants.ANTsImage, ants.ANTsImage] | None
        Tuple of original image and mask or ``None`` if extraction failed.
    """

    if not is_supported_file(image_path):
        return None

    try:
        img = ants.image_read(image_path)
        brain_extractor = _get_brain_extractor()
        mask = brain_extractor(img)
    except Exception:
        return None

    return img, mask


def overlay_mask(image: ants.ANTsImage, mask: ants.ANTsImage, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Return RGB array of ``image`` with ``mask`` overlaid.

    Parameters
    ----------
    image : ants.ANTsImage
        Source MRI image.
    mask : ants.ANTsImage
        Binary mask indicating region of interest.
    color : tuple[int, int, int], optional
        RGB color for the mask overlay. Defaults to red.

    Returns
    -------
    np.ndarray
        RGB image array suitable for ``st.image``.
    """

    img_np = image.numpy().astype(float)
    img_np -= img_np.min()
    maxv = img_np.max()
    if maxv > 0:
        img_np /= maxv
    img_uint8 = (img_np * 255).astype(np.uint8)
    rgb = np.stack([img_uint8] * 3, axis=-1)

    mask_np = mask.numpy() > 0
    for i, c in enumerate(color):
        rgb[..., i][mask_np] = c

    return rgb
