# -*- coding: utf-8 -*-
"""Utilities for MRI image analysis using ANTsPyNet."""

import streamlit as st

try:  # heavy deps may be missing in the test environment
    import ants
except Exception:  # pragma: no cover - optional dependency
    class _DummyAnts:
        pass

    ants = _DummyAnts()

try:  # antspynet is optional during unit tests
    from antspynet.utilities import brain_extraction
except Exception:  # pragma: no cover - optional dependency
    brain_extraction = None
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

_ALLOWED_EXTS = {".nii", ".nii.gz", ".png", ".jpg", ".jpeg"}

def is_supported_file(path: str) -> bool:
    """Return ``True`` if ``path`` has a supported medical image extension.

    In addition to the file extension, basic header checks are performed for
    common image formats (PNG/JPEG).  This guards against users renaming an
    unsupported file with a valid extension.
    """

    ext = Path(path).suffix.lower()
    if ext == ".gz":
        ext = Path(path).with_suffix("").suffix.lower() + ".gz"

    if ext not in _ALLOWED_EXTS:
        return False

    if ext in {".png", ".jpg", ".jpeg"}:
        # Verify the image header using Pillow. ``Image.open`` will raise an
        # exception if the file does not contain a valid image of the expected
        # type. ``verify`` avoids loading the full image into memory.
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            return False

    return True


@st.cache_resource
def _get_brain_extractor():
    """Return the ANTsPyNet ``brain_extraction`` function.

    Caching avoids re-loading heavy model weights on every rerun.
    """
    if brain_extraction is None:
        raise ImportError("antspynet is not available")
    return brain_extraction


def extract_brain(image_path: str) -> Tuple["ants.ANTsImage", "ants.ANTsImage"] | None:
    """Run brain extraction and return image and mask.

    Parameters
    ----------
    image_path : str
        Path to a NIfTI image on disk.

    Returns
    -------
    Tuple["ants.ANTsImage", "ants.ANTsImage"] | None
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


def overlay_mask(image: "ants.ANTsImage", mask: "ants.ANTsImage", color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
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
