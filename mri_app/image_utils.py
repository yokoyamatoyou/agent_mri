"""Utilities for MRI image analysis using ANTsPyNet."""

import ants
from antspynet.utilities import brain_extraction


def extract_brain(image_path: str) -> ants.ANTsImage | None:
    """Run brain extraction and return masked image.

    Parameters
    ----------
    image_path : str
        Path to a NIfTI image on disk.

    Returns
    -------
    ants.ANTsImage | None
        Masked image or ``None`` if extraction failed.
    """

    try:
        img = ants.image_read(image_path)
        mask = brain_extraction(img)
    except Exception:
        return None

    return img * mask
