"""Utilities for MRI image analysis using ANTsPyNet."""

import ants
from antspynet.utilities import brain_extraction
from pathlib import Path

_ALLOWED_EXTS = {".nii", ".nii.gz", ".png", ".jpg", ".jpeg"}

def is_supported_file(path: str) -> bool:
    """Return ``True`` if file has a supported extension."""
    ext = Path(path).suffix.lower()
    if ext == ".gz":
        ext = Path(path).with_suffix("").suffix.lower() + ".gz"
    return ext in _ALLOWED_EXTS


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

    if not is_supported_file(image_path):
        return None

    try:
        img = ants.image_read(image_path)
        mask = brain_extraction(img)
    except Exception:
        return None

    return img * mask
