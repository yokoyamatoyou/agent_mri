import ants
from antspynet.utilities import brain_extraction


def extract_brain(image_path: str) -> ants.ANTsImage:
    """Run brain extraction and return masked image."""
    img = ants.image_read(image_path)
    mask = brain_extraction(img)
    return img * mask
