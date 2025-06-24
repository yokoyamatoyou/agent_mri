import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tempfile
from mri_app import image_utils


def test_extract_brain_invalid_extension(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
        result = image_utils.extract_brain(tmp.name)
        assert result is None


def test_extract_brain_handles_error(monkeypatch):
    def fake_image_read(path):
        raise RuntimeError('read error')
    monkeypatch.setattr(image_utils.ants, 'image_read', fake_image_read)
    monkeypatch.setattr(image_utils, 'brain_extraction', lambda img: None)
    with tempfile.NamedTemporaryFile(suffix='.nii') as tmp:
        result = image_utils.extract_brain(tmp.name)
        assert result is None
