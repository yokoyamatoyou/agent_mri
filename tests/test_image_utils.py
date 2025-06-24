# -*- coding: utf-8 -*-
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tempfile
from mri_app import image_utils
import numpy as np
from PIL import Image
import pytest


class DummyImage:
    def __init__(self, arr):
        self._arr = np.array(arr)

    def numpy(self):
        return self._arr

    def __mul__(self, other):
        return DummyImage(self._arr * other._arr)


def test_is_supported_file_header_check(tmp_path):
    """Files renamed to an allowed extension should be detected."""
    fake = tmp_path / "bad.jpg"
    fake.write_bytes(b"not an image")
    assert image_utils.is_supported_file(str(fake)) is False

    good = tmp_path / "good.png"
    Image.new("RGB", (1, 1)).save(good)
    assert image_utils.is_supported_file(str(good)) is True


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


def test_extract_brain_success(monkeypatch):
    img = DummyImage(np.ones((2, 2)))
    mask = DummyImage([[1, 0], [0, 1]])

    monkeypatch.setattr(image_utils.ants, "image_read", lambda p: img)
    monkeypatch.setattr(image_utils, "brain_extraction", lambda x: mask)

    with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
        res = image_utils.extract_brain(tmp.name)
        assert isinstance(res, tuple)
        assert res[0] is img
        assert res[1] is mask


def test_overlay_mask():
    img = DummyImage(np.zeros((2, 2)))
    mask = DummyImage([[0, 1], [0, 0]])

    arr = image_utils.overlay_mask(img, mask)
    assert arr.shape == (2, 2, 3)
    assert arr[0, 1, 0] == 255
    assert arr[0, 1, 1] == 0
    assert arr[0, 1, 2] == 0


def test_overlay_mask_custom_color():
    """overlay_mask should honor the given color tuple."""
    img = DummyImage(np.zeros((1, 1)))
    mask = DummyImage([[1]])

    arr = image_utils.overlay_mask(img, mask, color=(0, 255, 0))
    assert (arr == [0, 255, 0]).all()


def test_overlay_mask_shape_mismatch():
    """overlay_mask should error when shapes differ."""
    img = DummyImage(np.zeros((2, 2)))
    mask = DummyImage(np.zeros((1, 1)))

    with pytest.raises(ValueError):
        image_utils.overlay_mask(img, mask)
