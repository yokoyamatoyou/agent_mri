import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import tempfile
import pytest

from mri_app.openai_client import OpenAIClient, GPTReport

class DummyResponse(dict):
    pass


def test_parse_valid_response(monkeypatch):
    resp_json = {
        "is_finding_present": False,
        "finding_summary": None,
        "detailed_description": None,
        "confidence_score": 0.9,
        "anatomical_location": None,
    }
    dummy = DummyResponse({"choices": [{"message": {"content": json.dumps(resp_json)}}]})

    def fake_create(*args, **kwargs):
        return dummy

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("openai.ChatCompletion.create", fake_create)

    client = OpenAIClient()
    with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
        report = client.analyze_image(tmp.name)

    assert isinstance(report, GPTReport)
    assert report.is_finding_present is False


def test_invalid_json(monkeypatch):
    dummy = DummyResponse({"choices": [{"message": {"content": "not-json"}}]})

    def fake_create(*args, **kwargs):
        return dummy

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("openai.ChatCompletion.create", fake_create)

    client = OpenAIClient()
    with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
        with pytest.raises(ValueError):
            client.analyze_image(tmp.name)


def test_api_error(monkeypatch):
    def fake_create(*args, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("openai.ChatCompletion.create", fake_create)

    client = OpenAIClient()
    with tempfile.NamedTemporaryFile(suffix=".nii") as tmp:
        with pytest.raises(RuntimeError):
            client.analyze_image(tmp.name)


def test_unsupported_extension(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    client = OpenAIClient()
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        with pytest.raises(ValueError):
            client.analyze_image(tmp.name)


def test_file_not_found(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    client = OpenAIClient()
    with pytest.raises(FileNotFoundError):
        client.analyze_image("/nonexistent/path.nii")
