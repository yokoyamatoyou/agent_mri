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
    with tempfile.NamedTemporaryFile(suffix=".nii") as tmp1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as tmp2:
        report = client.analyze_image(tmp1.name, tmp2.name)

    assert isinstance(report, GPTReport)
    assert report.is_finding_present is False


def test_parse_report_helper():
    """Ensure ``parse_report`` converts JSON text to ``GPTReport``."""

    text = json.dumps({
        "is_finding_present": True,
        "finding_summary": "dummy",
        "detailed_description": "desc",
        "confidence_score": 0.5,
        "anatomical_location": "brain",
    })

    report = OpenAIClient.parse_report(text)
    assert isinstance(report, GPTReport)
    assert report.is_finding_present is True


def test_report_to_json():
    """GPTReport should serialize to JSON string."""

    report = GPTReport(
        is_finding_present=True,
        finding_summary="s",
        detailed_description="d",
        confidence_score=0.7,
        anatomical_location="brain",
    )
    data = json.loads(report.to_json())
    assert data["is_finding_present"] is True


def test_parse_report_missing_key():
    """parse_report should fail when required keys are missing."""
    text = json.dumps({"confidence_score": 0.1})
    with pytest.raises(ValueError):
        OpenAIClient.parse_report(text)


def test_invalid_json(monkeypatch):
    dummy = DummyResponse({"choices": [{"message": {"content": "not-json"}}]})

    def fake_create(*args, **kwargs):
        return dummy

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("openai.ChatCompletion.create", fake_create)

    client = OpenAIClient()
    with tempfile.NamedTemporaryFile(suffix=".nii") as tmp1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as tmp2:
        with pytest.raises(ValueError):
            client.analyze_image(tmp1.name, tmp2.name)


def test_api_error(monkeypatch):
    def fake_create(*args, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr("openai.ChatCompletion.create", fake_create)

    client = OpenAIClient()
    with tempfile.NamedTemporaryFile(suffix=".nii") as tmp1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as tmp2:
        with pytest.raises(RuntimeError):
            client.analyze_image(tmp1.name, tmp2.name)


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


def test_missing_api_key(monkeypatch):
    """OpenAIClient should raise when API key env var is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIClient()
