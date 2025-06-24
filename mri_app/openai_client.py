"""OpenAI client for MRI report generation."""

import json
import os
from typing import Any

import openai
import streamlit as st
from pydantic import BaseModel

from .image_utils import is_supported_file

class GPTReport(BaseModel):
    """Structured response from GPT."""

    is_finding_present: bool
    finding_summary: str | None = None
    detailed_description: str | None = None
    confidence_score: float
    anatomical_location: str | None = None

    def to_json(self) -> str:
        """Return the report as a JSON string."""
        return self.model_dump_json(ensure_ascii=False)

class OpenAIClient:
    def __init__(self, api_key_env: str = "OPENAI_API_KEY"):
        key = os.getenv(api_key_env)
        if not key:
            raise ValueError(f"Environment variable {api_key_env} is not set")
        openai.api_key = key


    def analyze_image(self, image_path: str, mask_path: str | None = None) -> GPTReport:
        """Send an MRI image (and optional mask) to GPT and parse JSON report."""

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        if not is_supported_file(image_path):
            raise ValueError(f"Unsupported file type: {image_path}")

        files = []
        try:
            img_f = open(image_path, "rb")
            files.append({"file": img_f})
            if mask_path is not None:
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(mask_path)
                if not is_supported_file(mask_path):
                    raise ValueError(f"Unsupported file type: {mask_path}")
                mask_f = open(mask_path, "rb")
                files.append({"file": mask_f})

            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": "You are a radiology assistant."},
                    {"role": "user", "content": "Analyze these MRI images and reply in JSON."},
                ],
                files=files,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e
        finally:
            for f in files:
                f["file"].close()

        try:
            content: str = response["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Malformed OpenAI response: {e}") from e

        return self.parse_report(content)

    @staticmethod
    def parse_report(text: str) -> GPTReport:
        """Parse JSON string ``text`` and return ``GPTReport``."""

        try:
            data: Any = json.loads(text)
            report = GPTReport.model_validate(data)
            if report.is_finding_present is None or report.confidence_score is None:
                raise ValueError("Missing required fields")
            return report
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e


@st.cache_resource
def get_openai_client() -> OpenAIClient:
    """Return cached ``OpenAIClient`` instance."""
    return OpenAIClient()
