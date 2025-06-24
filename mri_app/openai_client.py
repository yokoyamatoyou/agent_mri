"""OpenAI client for MRI report generation."""

import json
import os
from typing import Any

import openai
from pydantic import BaseModel

from .image_utils import is_supported_file

class GPTReport(BaseModel):
    """Structured response from GPT."""

    is_finding_present: bool
    finding_summary: str | None = None
    detailed_description: str | None = None
    confidence_score: float
    anatomical_location: str | None = None

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
            data: Any = json.loads(content)
            return GPTReport.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e
