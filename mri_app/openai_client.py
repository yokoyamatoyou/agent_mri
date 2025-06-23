"""OpenAI client for MRI report generation."""

import json
import os
from typing import Any

import openai
from pydantic import BaseModel

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

    def analyze_image(self, image_path: str) -> GPTReport:
        """Send an MRI image to GPT and parse JSON report."""

        try:
            with open(image_path, "rb") as f:
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {"role": "system", "content": "You are a radiology assistant."},
                        {"role": "user", "content": "Analyze this MRI image and reply in JSON."},
                    ],
                    files=[{"file": f}],
                )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

        try:
            content: str = response["choices"][0]["message"]["content"]
            data: Any = json.loads(content)
            return GPTReport.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e
