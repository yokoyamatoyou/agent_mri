import os
from typing import Dict, Any
import openai
from pydantic import BaseModel

class GPTResponse(BaseModel):
    report: str

class OpenAIClient:
    def __init__(self, api_key_env: str = "OPENAI_API_KEY"):
        key = os.getenv(api_key_env)
        if not key:
            raise ValueError(f"Environment variable {api_key_env} is not set")
        openai.api_key = key

    def analyze_image(self, image_path: str) -> GPTResponse:
        try:
            with open(image_path, "rb") as f:
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {"role": "system", "content": "You are a radiology assistant."},
                        {"role": "user", "content": "Analyze this MRI image."}
                    ],
                    files=[{"file": f}]
                )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

        try:
            data = GPTResponse(**{"report": response["choices"][0]["message"]["content"]})
            return data
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e
