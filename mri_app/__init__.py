# -*- coding: utf-8 -*-
"""MRI app package initialization."""

from dotenv import load_dotenv

# Load environment variables from a .env file if present. Existing variables are
# not overwritten. This allows developers to configure OPENAI_API_KEY locally
# without modifying code.
load_dotenv()

