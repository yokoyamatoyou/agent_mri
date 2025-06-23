# MRI Reading Assistant

This project is an example MVP combining **ANTsPyNet** and **GPT-4.1**.  
It provides a simple Streamlit application to upload an MRI image, run brain
extraction and request a textual report from OpenAI.

## Setup

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare environment variables.  Copy `.env.example` and set your
   `OPENAI_API_KEY`.

3. Launch the application
   ```bash
   streamlit run app.py
   ```

Unit tests can be executed with `pytest`.
