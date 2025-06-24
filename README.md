# MRI Reading Assistant

This project is an example MVP combining **ANTsPyNet** and **GPT-4.1**.
It provides a simple Streamlit application to upload an MRI image, run brain
extraction and request a textual report from OpenAI.

On first execution ANTsPyNet downloads pretrained weights under
`~/.antspynet/`.  Ensure the machine can access the internet when running
the application for the first time.

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

After processing an image the app shows the extracted brain, textual findings
and offers a **Download JSON** button to save the structured report.

Unit tests can be executed with `pytest`.

The results shown by the application are intended for assisting radiologists
and do not constitute a medical diagnosis.
