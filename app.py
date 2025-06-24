"""MRI Reading Assistant MVP using Streamlit."""
import os
import tempfile

import streamlit as st
import ants

from mri_app.image_utils import extract_brain, overlay_mask
from mri_app.openai_client import OpenAIClient


def main():
    st.title("MRI Reading Assistant")

    uploaded = st.file_uploader("Upload NIfTI image", type=["nii", "nii.gz"])
    if uploaded is None:
        st.info("Please upload an MRI file.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        st.info("Performing brain extraction...")
        result = extract_brain(tmp_path)
        if result is None:
            st.error("Brain extraction failed.")
            return
        image, mask = result
        st.image(overlay_mask(image, mask), caption="Extracted brain")

        client = OpenAIClient()
        st.info("Sending to GPT-4.1...")
        response = client.analyze_image(tmp_path)
        st.markdown(response.report)
    except Exception as e:
        st.error(f"Processing failed: {e}")
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
