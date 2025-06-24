# -*- coding: utf-8 -*-
"""MRI Reading Assistant MVP using Streamlit."""

import os
import tempfile

import streamlit as st
import ants

from mri_app.image_utils import extract_brain, overlay_mask
from mri_app.openai_client import get_openai_client


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

        # 画像表示とレポートを並べて表示するカラムを用意
        img_col, text_col = st.columns(2)
        with img_col:
            st.image(overlay_mask(image, mask), caption="Extracted brain")

        client = get_openai_client()
        st.info("Sending to GPT-4.1...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as mtmp:
            ants.image_write(mask, mtmp.name)
            mask_path = mtmp.name

        response = client.analyze_image(tmp_path, mask_path)
        os.remove(mask_path)

        # OpenAIから返ってきたレポートを表示
        with text_col:
            if response.is_finding_present:
                st.markdown(f"**所見要約:** {response.finding_summary}")
                st.markdown(f"**詳細説明:** {response.detailed_description}")
                st.markdown(f"**推定部位:** {response.anatomical_location}")
                st.markdown(f"**信頼度:** {response.confidence_score:.2f}")
            else:
                st.markdown("**所見:** 異常所見は検出されませんでした。")

        with text_col:
            st.download_button(
                label="Download JSON",
                data=response.to_json(),
                file_name="report.json",
                mime="application/json",
            )

        st.markdown("*本結果はAIによる補助情報であり、診断は専門医が行ってください*")
    except Exception as e:
        st.error(f"Processing failed: {e}")
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
