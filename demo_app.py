"""Local Streamlit interface for the existing Tesseract OCR pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from manga_pipeline import DatasetError, load_sequence, run_tesseract_sequence

ROOT = Path(__file__).resolve().parent
DEMO_ARTIFACTS = ROOT / "artifacts" / "demo"


@st.cache_data
def load_sequence_options() -> list[dict[str, str]]:
    manifest_path = ROOT / "sequences.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read sequence manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, list):
        raise RuntimeError(f"Sequence manifest must contain a list: {manifest_path}")

    options = []
    for record in manifest:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("sequence_id"), str)
            or not isinstance(record.get("split"), str)
        ):
            raise RuntimeError(f"Invalid sequence entry in manifest: {manifest_path}")
        options.append(
            {"sequence_id": record["sequence_id"], "split": record["split"]}
        )
    return options


def save_demo_result(result: dict[str, Any], sequence_id: str) -> Path:
    output_path = DEMO_ARTIFACTS / "pilot" / f"{sequence_id}_tesseract_psm6.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


st.set_page_config(page_title="Manga OCR Studio", layout="wide")
st.title("Manga OCR Studio")
st.caption("A local demo using the project’s existing Tesseract OCR pipeline.")

try:
    sequence_options = load_sequence_options()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

if not sequence_options:
    st.error("No sequences are available in sequences.json.")
    st.stop()

option_by_id = {item["sequence_id"]: item for item in sequence_options}
sequence_id = st.selectbox(
    "Sequence",
    options=list(option_by_id),
    format_func=lambda selected_id: (
        f"{selected_id} ({option_by_id[selected_id]['split']})"
    ),
)

try:
    sequence = load_sequence(ROOT, sequence_id, include_labels=False)
except (DatasetError, OSError) as error:
    st.error(f"Could not load the selected sequence: {error}")
    st.stop()

st.subheader("Pages")
page_columns = st.columns(3)
for page_number, (column, loaded_image) in enumerate(
    zip(page_columns, sequence.images, strict=True),
    start=1,
):
    with column:
        st.markdown(f"**Page {page_number}**")
        try:
            with Image.open(loaded_image.path) as image:
                st.image(image.copy(), width="stretch")
        except OSError as error:
            st.error(f"Could not preview page {page_number}: {error}")

if st.button("Run OCR", type="primary"):
    st.session_state.pop("ocr_result", None)
    st.session_state.pop("ocr_sequence_id", None)
    try:
        with st.spinner("Running local Tesseract OCR on all three pages..."):
            pages = run_tesseract_sequence(
                ROOT,
                sequence_id,
                artifact_dir=DEMO_ARTIFACTS,
                psm=6,
                min_confidence=0,
                min_lines=1,
                force=True,
            )
            result = {"sequence_id": sequence_id, "pages": pages}
            output_path = save_demo_result(result, sequence_id)
        st.session_state["ocr_result"] = result
        st.session_state["ocr_sequence_id"] = sequence_id
        st.session_state["ocr_output_path"] = str(output_path)
    except (DatasetError, OSError, ValueError) as error:
        st.error(f"OCR failed: {error}")
        st.info("Confirm that local Tesseract is installed and available on PATH.")

if (
    st.session_state.get("ocr_sequence_id") == sequence_id
    and "ocr_result" in st.session_state
):
    result = st.session_state["ocr_result"]
    pages = result["pages"]
    total_lines = sum(len(page) for page in pages)
    st.subheader(f"Recognized text ({total_lines} lines)")

    text_columns = st.columns(3)
    for page_number, (column, page) in enumerate(
        zip(text_columns, pages, strict=True),
        start=1,
    ):
        with column:
            st.markdown(f"**Page {page_number}**")
            if page:
                st.text("\n".join(line["text"] for line in page))
            else:
                st.caption("No text recognized.")

    st.subheader("Resulting JSON")
    st.json(result)
    st.caption(f"Demo output saved to `{st.session_state['ocr_output_path']}`")
