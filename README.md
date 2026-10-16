# BYTE Society Manga Task

## About the Project

This project processes three consecutive English manga pages and extracts text in reading order. It uses Tesseract OCR and a logistic regression filter trained with weak labels to improve the selection of text lines. The final predictions are saved in the required JSONL format.

## How It Works

- Tesseract OCR extracts text from manga pages.
- Geometric right-to-left ordering arranges the detected text.
- I trained a logistic regression classifier using weak labels to filter OCR lines.
- The pipeline generates predictions in JSONL format.

## Dataset

The dataset contains 80 development sequences and 15 test sequences.
Each sequence contains three manga pages.

## Model and Results

I used a SciPy logistic regression classifier with a threshold of 0.7.
I selected this threshold through grouped cross-validation and evaluated
the pipeline using the supplied scoring script.

Original baseline balanced joint F1: 0.146216
Filtered pipeline balanced joint F1: 0.152115

These are validation results, not test-set scores.

## Limitations

The OCR sometimes recognizes sound effects and other unwanted text. Speaker identification did not work reliably in my experiments, so all retained lines use UNKNOWN as the speaker label. The filter improved balanced joint F1 slightly on validation, but text-order score and token coverage decreased.

## Running the Project

Install the required dependencies and Tesseract OCR.

Validate the dataset:

    python manga_pipeline.py validate

Run the tests:

    python -m unittest -v test_manga_pipeline

Generate predictions using the final fitted filter:

    python manga_pipeline.py generate-test \
      --psm 11 --min-confidence 70 \
      --reading-order rtl_geometry \
      --weak-filter-model PATH_TO_FINAL_MODEL \
      --output artifacts/submission.jsonl

Replace PATH_TO_FINAL_MODEL with the actual path to the
model_threshold_0.7.json file.

## References and Licences

- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
  Licence: Apache-2.0
- SciPy: https://github.com/scipy/scipy
  Licence: BSD-3-Clause
- OpenCV: https://github.com/opencv/opencv
  Licence: Apache-2.0
- Streamlit: https://github.com/streamlit/streamlit
  Licence: Apache-2.0
- Comic Text Detector, evaluated experimentally:
  https://github.com/dmMaze/comic-text-detector
  Licence: GPL-3.0
- Qwen2-VL-2B model, evaluated experimentally:
  https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- Qwen2.5-VL-3B model, evaluated experimentally:
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
  Licence: Qwen Research License

The official score.py file was supplied by BYTE Society.
The vision-language models were tested locally for speaker attribution,
but neither was selected for the final predictions.