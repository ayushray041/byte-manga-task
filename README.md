# BYTE Society Manga Task

## About the Project

This project is made for the BYTE Society Applied ML task.
The goal is to [explain what the project does in your own words].

## How It Works

- Tesseract is used to extract text from manga pages.
- The text is arranged using right-to-left reading order.
- I trained a logistic regression model to filter OCR text using weak labels.
- The output is generated in JSONL format.

## Dataset

The dataset contains 80 development sequences and 15 test
sequences. Each sequence has three manga pages.

## Model and Experiments

I used a SciPy logistic regression classifier with a threshold
of 0.7. I compared it with the original OCR pipeline and other
approaches using the supplied scoring script.

The final validation balanced joint F1 was 0.152115, compared
with 0.146216 for the original baseline.

## Limitations

Speaker identification was not reliable in my experiments,
so the final predictions use UNKNOWN for all retained speakers.
OCR can also recognize some unwanted text.

## Tools Used

- Python
- Tesseract OCR
- SciPy
- OpenCV
- The supplied scoring script

## Output

The final predictions are saved in submission.jsonl.
The file contains 15 test sequences in the required format.

## Optional local OCR demo

The Streamlit demo is separate from the CLI pipeline and does not change
submission files. It reads sequence IDs and page images from `sequences.json`
and runs the existing local Tesseract OCR pipeline; it does not use gold labels
or upload images.

Install Streamlit into the project environment if it is not already installed,
then start the app from the project root:

```sh
.venv/bin/python -m pip install streamlit
.venv/bin/python -m streamlit run demo_app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. Demo OCR outputs are stored separately
under `artifacts/demo/`.