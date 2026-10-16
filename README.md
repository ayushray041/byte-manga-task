# BYTE Society Manga Task — Submission Package

This package contains the final generated prediction file, the local OCR/filter
pipeline and its tests, the fitted threshold-0.7 filter, the supplied scorer
(unchanged), and compact experiment evidence. It deliberately excludes the
supplied dataset, virtual environment, Hugging Face cache, and large detector or
vision-language model weights. The 15-record `submission.jsonl` is copied from
the verified project output; it is not a claim that the transcriptions are
correct.

## Dataset layout and output schema

The original dataset is not included. To run inference or the complete dataset
validator, restore the supplied files at this package root without changing
their content or layout:

```text
sequences.json
sample_submission.jsonl
development/labels.jsonl
development/images/...
test/images/...
```

Each manifest sequence has a unique `sequence_id`, a split, and three image
paths in manifest order. Each development-label and prediction record contains
`sequence_id` and `pages`, where `pages` is exactly three lists; every line
entry has string `speaker` and `text` values. The test template supplies the
required test IDs and order. Inference reads the test manifest/images and
template IDs, not development labels. Empty page lists remain empty when no
line survives filtering.

## Local setup and reproducible commands

The verified environment was macOS Apple Silicon, Python 3.13.16, and Tesseract
5.5.3. Install the local OCR executable and Python dependencies:

```sh
brew install tesseract
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

With the supplied dataset restored at the package root, validate it, run the
tests and compile the Python files:

```sh
.venv/bin/python manga_pipeline.py validate
.venv/bin/python -m unittest -v test_manga_pipeline
.venv/bin/python -m compileall -q manga_pipeline.py weak_text_filter.py test_manga_pipeline.py score.py
```

Regenerate a separate output (leaves the packaged `submission.jsonl` intact)
with the selected local Tesseract/filter configuration:

```sh
.venv/bin/python manga_pipeline.py generate-test \
  --psm 11 --min-confidence 70 --reading-order rtl_geometry \
  --weak-filter-model models/weak_text_filter_final_all80_20261014/model_threshold_0.7.json \
  --output submission_regenerated.jsonl
.venv/bin/python manga_pipeline.py validate-submission \
  submission_regenerated.jsonl
```

The supplied scorer and manifest/template must remain alongside the code. To
repeat the frozen sequence-level four-fold OOF threshold calibration after
restoring the dataset, run:

```sh
.venv/bin/python manga_pipeline.py calibrate-weak-filter \
  --seed 42 --validation-fraction 0.2 --folds 4
```

The experiment evidence in `experiments/` records the existing run. Calibration
uses the existing project procedure and writes experiment artifacts below
`artifacts/`; it does not tune against the 16 fixed-validation references.

To refit the locked 0.7 model on all 80 labeled development sequences (after
restoring the full development data), use a new output directory:

```sh
.venv/bin/python manga_pipeline.py fit-final-filter \
  --output-dir models/weak_text_filter_retrained
```

This final fit uses development labels as weak training targets only. The
threshold remains fixed at 0.7; no test images or test targets participate in
fitting.

## Method and split

The frozen seed-42 split groups by sequence: 64 development sequences for
training and four-fold grouped OOF calibration, and 16 held out for final
validation. All three pages of a sequence stay together. The persisted split
is `experiments/fixed_validation_split.json`.

The OCR baseline is local Tesseract PSM 11 with confidence >=70 and geometric
right-to-left line ordering. The optional lightweight filter is L2-regularized
logistic regression (SciPy L-BFGS-B) over OCR-time text/confidence/box features.
Training targets are conservative weak labels from fuzzy monotonic token
alignment: strong matches are positive, symbol-only candidates without matched
content are negative, and ambiguous candidates are excluded. The grouped OOF
threshold grid was 0.3, 0.4, 0.5, 0.6, and 0.7; 0.7 was selected by OOF
balanced joint F1, without using fixed-validation labels for threshold choice.
The final model was then refit at the fixed threshold on all 80 labeled
development sequences: 2,641 positive, 403 negative, and 1,852 ambiguous
candidates (3,044 usable examples).

### Actual official validation results

The supplied `score.py` produced the following metrics on the fixed 16-sequence
validation split. The final predictions use the filtered configuration. The
raw Tesseract/RTL row is retained as the baseline; this is a mixed, modest
trade-off and not evidence of generally superior OCR.

| Configuration | Text-order score | Balanced joint F1 | Joint F1 | Matched-token coverage |
| --- | ---: | ---: | ---: | ---: |
| Raw Tesseract + geometric RTL baseline | 0.343461 | 0.146216 | 0.296254 | 0.352127 |
| Threshold-0.7 filter + same OCR/RTL | 0.333069 | 0.152115 | 0.300306 | 0.337458 |

Grouped OOF metrics on the 64 training sequences:

| Configuration | Text-order score | Balanced joint F1 | Joint F1 | Matched-token coverage |
| --- | ---: | ---: | ---: | ---: |
| All-`UNKNOWN` OCR baseline | 0.396542 | 0.140622 | 0.294711 | 0.427710 |
| Filter threshold 0.3 | 0.402259 | 0.145691 | 0.302086 | 0.427297 |
| Filter threshold 0.4 | 0.402280 | 0.146048 | 0.302704 | 0.427092 |
| Filter threshold 0.5 | 0.402747 | 0.147340 | 0.304603 | 0.426387 |
| Filter threshold 0.6 | 0.402260 | 0.149700 | 0.307195 | 0.423371 |
| Filter threshold 0.7 (selected) | 0.399218 | 0.152494 | 0.309252 | 0.412193 |

At threshold 0.7, balanced joint F1 and joint F1 improved in each of the four
OOF folds, while matched-token coverage fell in every fold. The 16-sequence
validation has only one gold-empty page; neither baseline nor filtered OCR
predicted it empty.

## Speaker decision and rejected experiments

Every retained final line is assigned `UNKNOWN`. The tested local
`mlx-community/Qwen2.5-VL-3B-Instruct-4bit` speaker-only model was rejected on
the frozen validation set: with exactly the same OCR text/order it obtained
balanced joint F1 0.142920 and speaker accuracy on matched tokens 0.693793,
below all-`UNKNOWN` at 0.152115 and 0.712458. It assigned 753 lines `UNKNOWN`,
21 `NARRATION`, and 9 `character1`; no sequence had multiple character
identities. The earlier Qwen2-VL-2B result was also worse (balanced joint F1
0.085472 vs. 0.146216 for its raw-OCR all-`UNKNOWN` control). No speaker-model
weights are included.

The comic-text-detector plus rectified Tesseract experiment increased
text-order score to 0.398062 and matched-token coverage to 0.381991, but reduced
balanced joint F1 to 0.099959 and joint F1 to 0.224178 versus the raw Tesseract
baseline. It is rejected; its ONNX weights are not included.

The filter uses weak/noisy transcript alignment rather than human-verified
text-region annotations. It may retain sound effects that resemble transcript
text and can remove genuine text. Validation is a small fixed set. The final
test file passed schema/ID-order validation; this does not verify text
transcription or speaker accuracy.

## Tools, licenses, and provenance

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Apache-2.0.
- [Pillow](https://pypi.org/project/pillow/) — HPND; [NumPy](https://pypi.org/project/numpy/)
  and [SciPy](https://pypi.org/project/scipy/) — BSD licenses.
  [OpenCV-Python](https://pypi.org/project/opencv-python/) is included because
  the complete tests exercise the optional detector decoder; OpenCV is not
  needed for selected Tesseract/filter inference; OpenCV is Apache-2.0.
  The `requirements.txt` lists direct dependencies without unverified version
  pins.
- The official task `score.py` is copied unchanged from the supplied workspace
  and included because the pipeline imports it; it is not a third-party
  inference dependency.
- [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector)
  is GPL-3.0. Its referenced
  [manga-image-translator beta-0.2.1 release](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1)
  hosts the tested ONNX asset, which has no separate license statement; the
  experiment treated it as GPL-3.0. Neither code nor weights are required by
  the selected pipeline, and the weights are not packaged.
- The rejected Qwen2.5-VL model is
  [mlx-community/Qwen2.5-VL-3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-VL-3B-Instruct-4bit),
  a conversion of [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
  The upstream [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)
  restricts use to non-commercial research/evaluation absent separate
  permission. No Qwen weights or model runtime are included or needed.

## Package contents

- `submission.jsonl` — verified copy of the final 15-sequence project output.
- `manga_pipeline.py`, `weak_text_filter.py`, `test_manga_pipeline.py`, and
  unchanged `score.py`.
- `requirements.txt` — direct Python dependencies for the selected pipeline
  and tests.
- `models/weak_text_filter_final_all80_20261014/` — final threshold-0.7 JSON
  model and its fit report.
- `experiments/` — frozen split, four-fold OOF assignments/metrics, final
  baseline and filtered official scorer reports, comparison CSVs, and concise
  Qwen/detector experiment reports.
- No dataset, image files, virtual environment, local caches, or large model
  weights.

## Optional local Streamlit demo

The project includes a local browser interface for running OCR on manga sequences.

The supplied dataset is not included in this repository. Restore the official dataset files at the project root before running the demo.

Install the Python dependencies and ensure Tesseract OCR is installed on your computer. Then run:

```sh
python -m pip install -r requirements.txt
python -m streamlit run demo_app.py --server.address 127.0.0.1

## Optional local Streamlit demo

The project includes a local browser interface for running OCR on manga sequences.

The supplied dataset is not included in this repository. Restore the official dataset files at the project root before running the demo.

Install the Python dependencies and ensure Tesseract OCR is installed on your computer. Then run:

python -m pip install -r requirements.txt
python -m streamlit run demo_app.py --server.address 127.0.0.1

Open http://127.0.0.1:8501 in your browser. The demo runs locally and saves its outputs separately under artifacts/demo/; it does not replace the final submission.
