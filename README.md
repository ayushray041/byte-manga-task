# BYTE Society Manga Task

## About

This project processes three consecutive English manga pages and extracts text in reading order. It uses Tesseract OCR and a small logistic-regression filter trained with weak labels. Predictions are written in the JSONL format required by the task.

## Dataset

- 80 labelled development sequences
- 15 test sequences
- 3 pages per sequence (285 image files in total)

The dataset and the official `score.py` were supplied by BYTE Society and are not changed by this project.

## How it works

1. Read the page paths and order from `sequences.json`.
2. Run local Tesseract OCR (PSM 11, confidence threshold 70).
3. Arrange recognized lines using geometric right-to-left ordering.
4. Use an L2-regularized logistic-regression filter to keep likely story-text lines. It uses OCR text and page-geometry features, with a final threshold of 0.7.
5. Write one JSONL object for each test sequence, with three page lists.

For the final fit, weak alignment produced 2,641 positive candidates, 403 negative candidates, and 1,852 ambiguous candidates. Ambiguous candidates were excluded, leaving 3,044 examples for fitting. The labels are weak/noisy labels, not manually verified annotations.

## Validation and experiments

I used a sequence-level split with seed 42: 64 training sequences and 16 validation sequences. Four-fold grouped cross-validation on the 64 training sequences selected the 0.7 filter threshold. The 16 validation sequences were kept separate during threshold selection.

| Configuration | Text-order score | Balanced joint F1 | Joint F1 | Matched-token coverage |
|---|---:|---:|---:|---:|
| Tesseract + geometric RTL baseline | 0.343461 | 0.146216 | 0.296254 | 0.352127 |
| Weak-label filter, threshold 0.7 | 0.333069 | 0.152115 | 0.300306 | 0.337458 |

The filter gives a small increase in balanced joint F1 on the fixed validation set, while text-order score and matched-token coverage decrease. This is a mixed result, not proof of generally better transcription.

I also tested a comic-text detector and local Qwen vision-language models for speaker attribution. The detector had lower balanced joint F1 on the fixed validation set. The Qwen speaker models also performed worse than the all-`UNKNOWN` speaker baseline, so they were not selected.

## Final output and limitations

The final predictions are in `submission.jsonl` in this package. The file has 15 sequences, 45 page lists and 778 OCR entries. Five page lists are empty. All retained entries use `UNKNOWN` for the speaker because the tested speaker-attribution models did not improve validation results.

OCR can still include sound effects or other unwanted text, and empty-page detection is imperfect. Passing schema validation confirms the output format, not transcription accuracy.

## Setup and commands

The project was tested on macOS Apple Silicon with Python 3.13.16 and Tesseract 5.5.3. Install Python dependencies and Tesseract first. On macOS, Tesseract can be installed with `brew install tesseract`.

Create and activate a virtual environment, install the dependencies, then restore the supplied dataset at the project root with the original layout (`sequences.json`, `sample_submission.jsonl`, `development/labels.jsonl`, `development/images/`, and `test/images/`). The dataset is not included in this repository.

```sh
python -m pip install -r requirements.txt
python manga_pipeline.py validate
python -m unittest -v test_manga_pipeline
```

To generate predictions with the packaged threshold-0.7 filter, use the actual model path and a new output path:

```sh
python manga_pipeline.py generate-test \
  --psm 11 --min-confidence 70 --reading-order rtl_geometry \
  --weak-filter-model models/weak_text_filter_final_all80_20261014/model_threshold_0.7.json \
  --output artifacts/submission.jsonl
python manga_pipeline.py validate-submission artifacts/submission.jsonl
```

### Optional local Streamlit demo

The demo is a local browser interface; it is separate from the submission pipeline. With the dataset restored and dependencies installed, run:

```sh
python -m streamlit run demo_app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. Demo outputs are stored under `artifacts/demo/` and do not replace submission files.

## External code, tools and model references

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — OCR engine; Apache-2.0.
- [SciPy](https://github.com/scipy/scipy) — used to fit the logistic-regression filter; BSD-3-Clause.
- [OpenCV](https://github.com/opencv/opencv) — image processing and the experimental detector path; OpenCV 4.5+ is Apache-2.0.
- [Streamlit](https://github.com/streamlit/streamlit) — optional local demo interface; Apache-2.0.
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) — local runtime used in VLM experiments; MIT.
- [Comic Text Detector](https://github.com/dmMaze/comic-text-detector) — experimental manga text detector; GPL-3.0. Its tested ONNX asset was obtained from the [manga-image-translator release](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1); the asset did not state a separate licence, so check its upstream terms before reuse.
- [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) — tested locally for speaker attribution; the model card lists Apache-2.0.
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — tested locally for speaker attribution; the upstream Qwen Research License limits use to non-commercial research/evaluation unless separate permission is obtained. See its [licence](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE).
- `score.py` and the task dataset/template were supplied by BYTE Society.

The Qwen models and comic-text detector were experiments only and are not selected for final inference. The final predictions use local Tesseract OCR and the trained weak-label filter.