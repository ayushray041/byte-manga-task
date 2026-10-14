"""Train and evaluate a conservative OCR-line filter from training-only weak labels."""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize

from manga_pipeline import (
    DatasetError,
    OCRLine,
    _atomic_write,
    _official_score,
    _read_json,
    _read_jsonl,
    _write_jsonl,
    _load_fixed_split,
    _manifest,
    _run_tesseract_ocr_lines,
    _empty_page_metrics,
    load_sequence,
)
from score import TOKEN, align_tokens, norm

FEATURE_NAMES = (
    "confidence",
    "log_character_count",
    "log_token_count",
    "alphanumeric_ratio",
    "letter_ratio",
    "digit_ratio",
    "punctuation_ratio",
    "uppercase_ratio",
    "log_width_ratio",
    "log_height_ratio",
    "log_area_ratio",
    "log_aspect_ratio",
    "center_x",
    "center_y",
    "vertical_orientation",
)
EXPERIMENT_NAME = "weak_text_filter_seed42_v2"
CONFIDENCE_THRESHOLD = 0.5
CV_EXPERIMENT_NAME = "weak_text_filter_groupcv_seed42"
THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
FINAL_FILTER_THRESHOLD = 0.7


def line_features(text: str, confidence: float, box: tuple[int, int, int, int],
                  page_size: tuple[int, int]) -> dict[str, float]:
    """Extract only values available from an OCR line and its page image."""
    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        raise DatasetError("page dimensions must be positive")
    left, top, width, height = box
    if width < 0 or height < 0:
        raise DatasetError("OCR bounding-box dimensions cannot be negative")
    normalized = norm(text)
    tokens = TOKEN.findall(normalized)
    visible_characters = [character for character in text if not character.isspace()]
    char_count = max(1, len(visible_characters))
    letters = sum(character.isalpha() for character in visible_characters)
    digits = sum(character.isdigit() for character in visible_characters)
    alphanumeric = letters + digits
    punctuation = sum(
        not character.isalnum() and not character.isspace()
        for character in visible_characters
    )
    width_ratio = width / page_width
    height_ratio = height / page_height
    area_ratio = width_ratio * height_ratio
    aspect_ratio = width / max(height, 1)
    return {
        "confidence": min(100.0, max(0.0, confidence)) / 100.0,
        "log_character_count": math.log1p(len(normalized)),
        "log_token_count": math.log1p(len(tokens)),
        "alphanumeric_ratio": alphanumeric / char_count,
        "letter_ratio": letters / char_count,
        "digit_ratio": digits / char_count,
        "punctuation_ratio": punctuation / char_count,
        "uppercase_ratio": (
            sum(character.isupper() for character in visible_characters if character.isalpha())
            / max(letters, 1)
        ),
        "log_width_ratio": math.log1p(width_ratio),
        "log_height_ratio": math.log1p(height_ratio),
        "log_area_ratio": math.log1p(area_ratio),
        "log_aspect_ratio": math.log1p(aspect_ratio),
        "center_x": (left + width / 2) / page_width,
        "center_y": (top + height / 2) / page_height,
        "vertical_orientation": max(0.0, (height - width) / max(height + width, 1)),
    }


def weak_candidate_label(text: str, matched_content_coverage: float) -> str:
    """Use strong text alignment as positive evidence and only obvious symbol runs as negatives."""
    if matched_content_coverage >= 0.6:
        return "positive"
    if matched_content_coverage <= 0.05 and not any(character.isalnum() for character in text):
        return "negative"
    return "ambiguous"


def aligned_candidate_coverages(gold_page: list[dict[str, str]], candidates: list[Any]) -> list[float]:
    """Project the scorer's fuzzy monotonic token alignment back onto OCR lines."""
    gold_tokens = [
        (token, None)
        for item in gold_page
        for token in TOKEN.findall(norm(item["text"]))
    ]
    predicted_tokens: list[tuple[str, int]] = []
    content_counts = [0] * len(candidates)
    for line_index, candidate in enumerate(candidates):
        tokens = TOKEN.findall(norm(candidate.text))
        content_counts[line_index] = sum(any(char.isalnum() for char in token) for token in tokens)
        predicted_tokens.extend((token, line_index) for token in tokens)
    matched_counts = [0.0] * len(candidates)
    for gold_pair, predicted_pair, weight in align_tokens(gold_tokens, predicted_tokens):
        gold_token, _ = gold_pair
        token, line_index = predicted_pair
        if any(character.isalnum() for character in token) and any(
            character.isalnum() for character in gold_token
        ):
            matched_counts[line_index] += weight
    return [
        matched / max(content_count, 1)
        for matched, content_count in zip(matched_counts, content_counts, strict=True)
    ]


def _candidate_cache_path(artifact_dir: Path, sequence_id: str, page_index: int) -> Path:
    import hashlib

    sequence_key = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()
    return (
        artifact_dir
        / "intermediate"
        / sequence_key
        / "weak_filter_tesseract_psm11_conf70_raw_rtl_geometry"
        / f"page_{page_index + 1:02d}.json"
    )


def _load_candidate_pages(root: Path, sequence_id: str, artifact_dir: Path) -> list[dict[str, Any]]:
    sequence = load_sequence(root, sequence_id, include_labels=False)
    pages: list[dict[str, Any]] = []
    for page_index, image in enumerate(sequence.images):
        cache_path = _candidate_cache_path(artifact_dir, sequence_id, page_index)
        if cache_path.is_file():
            cached = _read_json(cache_path)
            if (
                not isinstance(cached, dict)
                or not isinstance(cached.get("size"), list)
                or len(cached["size"]) != 2
                or not isinstance(cached.get("lines"), list)
            ):
                raise DatasetError(f"invalid resumable weak-filter OCR cache: {cache_path}")
            size = tuple(cached["size"])
            rows = cached["lines"]
        else:
            from PIL import Image

            with Image.open(image.path) as source:
                size = source.size
            lines = _run_tesseract_ocr_lines(
                image.path,
                psm=11,
                min_confidence=70,
                reading_order="rtl_geometry",
            )
            rows = [
                {"text": line.text, "confidence": line.confidence, "box": list(line.box)}
                for line in lines
            ]
            _atomic_write(
                cache_path,
                json.dumps({"size": list(size), "lines": rows}, ensure_ascii=False, indent=2)
                + "\n",
            )
        parsed_lines: list[OCRLine] = []
        for line_number, row in enumerate(rows, 1):
            if (
                not isinstance(row, dict)
                or set(row) != {"text", "confidence", "box"}
                or not isinstance(row.get("text"), str)
                or not isinstance(row.get("box"), list)
                or len(row["box"]) != 4
            ):
                raise DatasetError(
                    f"invalid cached OCR line {line_number} for {sequence_id}, "
                    f"page {page_index + 1}"
                )
            raw_box = row["box"]
            box: tuple[int, int, int, int] = (
                int(raw_box[0]),
                int(raw_box[1]),
                int(raw_box[2]),
                int(raw_box[3]),
            )
            parsed_lines.append(
                OCRLine(
                    text=row["text"],
                    confidence=float(row["confidence"]),
                    box=box,
                )
            )
        pages.append(
            {
                "size": (int(size[0]), int(size[1])),
                "lines": parsed_lines,
            }
        )
    return pages


def _training_labels(root: Path, training_ids: list[str]) -> dict[str, list[list[dict[str, str]]]]:
    """Read only records selected for the training portion of the frozen split."""
    wanted = set(training_ids)
    labels: dict[str, list[list[dict[str, str]]]] = {}
    path = root / "development" / "labels.jsonl"
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"{path}:{line_number}: {error}") from error
            if isinstance(row, dict) and row.get("sequence_id") in wanted:
                sequence_id = row["sequence_id"]
                if sequence_id in labels or not isinstance(row.get("pages"), list) or len(row["pages"]) != 3:
                    raise DatasetError(f"{path}:{line_number}: invalid training label record")
                labels[sequence_id] = row["pages"]
    missing = wanted - labels.keys()
    if missing:
        raise DatasetError(f"training labels missing IDs: {', '.join(sorted(missing))}")
    return labels


def build_training_examples(
    root: Path, training_ids: list[str], labels: dict[str, list[list[dict[str, str]]]],
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    counts = {"positive": 0, "negative": 0, "ambiguous": 0}
    for sequence_id in training_ids:
        pages = _load_candidate_pages(root, sequence_id, artifact_dir)
        for page_index, page in enumerate(pages):
            lines = page["lines"]
            coverage = aligned_candidate_coverages(labels[sequence_id][page_index], lines)
            for line_index, (line, matched_coverage) in enumerate(
                zip(lines, coverage, strict=True)
            ):
                label = weak_candidate_label(line.text, matched_coverage)
                counts[label] += 1
                examples.append(
                    {
                        "sequence_id": sequence_id,
                        "page_index": page_index,
                        "line_index": line_index,
                        "label": label,
                        "matched_content_coverage": matched_coverage,
                        "features": line_features(
                            line.text, line.confidence, line.box, page["size"]
                        ),
                    }
                )
    return examples, counts


def fit_logistic_filter(examples: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [item for item in examples if item["label"] in {"positive", "negative"}]
    labels = np.asarray([item["label"] == "positive" for item in usable], dtype=float)
    class_counts = {label: int(sum(item["label"] == label for item in usable))
                    for label in ("positive", "negative")}
    if min(class_counts.values()) < 2:
        raise DatasetError(
            "weak labels are too sparse to fit a classifier: "
            f"positives={class_counts['positive']}, negatives={class_counts['negative']}"
        )
    matrix = np.asarray(
        [[item["features"][name] for name in FEATURE_NAMES] for item in usable],
        dtype=float,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    normalized = (matrix - means) / scales
    sample_weights = np.where(
        labels > 0,
        len(labels) / (2 * class_counts["positive"]),
        len(labels) / (2 * class_counts["negative"]),
    )

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients, intercept = parameters[:-1], parameters[-1]
        logits = normalized @ coefficients + intercept
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        loss = np.mean(
            sample_weights * (np.logaddexp(0, logits) - labels * logits)
        ) + 0.5 * np.dot(coefficients, coefficients)
        residual = sample_weights * (probabilities - labels) / len(labels)
        gradient = np.r_[normalized.T @ residual + coefficients, residual.sum()]
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(matrix.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success and result.status != 1:
        raise DatasetError(f"logistic fit failed: {result.message}")
    return {
        "model": "L2-regularized logistic regression (scipy L-BFGS-B)",
        "scipy_version": scipy.__version__,
        "optimizer": "L-BFGS-B",
        "max_iterations": 1000,
        "l2_strength": 1.0,
        "class_weight": "balanced",
        "feature_names": list(FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "threshold": CONFIDENCE_THRESHOLD,
        "class_counts": class_counts,
        "training_examples": len(usable),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def development_sequence_ids(root: Path) -> list[str]:
    """Return every labeled development sequence in manifest order, excluding test IDs."""
    records = _manifest(root)
    sequence_ids = [
        record["sequence_id"]
        for record in records
        if record["split"] == "development"
    ]
    if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
        raise DatasetError("development manifest IDs must be nonempty and unique")
    return sequence_ids


def fit_final_development_filter(
    root: Path, *, artifact_dir: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Fit the locked OOF-selected threshold on all labeled development sequences."""
    root = root.resolve()
    experiment_dir = (
        output_dir.resolve()
        if output_dir is not None
        else artifact_dir / "experiments" / "weak_text_filter_final_all80"
    )
    if output_dir is not None and experiment_dir.exists():
        raise DatasetError(
            f"refusing to overwrite existing final filter artifact directory: "
            f"{experiment_dir}"
        )
    training_ids = development_sequence_ids(root)
    labels = _training_labels(root, training_ids)
    if set(labels) != set(training_ids):
        raise DatasetError("final filter labels do not match development sequence IDs")
    examples, label_counts = build_training_examples(
        root, training_ids, labels, artifact_dir
    )
    model = fit_logistic_filter(examples)
    model.update(
        {
            "threshold": FINAL_FILTER_THRESHOLD,
            "training_scope": "all_labeled_development_sequences",
            "training_sequence_count": len(training_ids),
            "training_sequence_ids": training_ids,
            "weak_label_counts": label_counts,
            "selected_threshold_source": (
                "four-fold grouped OOF on the frozen seed-42 training split"
            ),
        }
    )

    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_path = experiment_dir / "model_threshold_0.7.json"
    examples_path = experiment_dir / "training_examples.jsonl"
    report_path = experiment_dir / "fit_report.json"
    _atomic_write(model_path, json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    _write_jsonl(examples_path, examples)
    report = {
        "model_path": str(model_path),
        "training_scope": model["training_scope"],
        "training_sequence_count": len(training_ids),
        "training_sequence_ids": training_ids,
        "weak_label_counts": label_counts,
        "usable_training_examples": model["training_examples"],
        "class_counts": model["class_counts"],
        "threshold": FINAL_FILTER_THRESHOLD,
        "threshold_source": model["selected_threshold_source"],
        "training_examples_path": str(examples_path),
    }
    _atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Final all-development filter model: {model_path}")
    print(
        f"Training sequences={len(training_ids)}; "
        f"weak_labels={label_counts}; threshold={FINAL_FILTER_THRESHOLD}"
    )
    return report


def predict_probability(model: dict[str, Any], features: dict[str, float]) -> float:
    values = np.asarray([features[name] for name in model["feature_names"]], dtype=float)
    normalized = (values - np.asarray(model["means"])) / np.asarray(model["scales"])
    logit = float(normalized @ np.asarray(model["coefficients"]) + model["intercept"])
    return 1 / (1 + math.exp(-max(-40, min(40, logit))))


def filter_candidate_pages(
    pages: list[dict[str, Any]], model: dict[str, Any]
) -> list[list[dict[str, str]]]:
    """Filter OCR candidates using only OCR text, boxes, confidence, and page size."""
    filtered: list[list[dict[str, str]]] = []
    for page in pages:
        kept = []
        for line in page["lines"]:
            features = line_features(line.text, line.confidence, line.box, page["size"])
            if predict_probability(model, features) >= model["threshold"]:
                kept.append({"speaker": "UNKNOWN", "text": line.text})
        filtered.append(kept)
    return filtered


def run_weak_filter_experiment(
    root: Path, *, artifact_dir: Path, seed: int = 42, validation_fraction: float = 0.2
) -> dict[str, Any]:
    root = root.resolve()
    training_ids, validation_ids, split_path = _load_fixed_split(
        root, artifact_dir, seed=seed, validation_fraction=validation_fraction
    )
    baseline_path = artifact_dir / "experiments" / "validation_psm11_conf70_raw_rtl_geometry.jsonl"
    if not baseline_path.is_file():
        raise DatasetError(f"frozen RTL baseline predictions are missing: {baseline_path}")
    baseline_records = _read_jsonl(baseline_path)
    if [record.get("sequence_id") for record in baseline_records] != validation_ids:
        raise DatasetError("saved RTL baseline IDs/order do not match the frozen validation split")

    labels = _training_labels(root, training_ids)
    examples, label_counts = build_training_examples(
        root, training_ids, labels, artifact_dir
    )
    model = fit_logistic_filter(examples)
    experiment_dir = artifact_dir / "experiments" / EXPERIMENT_NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)
    training_examples_path = experiment_dir / "training_examples.jsonl"
    _write_jsonl(training_examples_path, examples)
    model_path = experiment_dir / "logistic_model.json"
    _atomic_write(model_path, json.dumps(model, ensure_ascii=False, indent=2) + "\n")

    filtered_records: list[dict[str, Any]] = []
    feature_cache: dict[str, list[dict[str, Any]]] = {}
    validation_candidates: dict[str, list[dict[str, Any]]] = {}
    for sequence_id in validation_ids:
        pages = _load_candidate_pages(root, sequence_id, artifact_dir)
        validation_candidates[sequence_id] = pages
        filtered_records.append(
            {
                "sequence_id": sequence_id,
                "pages": filter_candidate_pages(pages, model),
            }
        )
        feature_cache[sequence_id] = [
            {
                "page_index": page_index,
                "features": [
                    {
                        "line_index": line_index,
                        "probability": predict_probability(
                            model,
                            line_features(line.text, line.confidence, line.box, page["size"]),
                        ),
                    }
                    for line_index, line in enumerate(page["lines"])
                ],
            }
            for page_index, page in enumerate(pages)
        ]

    for baseline, filtered in zip(baseline_records, filtered_records, strict=True):
        cached_text = [
            [entry["text"] for entry in page] for page in baseline["pages"]
        ]
        extracted_text = [
            [line.text for line in page["lines"]]
            for page in validation_candidates[filtered["sequence_id"]]
        ]
        if cached_text != extracted_text:
            raise DatasetError(
                f"local OCR candidate text/order differs from frozen baseline for "
                f"{filtered['sequence_id']}; filter evaluation stopped"
            )

    predictions_path = experiment_dir / "validation_filtered_predictions.jsonl"
    _write_jsonl(predictions_path, filtered_records)
    probability_path = experiment_dir / "validation_candidate_probabilities.json"
    _atomic_write(
        probability_path,
        json.dumps(feature_cache, ensure_ascii=False, indent=2) + "\n",
    )

    # Validation references are read only after training, model fitting, and prediction.
    references = _validation_references(root, validation_ids)
    references_path = experiment_dir / "validation_references.jsonl"
    _write_jsonl(references_path, references)
    baseline_report = _official_score(
        root, artifact_dir, reference_path=references_path,
        prediction_path=baseline_path, name=f"{EXPERIMENT_NAME}_baseline",
    )
    filtered_report = _official_score(
        root, artifact_dir, reference_path=references_path,
        prediction_path=predictions_path, name=f"{EXPERIMENT_NAME}_filtered",
    )
    empty_counts = {
        "training_gold_empty_pages": sum(
            not page for sequence_id in training_ids for page in labels[sequence_id]
        ),
        "validation": {
            "baseline": _empty_page_metrics(references, baseline_records),
            "filtered": _empty_page_metrics(references, filtered_records),
            "baseline_predicted_empty_pages": sum(
                not page for record in baseline_records for page in record["pages"]
            ),
            "filtered_predicted_empty_pages": sum(
                not page for record in filtered_records for page in record["pages"]
            ),
            "pages_made_empty_by_filter": sum(
                bool(baseline_page) and not filtered_page
                for baseline, filtered in zip(
                    baseline_records, filtered_records, strict=True
                )
                for baseline_page, filtered_page in zip(
                    baseline["pages"], filtered["pages"], strict=True
                )
            ),
        },
    }
    summary = {
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "fixed_split_file": str(split_path),
        "training_sequences": len(training_ids),
        "validation_sequences": len(validation_ids),
        "validation_ids": validation_ids,
        "training_weak_label_counts": label_counts,
        "usable_training_examples": model["training_examples"],
        "weak_label_policy": {
            "positive": "fuzzy monotonic token alignment to supplied training text with >=0.60 content-token coverage",
            "negative": "unmatched candidate containing no alphanumeric characters",
            "ambiguous": "all other candidates; excluded from model fitting",
        },
        "feature_names": list(FEATURE_NAMES),
        "model": model,
        "inference_inputs_exclude_gold": True,
        "baseline_prediction_file": str(baseline_path),
        "filtered_prediction_file": str(predictions_path),
        "validation_line_counts": {
            "baseline": sum(
                len(page) for record in baseline_records for page in record["pages"]
            ),
            "filtered": sum(
                len(page) for record in filtered_records for page in record["pages"]
            ),
        },
        "weak_training_examples_file": str(training_examples_path),
        "empty_page_counts": empty_counts,
        "configurations": {
            "baseline": {
                "macro": baseline_report["macro"],
                "empty_pages": empty_counts["validation"]["baseline"],
            },
            "filtered": {
                "macro": filtered_report["macro"],
                "empty_pages": empty_counts["validation"]["filtered"],
            },
        },
        "limitations": [
            "OCR-to-reference alignments provide noisy line-level weak labels, not human-verified bubble classifications.",
            "Only punctuation/symbol-only unmatched strings are treated as negatives; ambiguous mismatches are excluded.",
            "The learned classifier can remove valid short utterances or retain sound effects; validation size is only 16 sequences.",
        ],
    }
    summary_path = experiment_dir / "metrics.json"
    _atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    comparison_path = experiment_dir / "comparison.csv"
    import csv

    with comparison_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "configuration",
                "text_order_score",
                "balanced_joint_f1",
                "joint_f1",
                "matched_token_coverage",
                "speaker_accuracy_on_matched",
                "gold_empty_pages",
                "correctly_empty_gold_pages",
                "false_positive_gold_empty_pages",
            ]
        )
        for name in ("baseline", "filtered"):
            result = summary["configurations"][name]
            metrics = result["macro"]
            empty = result["empty_pages"]
            writer.writerow(
                [
                    name,
                    metrics["text_order_score"],
                    metrics["balanced_joint_f1"],
                    metrics["joint_f1"],
                    metrics["matched_token_coverage"],
                    metrics["speaker_accuracy_on_matched"],
                    empty["gold_empty_pages"],
                    empty["correctly_empty_gold_pages"],
                    empty["false_positive_gold_empty_pages"],
                ]
            )
    print(f"Weak-label training candidates: {label_counts}")
    print(
        "Training gold-empty pages: "
        f"{empty_counts['training_gold_empty_pages']}"
    )
    print(f"Fitted model: {model_path}")
    print(f"Metrics: {summary_path}")
    print(f"Comparison: {comparison_path}")
    return summary


def _validation_references(
    root: Path, validation_ids: list[str]
) -> list[dict[str, Any]]:
    wanted = set(validation_ids)
    labels_path = root / "development" / "labels.jsonl"
    references: dict[str, dict[str, Any]] = {}
    with labels_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("sequence_id") in wanted:
                sequence_id = row["sequence_id"]
                references[sequence_id] = {
                    "sequence_id": sequence_id,
                    "pages": row["pages"],
                }
    missing = wanted - references.keys()
    if missing:
        raise DatasetError(f"validation labels missing IDs: {', '.join(sorted(missing))}")
    return [references[sequence_id] for sequence_id in validation_ids]


def _make_predictions(
    sequence_id: str, pages: list[dict[str, Any]], model: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    result: list[list[dict[str, str]]] = []
    for page in pages:
        output = []
        for line in page["lines"]:
            if model is not None:
                features = line_features(
                    line.text, line.confidence, line.box, page["size"]
                )
                probability = predict_probability(model, features)
                if probability < (model["threshold"] if threshold is None else threshold):
                    continue
            output.append({"speaker": "UNKNOWN", "text": line.text})
        result.append(output)
    return {"sequence_id": sequence_id, "pages": result}


def _score_records(
    root: Path, artifact_dir: Path, experiment_dir: Path,
    name: str, references: list[dict[str, Any]], predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_path = experiment_dir / f"{name}_references.jsonl"
    prediction_path = experiment_dir / f"{name}_predictions.jsonl"
    _write_jsonl(reference_path, references)
    _write_jsonl(prediction_path, predictions)
    return _official_score(
        root,
        artifact_dir,
        reference_path=reference_path,
        prediction_path=prediction_path,
        name=f"{CV_EXPERIMENT_NAME}_{name}",
    )


def _metric_spread(values: list[float | None]) -> dict[str, float | int | None]:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return {"folds": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "folds": len(defined),
        "mean": statistics.mean(defined),
        "std": statistics.pstdev(defined),
        "min": min(defined),
        "max": max(defined),
    }


def grouped_sequence_folds(
    sequence_ids: list[str], folds_count: int, seed: int
) -> list[list[str]]:
    if folds_count < 2 or folds_count > len(sequence_ids):
        raise DatasetError("fold count must be between two and the number of sequences")
    if len(set(sequence_ids)) != len(sequence_ids):
        raise DatasetError("sequence IDs must be unique before fold assignment")
    shuffled_ids = sequence_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)
    return [shuffled_ids[index::folds_count] for index in range(folds_count)]


def run_grouped_cv_and_final_validation(
    root: Path, *, artifact_dir: Path, seed: int = 42,
    validation_fraction: float = 0.2, folds_count: int = 4,
) -> dict[str, Any]:
    """Choose a threshold by grouped OOF training data, then score the fixed holdout once."""
    if folds_count < 2:
        raise DatasetError("grouped cross-validation requires at least two folds")
    root = root.resolve()
    training_ids, validation_ids, split_path = _load_fixed_split(
        root, artifact_dir, seed=seed, validation_fraction=validation_fraction
    )
    baseline_path = (
        artifact_dir
        / "experiments"
        / "validation_psm11_conf70_raw_rtl_geometry.jsonl"
    )
    if not baseline_path.is_file():
        raise DatasetError(f"frozen RTL baseline predictions are missing: {baseline_path}")
    frozen_baseline = _read_jsonl(baseline_path)
    if [item.get("sequence_id") for item in frozen_baseline] != validation_ids:
        raise DatasetError(
            "saved RTL baseline IDs/order do not match the persisted fixed validation split"
        )
    if folds_count > len(training_ids):
        raise DatasetError("fold count cannot exceed training sequence count")

    # OCR extraction is label-free and cached by sequence; all three pages stay grouped.
    training_candidates = {
        sequence_id: _load_candidate_pages(root, sequence_id, artifact_dir)
        for sequence_id in training_ids
    }
    validation_candidates = {
        sequence_id: _load_candidate_pages(root, sequence_id, artifact_dir)
        for sequence_id in validation_ids
    }
    for baseline in frozen_baseline:
        actual = [
            [line.text for line in page["lines"]]
            for page in validation_candidates[baseline["sequence_id"]]
        ]
        expected = [[entry["text"] for entry in page] for page in baseline["pages"]]
        if actual != expected:
            raise DatasetError(
                f"candidate OCR differs from frozen baseline for "
                f"{baseline['sequence_id']}; evaluation stopped"
            )

    labels = _training_labels(root, training_ids)
    fold_ids = grouped_sequence_folds(training_ids, folds_count, seed)
    fold_for_sequence = {
        sequence_id: fold_index
        for fold_index, sequence_ids in enumerate(fold_ids)
        for sequence_id in sequence_ids
    }
    if set(fold_for_sequence) != set(training_ids):
        raise DatasetError("grouped folds do not partition the training sequence IDs")

    models: list[dict[str, Any]] = []
    fold_label_counts: list[dict[str, int]] = []
    oof_probabilities: dict[str, list[list[float]]] = {}
    oof_baseline_records: list[dict[str, Any]] = []
    for fold_index, held_out_ids in enumerate(fold_ids):
        held_out_set = set(held_out_ids)
        fold_training_ids = [
            sequence_id for sequence_id in training_ids if sequence_id not in held_out_set
        ]
        fold_labels = {sequence_id: labels[sequence_id] for sequence_id in fold_training_ids}
        examples, counts = build_training_examples(
            root, fold_training_ids, fold_labels, artifact_dir
        )
        model = fit_logistic_filter(examples)
        models.append(model)
        fold_label_counts.append(counts)
        for sequence_id in held_out_ids:
            pages = training_candidates[sequence_id]
            oof_probabilities[sequence_id] = [
                [
                    predict_probability(
                        model,
                        line_features(line.text, line.confidence, line.box, page["size"]),
                    )
                    for line in page["lines"]
                ]
                for page in pages
            ]
            oof_baseline_records.append(_make_predictions(sequence_id, pages))
    if set(oof_probabilities) != set(training_ids):
        raise DatasetError("OOF inference did not produce exactly one prediction per training sequence")
    oof_baseline_records = [
        next(record for record in oof_baseline_records if record["sequence_id"] == sequence_id)
        for sequence_id in training_ids
    ]

    experiment_dir = artifact_dir / "experiments" / CV_EXPERIMENT_NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)
    oof_reference_records = [
        {"sequence_id": sequence_id, "pages": labels[sequence_id]}
        for sequence_id in training_ids
    ]
    _write_jsonl(
        experiment_dir / "fold_assignments.jsonl",
        [
            {"sequence_id": sequence_id, "fold": fold_for_sequence[sequence_id]}
            for sequence_id in training_ids
        ],
    )
    _atomic_write(
        experiment_dir / "oof_probabilities.json",
        json.dumps(oof_probabilities, ensure_ascii=False, indent=2) + "\n",
    )
    _write_jsonl(
        experiment_dir / "oof_baseline_predictions.jsonl", oof_baseline_records
    )
    threshold_reports: dict[str, dict[str, Any]] = {}
    threshold_empty_metrics: dict[str, dict[str, int]] = {}
    for threshold in THRESHOLD_GRID:
        threshold_records = [
            _make_predictions(
                sequence_id,
                training_candidates[sequence_id],
                models[fold_for_sequence[sequence_id]],
                threshold,
            )
            for sequence_id in training_ids
        ]
        threshold_reports[f"{threshold:.1f}"] = _score_records(
            root,
            artifact_dir,
            experiment_dir,
            f"oof_threshold_{threshold:.1f}",
            oof_reference_records,
            threshold_records,
        )
        threshold_empty_metrics[f"{threshold:.1f}"] = _empty_page_metrics(
            oof_reference_records, threshold_records
        )
    baseline_oof_report = _score_records(
        root,
        artifact_dir,
        experiment_dir,
        "oof_baseline",
        oof_reference_records,
        oof_baseline_records,
    )

    selected_threshold = max(
        THRESHOLD_GRID,
        key=lambda threshold: (
            threshold_reports[f"{threshold:.1f}"]["macro"]["balanced_joint_f1"] or 0.0,
            threshold_reports[f"{threshold:.1f}"]["macro"]["text_order_score"] or 0.0,
            threshold_reports[f"{threshold:.1f}"]["macro"]["joint_f1"] or 0.0,
            -abs(threshold - CONFIDENCE_THRESHOLD),
        ),
    )
    selected_fold_metrics: list[dict[str, Any]] = []
    for fold_index, held_out_ids in enumerate(fold_ids):
        references = [
            {"sequence_id": sequence_id, "pages": labels[sequence_id]}
            for sequence_id in held_out_ids
        ]
        baseline_records = [
            next(
                record for record in oof_baseline_records
                if record["sequence_id"] == sequence_id
            )
            for sequence_id in held_out_ids
        ]
        filtered_records = [
            _make_predictions(
                sequence_id,
                training_candidates[sequence_id],
                models[fold_index],
                selected_threshold,
            )
            for sequence_id in held_out_ids
        ]
        baseline_report = _score_records(
            root, artifact_dir, experiment_dir,
            f"fold_{fold_index}_baseline", references, baseline_records,
        )
        filtered_report = _score_records(
            root, artifact_dir, experiment_dir,
            f"fold_{fold_index}_selected", references, filtered_records,
        )
        selected_fold_metrics.append(
            {
                "fold": fold_index,
                "sequence_count": len(held_out_ids),
                "sequence_ids": held_out_ids,
                "baseline": baseline_report["macro"],
                "filtered": filtered_report["macro"],
                "baseline_empty_pages": _empty_page_metrics(references, baseline_records),
                "filtered_empty_pages": _empty_page_metrics(references, filtered_records),
            }
        )

    # Calibration is complete. Fit once on all 64 training sequences.
    training_examples, final_label_counts = build_training_examples(
        root, training_ids, labels, artifact_dir
    )
    final_model = fit_logistic_filter(training_examples)
    final_model["threshold"] = selected_threshold
    final_model_path = experiment_dir / "final_training_model.json"
    _atomic_write(
        final_model_path,
        json.dumps(final_model, ensure_ascii=False, indent=2) + "\n",
    )
    final_validation_predictions = [
        _make_predictions(
            sequence_id, validation_candidates[sequence_id], final_model
        )
        for sequence_id in validation_ids
    ]
    prediction_path = experiment_dir / "final_validation_predictions.jsonl"
    _write_jsonl(prediction_path, final_validation_predictions)
    # Validation references are first read now, after fit and all inference.
    validation_references = _validation_references(root, validation_ids)
    validation_reference_path = experiment_dir / "final_validation_references.jsonl"
    _write_jsonl(validation_reference_path, validation_references)
    baseline_final_report = _official_score(
        root,
        artifact_dir,
        reference_path=validation_reference_path,
        prediction_path=baseline_path,
        name=f"{CV_EXPERIMENT_NAME}_final_validation_baseline",
    )
    filtered_final_report = _official_score(
        root,
        artifact_dir,
        reference_path=validation_reference_path,
        prediction_path=prediction_path,
        name=f"{CV_EXPERIMENT_NAME}_final_validation_filtered",
    )
    baseline_empty = _empty_page_metrics(validation_references, frozen_baseline)
    filtered_empty = _empty_page_metrics(validation_references, final_validation_predictions)

    summary = {
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "fixed_split_file": str(split_path),
        "training_sequence_count": len(training_ids),
        "fixed_validation_sequence_count": len(validation_ids),
        "grouped_cv_folds": [
            {"fold": index, "sequence_count": len(ids), "sequence_ids": ids}
            for index, ids in enumerate(fold_ids)
        ],
        "folds_group_by_sequence": True,
        "oof_sequences": len(oof_probabilities),
        "threshold_grid": list(THRESHOLD_GRID),
        "selected_threshold_by_oof_balanced_joint_f1": selected_threshold,
        "oof_baseline": baseline_oof_report["macro"],
        "oof_empty_pages": {
            "baseline": _empty_page_metrics(
                oof_reference_records, oof_baseline_records
            ),
            "thresholds": threshold_empty_metrics,
        },
        "oof_threshold_metrics": {
            threshold: report["macro"]
            for threshold, report in threshold_reports.items()
        },
        "selected_threshold_fold_metrics": selected_fold_metrics,
        "selected_threshold_fold_spread": {
            model_metric: {
                "baseline": _metric_spread(
                    [fold["baseline"][model_metric] for fold in selected_fold_metrics]
                ),
                "filtered": _metric_spread(
                    [fold["filtered"][model_metric] for fold in selected_fold_metrics]
                ),
            }
            for model_metric in (
                "text_order_score",
                "balanced_joint_f1",
                "joint_f1",
                "matched_token_coverage",
                "speaker_accuracy_on_matched",
            )
        },
        "cv_fold_weak_label_counts": fold_label_counts,
        "final_training_weak_label_counts": final_label_counts,
        "final_training_weak_examples": len(training_examples),
        "final_model_file": str(final_model_path),
        "final_validation_prediction_file": str(prediction_path),
        "final_validation": {
            "baseline": {
                "macro": baseline_final_report["macro"],
                "empty_pages": baseline_empty,
            },
            "filtered": {
                "macro": filtered_final_report["macro"],
                "empty_pages": filtered_empty,
            },
        },
        "validation_labels_used_for_fit_or_inference": False,
        "limitations": [
            "Weak labels are noisy alignment-derived proxies; only symbol-only unmatched candidates are negative.",
            "OOF threshold selection is confined to the 64 training sequences, but the grouped sample and 16-sequence final validation remain small.",
            "No threshold should be described as statistically significant from this experiment.",
        ],
    }
    summary_path = experiment_dir / "metrics.json"
    _atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Grouped OOF baseline: {baseline_oof_report['macro']}")
    print("Grouped OOF thresholds:")
    for threshold, report in threshold_reports.items():
        print(f"  {threshold}: {report['macro']}")
    print(f"Selected threshold from training-only OOF: {selected_threshold:.1f}")
    print(f"Final frozen validation baseline: {baseline_final_report['macro']}")
    print(f"Final frozen validation filtered: {filtered_final_report['macro']}")
    print(f"Metrics and fold details: {summary_path}")
    return summary
