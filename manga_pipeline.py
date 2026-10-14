#!/usr/bin/env python3
"""Validate and load the BYTE Society manga-task dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

PAGES_PER_SEQUENCE = 3
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DatasetError(ValueError):
    """Raised when a dataset or prediction file violates the task schema."""


@dataclass(frozen=True, slots=True)
class LoadedImage:
    path: Path
    content: bytes


@dataclass(frozen=True, slots=True)
class SequenceData:
    sequence_id: str
    split: str
    images: tuple[LoadedImage, ...]
    pages: tuple[tuple[dict[str, str], ...], ...] | None


@dataclass(frozen=True, slots=True)
class OCRLine:
    text: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class TextRegion:
    polygon: tuple[tuple[float, float], ...]
    confidence: float


def preprocess_image(image: Image.Image, mode: str) -> Image.Image:
    if mode == "raw":
        return image.copy()
    if mode not in {"gray2x", "threshold2x"}:
        raise DatasetError(f"unsupported image preprocessing mode {mode!r}")
    processed = ImageOps.autocontrast(ImageOps.grayscale(image))
    if mode == "threshold2x":
        processed = processed.point(lambda pixel: 255 if pixel >= 175 else 0)
    width, height = processed.size
    resampling = Image.Resampling.NEAREST if mode == "threshold2x" else Image.Resampling.LANCZOS
    return processed.resize((width * 2, height * 2), resample=resampling)


def order_ocr_lines(lines: list[OCRLine], mode: str) -> list[OCRLine]:
    if mode == "tesseract":
        return lines
    if mode != "rtl_geometry":
        raise DatasetError(f"unsupported OCR reading-order mode {mode!r}")
    if len(lines) < 2:
        return lines

    bands: list[list[Any]] = []
    for line in sorted(lines, key=lambda item: item.box[1] + item.box[3] / 2):
        _, top, _, height = line.box
        center_y = top + height / 2
        tolerance = max(8.0, height * 0.55)
        matching = next((band for band in bands if abs(band[0] - center_y) <= max(tolerance, band[1])), None)
        if matching is None:
            bands.append([center_y, tolerance, [line]])
        else:
            matching[2].append(line)
            matching[0] = sum(
                item.box[1] + item.box[3] / 2 for item in matching[2]
            ) / len(matching[2])
            matching[1] = max(matching[1], tolerance)
    bands.sort(key=lambda band: band[0])
    return [
        line
        for _, _, band in bands
        for line in sorted(band, key=lambda item: item.box[0], reverse=True)
    ]


def _deduplicate_text_regions(
    regions: list[TextRegion],
    overlap_threshold: float = 0.75,
) -> list[TextRegion]:
    def bounds(region: TextRegion) -> tuple[float, float, float, float]:
        xs = [point[0] for point in region.polygon]
        ys = [point[1] for point in region.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    kept: list[TextRegion] = []
    for region in sorted(regions, key=lambda item: item.confidence, reverse=True):
        left, top, right, bottom = bounds(region)
        area = max(0.0, right - left) * max(0.0, bottom - top)
        duplicate = False
        for previous in kept:
            other_left, other_top, other_right, other_bottom = bounds(previous)
            other_area = max(0.0, other_right - other_left) * max(
                0.0, other_bottom - other_top
            )
            intersection = max(0.0, min(right, other_right) - max(left, other_left))
            intersection *= max(0.0, min(bottom, other_bottom) - max(top, other_top))
            if min(area, other_area) and intersection / min(area, other_area) >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(region)
    return kept


def _decode_comic_text_regions(
    line_map: Any,
    image_shape: tuple[int, int],
    *,
    pad_x: int,
    pad_y: int,
    input_size: int = 1024,
    threshold: float = 0.3,
    minimum_score: float = 0.6,
) -> list[TextRegion]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise DatasetError(
            "Comic-text-detector requires OpenCV; install the compatible local "
            "package with `python -m pip install opencv-python`."
        ) from error

    height, width = image_shape
    map_height, map_width = line_map.shape
    contours, _ = cv2.findContours(
        (line_map > threshold).astype(np.uint8) * 255,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    scale_x = width / (input_size - pad_x)
    scale_y = height / (input_size - pad_y)
    regions: list[TextRegion] = []
    for contour in contours[:1000]:
        if cv2.contourArea(contour) < 2:
            continue
        mask = np.zeros((map_height, map_width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        confidence = float(cv2.mean(line_map, mask=mask)[0])
        if confidence < minimum_score:
            continue
        points = cv2.boxPoints(cv2.minAreaRect(contour))
        polygon = tuple(
            (
                float(np.clip(point[0] * scale_x, 0, width - 1)),
                float(np.clip(point[1] * scale_y, 0, height - 1)),
            )
            for point in points
        )
        regions.append(TextRegion(polygon=polygon, confidence=confidence))
    return _deduplicate_text_regions(regions)


def _comic_text_detector_regions(image: Any, network: Any) -> list[TextRegion]:
    try:
        import cv2
    except ImportError as error:
        raise DatasetError(
            "Comic-text-detector requires OpenCV; install the compatible local "
            "package with `python -m pip install opencv-python`."
        ) from error

    input_size = 1024
    height, width = image.shape[:2]
    ratio = min(input_size / height, input_size / width)
    resized_size = (round(width * ratio), round(height * ratio))
    resized = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        resized_size,
        interpolation=cv2.INTER_LINEAR,
    )
    pad_x = input_size - resized_size[0]
    pad_y = input_size - resized_size[1]
    padded = cv2.copyMakeBorder(
        resized,
        0,
        pad_y,
        0,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    network.setInput(
        cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, size=(input_size, input_size))
    )
    output_names = list(network.getUnconnectedOutLayersNames())
    outputs = dict(zip(output_names, network.forward(output_names), strict=True))
    if "det" not in outputs or outputs["det"].ndim != 4:
        raise DatasetError(
            "comic-text-detector ONNX output is missing its four-dimensional 'det' map"
        )
    line_map = outputs["det"][0, 0]
    return _decode_comic_text_regions(
        line_map,
        (height, width),
        pad_x=pad_x,
        pad_y=pad_y,
        input_size=input_size,
    )


def _rectified_region_ocr(
    image: Any,
    region: TextRegion,
    *,
    tesseract: str,
    page_number: int,
) -> list[OCRLine]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise DatasetError("Comic-text-detector OCR requires OpenCV") from error

    points = np.asarray(region.polygon, dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered = np.asarray(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )
    top_left, top_right, bottom_right, bottom_left = ordered
    crop_width = max(
        4,
        round(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))),
    )
    crop_height = max(
        4,
        round(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))),
    )
    padding = max(2, round(crop_height * 0.2))
    destination = np.asarray(
        [
            [padding, padding],
            [padding + crop_width - 1, padding],
            [padding + crop_width - 1, padding + crop_height - 1],
            [padding, padding + crop_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    crop = cv2.warpPerspective(
        image,
        transform,
        (crop_width + 2 * padding, crop_height + 2 * padding),
        borderMode=cv2.BORDER_REPLICATE,
    )
    enlarged = cv2.resize(
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )
    with tempfile.TemporaryDirectory(prefix="comic-text-ocr-") as temporary_dir:
        crop_path = Path(temporary_dir) / "region.png"
        if not cv2.imwrite(str(crop_path), enlarged):
            raise DatasetError("could not write temporary comic text crop")
        try:
            result = subprocess.run(
                [
                    tesseract,
                    str(crop_path),
                    "stdout",
                    "-l",
                    "eng",
                    "--psm",
                    "7",
                    "tsv",
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise DatasetError(
                f"Tesseract failed on detected page {page_number} region: {error}"
            ) from error

    inverse = np.linalg.inv(transform)
    image_height, image_width = image.shape[:2]
    mapped_lines: list[OCRLine] = []
    for line in parse_tesseract_lines(result.stdout, min_confidence=0):
        x, y, width, height = line.box
        local_corners = np.asarray(
            [[
                [x / 3, y / 3],
                [(x + width) / 3, y / 3],
                [(x + width) / 3, (y + height) / 3],
                [x / 3, (y + height) / 3],
            ]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(local_corners, inverse)[0]
        left = max(0, int(np.floor(mapped[:, 0].min())))
        top = max(0, int(np.floor(mapped[:, 1].min())))
        right = min(image_width, int(np.ceil(mapped[:, 0].max())))
        bottom = min(image_height, int(np.ceil(mapped[:, 1].max())))
        if right > left and bottom > top:
            mapped_lines.append(
                OCRLine(line.text, line.confidence, (left, top, right - left, bottom - top))
            )
    return mapped_lines


def _save_comic_detector_overlay(
    image_path: Path,
    regions: list[TextRegion],
    lines: list[OCRLine],
    output_path: Path,
) -> None:
    from PIL import ImageDraw, ImageFont

    with Image.open(image_path) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for region in regions:
        draw.polygon(region.polygon, outline=(30, 120, 255), width=2)
    for line in lines:
        x, y, width, height = line.box
        draw.rectangle((x, y, x + width, y + height), outline=(20, 160, 45), width=2)
        draw.text(
            (x, max(0, y - 12)),
            f"{line.confidence:.0f}: {line.text[:48]}",
            fill=(0, 100, 20),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_comic_text_sequence(
    root: Path,
    sequence_id: str,
    *,
    artifact_dir: Path,
    model_path: Path,
    force: bool = False,
) -> Path:
    sequence = load_sequence(root, sequence_id, include_labels=False)
    if sequence.split != "development":
        raise DatasetError("comic detector pilot sequence must belong to development")
    if not model_path.is_file():
        raise DatasetError(
            f"comic-text-detector ONNX weights not found at {model_path}; download the "
            "94,669,756-byte comictextdetector.pt.onnx asset from release beta-0.2.1"
        )
    try:
        import cv2
    except ImportError as error:
        raise DatasetError(
            "Comic-text-detector requires OpenCV; install it with "
            "`python -m pip install opencv-python`."
        ) from error
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        raise DatasetError("Tesseract is not installed or not available on PATH")

    sequence_key = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()
    config_name = "comictextdetector_onnx_opencv_tesseract_psm7"
    cache_dir = artifact_dir / "intermediate" / sequence_key / config_name
    pending = [
        index
        for index in range(PAGES_PER_SEQUENCE)
        if force or not (cache_dir / f"page_{index + 1:02d}.json").is_file()
    ]
    network = None
    if pending:
        try:
            network = cv2.dnn.readNetFromONNX(str(model_path))
            network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        except cv2.error as error:
            raise DatasetError(f"OpenCV could not load comic detector ONNX weights: {error}") from error

    pages: list[list[dict[str, str]]] = []
    for page_index, loaded_image in enumerate(sequence.images):
        cache_path = cache_dir / f"page_{page_index + 1:02d}.json"
        if page_index in pending:
            image = cv2.imread(str(loaded_image.path), cv2.IMREAD_COLOR)
            if image is None:
                raise DatasetError(f"OpenCV could not read development page {loaded_image.path}")
            regions = _comic_text_detector_regions(image, network)
            lines = [
                line
                for region in regions
                for line in _rectified_region_ocr(
                    image,
                    region,
                    tesseract=tesseract,
                    page_number=page_index + 1,
                )
            ]
            lines = order_ocr_lines(lines, "rtl_geometry")
            cache_record = {
                "regions": [
                    {
                        "polygon": [list(point) for point in region.polygon],
                        "confidence": region.confidence,
                    }
                    for region in regions
                ],
                "lines": [
                    {
                        "text": line.text,
                        "confidence": line.confidence,
                        "box": list(line.box),
                    }
                    for line in lines
                ],
            }
            _atomic_write(cache_path, json.dumps(cache_record, ensure_ascii=False, indent=2) + "\n")
        else:
            cache_record = _read_json(cache_path)
            if set(cache_record) != {"regions", "lines"}:
                raise DatasetError(f"invalid comic detector cache: {cache_path}")

        regions = [
            TextRegion(
                polygon=tuple((float(point[0]), float(point[1])) for point in item["polygon"]),
                confidence=float(item["confidence"]),
            )
            for item in cache_record["regions"]
        ]
        lines = [
            OCRLine(
                text=str(item["text"]),
                confidence=float(item["confidence"]),
                box=tuple(int(value) for value in item["box"]),
            )
            for item in cache_record["lines"]
        ]
        _save_comic_detector_overlay(
            loaded_image.path,
            regions,
            lines,
            artifact_dir
            / "diagnostics"
            / "comic_text_detector"
            / f"{sequence_id}_page{page_index + 1:02d}.png",
        )
        pages.append(_ocr_text_to_page("\n".join(line.text for line in lines)))

    output_path = artifact_dir / "pilot" / f"{sequence_id}_comic_text_detector.jsonl"
    _write_jsonl(output_path, [{"sequence_id": sequence_id, "pages": pages}])
    validate_submission(output_path, [sequence_id])
    print(f"Detector regions / OCR lines by page: {[len(json.loads((cache_dir / f'page_{i + 1:02d}.json').read_text())['regions']) for i in range(PAGES_PER_SEQUENCE)]} / {[len(page) for page in pages]}")
    return output_path
def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, DatasetError) as error:
        raise DatasetError(f"{path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise DatasetError(f"{path}: {error}") from error

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, DatasetError) as error:
            raise DatasetError(f"{path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise DatasetError(f"{path}:{line_number}: each JSONL record must be an object")
        records.append(record)
    return records


def _sequence_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DatasetError(f"{where}: sequence_id must be a nonempty, trimmed string")
    return value


def _validate_pages(value: Any, where: str) -> tuple[tuple[dict[str, str], ...], ...]:
    if not isinstance(value, list) or len(value) != PAGES_PER_SEQUENCE:
        raise DatasetError(f"{where}: pages must contain exactly {PAGES_PER_SEQUENCE} lists")

    pages: list[tuple[dict[str, str], ...]] = []
    for page_number, page in enumerate(value, 1):
        page_where = f"{where}, page {page_number}"
        if not isinstance(page, list):
            raise DatasetError(f"{page_where}: page must be a list")
        entries: list[dict[str, str]] = []
        for entry_number, entry in enumerate(page, 1):
            entry_where = f"{page_where}, entry {entry_number}"
            if not isinstance(entry, dict) or set(entry) != {"speaker", "text"}:
                raise DatasetError(f"{entry_where}: each entry needs exactly speaker and text")
            speaker, text = entry["speaker"], entry["text"]
            if not isinstance(speaker, str) or not speaker or speaker != speaker.strip():
                raise DatasetError(f"{entry_where}: speaker must be a nonempty, trimmed string")
            if not isinstance(text, str):
                raise DatasetError(f"{entry_where}: text must be a string")
            meaningful_text = unicodedata.normalize("NFKC", text)
            meaningful_text = "".join(
                char for char in meaningful_text if unicodedata.category(char) != "Cf"
            )
            if not meaningful_text.strip():
                raise DatasetError(f"{entry_where}: text must not be empty")
            if any(
                unicodedata.category(char) == "Cc" and not char.isspace()
                for char in text
            ):
                raise DatasetError(f"{entry_where}: text contains a control character")
            entries.append({"speaker": speaker, "text": text})
        pages.append(tuple(entries))
    return tuple(pages)


def _manifest(root: Path) -> list[dict[str, Any]]:
    records = _read_json(root / "sequences.json")
    if not isinstance(records, list) or not records:
        raise DatasetError("sequences.json must be a nonempty list")

    root_resolved = root.resolve()
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        where = f"sequences.json record {index}"
        if not isinstance(record, dict) or set(record) != {"sequence_id", "split", "images"}:
            raise DatasetError(f"{where}: expected exactly sequence_id, split, and images")
        sequence_id = _sequence_id(record["sequence_id"], where)
        if sequence_id in seen:
            raise DatasetError(f"{where}: duplicate sequence_id {sequence_id}")
        seen.add(sequence_id)

        split = record["split"]
        if not isinstance(split, str) or split not in {"development", "test"}:
            raise DatasetError(f"{where}: unsupported split {split!r}")
        images = record["images"]
        if not isinstance(images, list) or len(images) != PAGES_PER_SEQUENCE:
            raise DatasetError(f"{where}: images must contain exactly {PAGES_PER_SEQUENCE} paths")

        image_paths: list[Path] = []
        for page_number, raw_path in enumerate(images, 1):
            image_where = f"{where}, image {page_number}"
            if not isinstance(raw_path, str):
                raise DatasetError(f"{image_where}: path must be a string")
            relative_path = PurePosixPath(raw_path)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.parts[:3] != (split, "images", sequence_id)
                or relative_path.suffix.lower() != ".png"
            ):
                raise DatasetError(f"{image_where}: invalid relative image path {raw_path!r}")
            path = root / Path(*relative_path.parts)
            if not path.resolve().is_relative_to(root_resolved):
                raise DatasetError(f"{image_where}: image path resolves outside the dataset")
            if not path.is_file():
                raise DatasetError(f"{image_where}: missing file {raw_path!r}")
            image_paths.append(path)

        if len(set(image_paths)) != PAGES_PER_SEQUENCE:
            raise DatasetError(f"{where}: image paths must be unique")
        validated.append(
            {"sequence_id": sequence_id, "split": split, "images": tuple(image_paths)}
        )
    return validated


def _labels(root: Path, manifest: list[dict[str, Any]]) -> dict[str, tuple[tuple[dict[str, str], ...], ...]]:
    records = _read_jsonl(root / "development" / "labels.jsonl")
    development_ids = {
        record["sequence_id"] for record in manifest if record["split"] == "development"
    }
    labels: dict[str, tuple[tuple[dict[str, str], ...], ...]] = {}
    for line_number, record in enumerate(records, 1):
        where = f"development/labels.jsonl record {line_number}"
        if set(record) != {"sequence_id", "pages"}:
            raise DatasetError(f"{where}: expected exactly sequence_id and pages")
        sequence_id = _sequence_id(record["sequence_id"], where)
        if sequence_id not in development_ids:
            raise DatasetError(f"{where}: unknown or non-development sequence_id {sequence_id}")
        if sequence_id in labels:
            raise DatasetError(f"{where}: duplicate sequence_id {sequence_id}")
        labels[sequence_id] = _validate_pages(record["pages"], where)

    missing = development_ids - labels.keys()
    if missing:
        raise DatasetError(f"development labels missing sequence_id(s): {', '.join(sorted(missing))}")
    return labels


def validate_submission(path: Path, expected_ids: list[str]) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    actual_ids: list[str] = []
    validated: list[dict[str, Any]] = []
    for line_number, record in enumerate(records, 1):
        where = f"{path}, record {line_number}"
        if set(record) != {"sequence_id", "pages"}:
            raise DatasetError(f"{where}: expected exactly sequence_id and pages")
        sequence_id = _sequence_id(record["sequence_id"], where)
        actual_ids.append(sequence_id)
        pages = _validate_pages(record["pages"], where)
        validated.append(
            {"sequence_id": sequence_id, "pages": [list(page) for page in pages]}
        )
    if actual_ids != expected_ids:
        raise DatasetError(
            "submission sequence IDs must exactly match the test template order "
            f"(expected {len(expected_ids)}, got {len(actual_ids)})"
        )
    return validated


def validate_dataset(root: Path) -> dict[str, int]:
    root = root.resolve()
    manifest = _manifest(root)
    labels = _labels(root, manifest)
    test_ids = [record["sequence_id"] for record in manifest if record["split"] == "test"]
    template = _read_jsonl(root / "sample_submission.jsonl")
    template_ids = []
    for line_number, record in enumerate(template, 1):
        where = f"sample_submission.jsonl record {line_number}"
        if set(record) != {"sequence_id", "pages"}:
            raise DatasetError(f"{where}: expected exactly sequence_id and pages")
        template_ids.append(_sequence_id(record["sequence_id"], where))
        if not isinstance(record["pages"], list) or len(record["pages"]) != PAGES_PER_SEQUENCE:
            raise DatasetError(f"{where}: pages must contain exactly {PAGES_PER_SEQUENCE} lists")
        if any(not isinstance(page, list) for page in record["pages"]):
            raise DatasetError(f"{where}: each page must be a list")
    if template_ids != test_ids:
        raise DatasetError("sample_submission.jsonl IDs/order do not match test manifest IDs/order")

    image_count = sum(len(record["images"]) for record in manifest)
    text_entries = sum(len(page) for pages in labels.values() for page in pages)
    return {
        "development_sequences": len(labels),
        "test_sequences": len(test_ids),
        "image_files": image_count,
        "development_text_entries": text_entries,
    }


def load_sequence(
    root: Path,
    sequence_id: str,
    *,
    include_labels: bool = True,
) -> SequenceData:
    """Load image bytes and development labels by ID in manifest page order."""
    root = root.resolve()
    manifest = _manifest(root)
    matches = [record for record in manifest if record["sequence_id"] == sequence_id]
    if not matches:
        raise DatasetError(f"unknown sequence_id {sequence_id!r}")
    record = matches[0]
    labels = _labels(root, manifest).get(sequence_id) if include_labels else None
    loaded_images: list[LoadedImage] = []
    for page_number, path in enumerate(record["images"], 1):
        try:
            content = path.read_bytes()
        except OSError as error:
            raise DatasetError(f"cannot read image for page {page_number}: {path}: {error}") from error
        if not content.startswith(PNG_SIGNATURE):
            raise DatasetError(f"page {page_number} is not a PNG image: {path}")
        loaded_images.append(LoadedImage(path=path, content=content))
    return SequenceData(
        sequence_id=sequence_id,
        split=record["split"],
        images=tuple(loaded_images),
        pages=labels,
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _ocr_text_to_page(text: str) -> list[dict[str, str]]:
    return [
        {"speaker": "UNKNOWN", "text": line.strip()}
        for line in text.splitlines()
        if line.strip()
    ]


def parse_tesseract_lines(tsv: str, min_confidence: int = 0) -> list[OCRLine]:
    lines: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        try:
            key = tuple(
                int(row[name])
                for name in ("page_num", "block_num", "par_num", "line_num")
            )
            confidence = float(row["conf"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError(f"invalid Tesseract TSV output: {error}") from error
        line = lines.setdefault(key, {"words": [], "confidence": []})
        line["words"].append(row["text"].strip())
        line["confidence"].append(confidence)
        left, top = int(row["left"]), int(row["top"])
        right, bottom = left + int(row["width"]), top + int(row["height"])
        line.setdefault("lefts", []).append(left)
        line.setdefault("tops", []).append(top)
        line.setdefault("rights", []).append(right)
        line.setdefault("bottoms", []).append(bottom)
    recognized: list[OCRLine] = []
    for key in sorted(lines):
        line = lines[key]
        confidence = sum(line["confidence"]) / len(line["confidence"])
        if confidence >= min_confidence:
            recognized.append(
                OCRLine(
                    text=" ".join(line["words"]),
                    confidence=confidence,
                    box=(
                        min(line["lefts"]),
                        min(line["tops"]),
                        max(line["rights"]) - min(line["lefts"]),
                        max(line["bottoms"]) - min(line["tops"]),
                    ),
                )
            )
    return recognized


def _parse_tesseract_tsv(
    tsv: str,
    min_confidence: int,
    reading_order: str = "tesseract",
) -> list[dict[str, str]]:
    lines = order_ocr_lines(parse_tesseract_lines(tsv, min_confidence), reading_order)
    return _ocr_text_to_page("\n".join(line.text for line in lines))


def _run_tesseract_ocr_lines(
    path: Path,
    *,
    psm: int,
    min_confidence: int,
    reading_order: str,
) -> list[OCRLine]:
    executable = shutil.which("tesseract")
    if executable is None:
        raise DatasetError(
            "Tesseract is not installed. On macOS install the local OCR engine with "
            "`brew install tesseract`."
        )
    try:
        result = subprocess.run(
            [
                executable,
                str(path),
                "stdout",
                "-l",
                "eng",
                "--psm",
                str(psm),
                "tsv",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DatasetError(f"Tesseract failed on {path}: {error}") from error
    return order_ocr_lines(
        parse_tesseract_lines(result.stdout, min_confidence),
        reading_order,
    )


def _run_tesseract(
    path: Path,
    psm: int,
    min_confidence: int,
    reading_order: str,
) -> list[dict[str, str]]:
    lines = _run_tesseract_ocr_lines(
        path,
        psm=psm,
        min_confidence=min_confidence,
        reading_order=reading_order,
    )
    return _ocr_text_to_page("\n".join(line.text for line in lines))


def run_tesseract_sequence(
    root: Path,
    sequence_id: str,
    *,
    artifact_dir: Path,
    psm: int,
    min_confidence: int = 0,
    min_lines: int = 1,
    preprocessing: str = "raw",
    reading_order: str = "tesseract",
    force: bool = False,
) -> list[list[dict[str, str]]]:
    if psm not in {6, 11}:
        raise DatasetError("supported Tesseract page segmentation modes are 6 and 11")
    if not 0 <= min_confidence <= 100:
        raise DatasetError("Tesseract minimum confidence must be between 0 and 100")
    if min_lines < 1:
        raise DatasetError("minimum OCR lines per page must be positive")
    if preprocessing not in {"raw", "gray2x", "threshold2x"}:
        raise DatasetError(f"unsupported image preprocessing mode {preprocessing!r}")
    if reading_order not in {"tesseract", "rtl_geometry"}:
        raise DatasetError(f"unsupported OCR reading-order mode {reading_order!r}")
    sequence = load_sequence(root, sequence_id, include_labels=False)
    key = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()
    config_name = f"tesseract_psm{psm}_conf{min_confidence}"
    config_name += f"_{preprocessing}_{reading_order}"
    if min_lines > 1:
        config_name += f"_minlines{min_lines}"
    cache_dir = artifact_dir / "intermediate" / key / config_name
    pages: list[list[dict[str, str]]] = []
    for page_index, image in enumerate(sequence.images):
        cache_path = cache_dir / f"page_{page_index + 1:02d}.json"
        if cache_path.is_file() and not force:
            page = _page_cache(_read_json(cache_path), f"cached OCR page {page_index + 1}")
        else:
            image_path = image.path
            if preprocessing != "raw":
                image_path = cache_dir / f"page_{page_index + 1:02d}_{preprocessing}.png"
                if not image_path.is_file():
                    with Image.open(image.path) as source:
                        processed = preprocess_image(source, preprocessing)
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        processed.save(image_path)
            page = _run_tesseract(image_path, psm, min_confidence, reading_order)
            if len(page) < min_lines:
                page = []
            _atomic_write(cache_path, json.dumps(page, ensure_ascii=False, indent=2) + "\n")
        pages.append(page)
    return pages


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )


def _official_score(
    root: Path,
    artifact_dir: Path,
    *,
    reference_path: Path,
    prediction_path: Path,
    name: str,
) -> dict[str, Any]:
    report_path = artifact_dir / "experiments" / f"{name}_score.json"
    command = [
        sys.executable,
        str(root / "score.py"),
        "--references",
        str(reference_path),
        "--predictions",
        str(prediction_path),
        "--output",
        str(report_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise DatasetError(f"official scorer failed: {detail.strip()}") from error
    report = _read_json(report_path)
    print(completed.stdout.strip())
    return report


def _sequence_level_split(
    sequence_ids: list[str],
    labels: dict[str, tuple[tuple[dict[str, str], ...], ...]],
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[list[str], list[str]]:
    validation_count = max(1, round(len(sequence_ids) * validation_fraction))
    empty_sequences = [
        sequence_id
        for sequence_id in sequence_ids
        if any(not page for page in labels[sequence_id])
    ]
    empty_sequence_set = set(empty_sequences)
    nonempty_sequences = [
        sequence_id for sequence_id in sequence_ids if sequence_id not in empty_sequence_set
    ]
    generator = random.Random(seed)
    generator.shuffle(empty_sequences)
    generator.shuffle(nonempty_sequences)

    if not empty_sequences:
        validation_ids = empty_sequences + nonempty_sequences[:validation_count]
    else:
        empty_validation_count = min(
            len(empty_sequences),
            max(1, round(validation_count * len(empty_sequences) / len(sequence_ids))),
        )
        nonempty_validation_count = validation_count - empty_validation_count
        if nonempty_validation_count > len(nonempty_sequences):
            empty_validation_count += nonempty_validation_count - len(nonempty_sequences)
            nonempty_validation_count = len(nonempty_sequences)
        validation_ids = (
            empty_sequences[:empty_validation_count]
            + nonempty_sequences[:nonempty_validation_count]
        )
        generator.shuffle(validation_ids)
    validation_set = set(validation_ids)
    training_ids = [sequence_id for sequence_id in sequence_ids if sequence_id not in validation_set]
    return training_ids, validation_ids


def _empty_page_metrics(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, int]:
    gold_empty = 0
    correctly_empty = 0
    false_positive_empty = 0
    for reference, prediction in zip(references, predictions, strict=True):
        for gold_page, predicted_page in zip(
            reference["pages"], prediction["pages"], strict=True
        ):
            if not gold_page:
                gold_empty += 1
                correctly_empty += not predicted_page
                false_positive_empty += bool(predicted_page)
    return {
        "gold_empty_pages": gold_empty,
        "correctly_empty_gold_pages": correctly_empty,
        "false_positive_gold_empty_pages": false_positive_empty,
    }


def _load_fixed_split(
    root: Path,
    artifact_dir: Path,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[list[str], list[str], Path]:
    experiments_dir = artifact_dir / "experiments"
    split_path = experiments_dir / "fixed_validation_split.json"
    manifest = _manifest(root)
    development_ids = [
        record["sequence_id"] for record in manifest if record["split"] == "development"
    ]
    if split_path.is_file():
        split_record = _read_json(split_path)
    else:
        previous_path = experiments_dir / "validation_metrics.json"
        previous = _read_json(previous_path) if previous_path.is_file() else {}
        if (
            isinstance(previous, dict)
            and isinstance(previous.get("training_sequences"), list)
            and isinstance(previous.get("validation_sequences"), list)
        ):
            split_record = {
                "split_seed": previous.get("split_seed", seed),
                "validation_fraction": previous.get("validation_fraction", validation_fraction),
                "training_sequences": previous["training_sequences"],
                "validation_sequences": previous["validation_sequences"],
            }
        else:
            training_ids, validation_ids = _sequence_level_split(
                development_ids,
                _labels(root, manifest),
                seed=seed,
                validation_fraction=validation_fraction,
            )
            split_record = {
                "split_seed": seed,
                "validation_fraction": validation_fraction,
                "training_sequences": training_ids,
                "validation_sequences": validation_ids,
            }
        _atomic_write(split_path, json.dumps(split_record, indent=2) + "\n")

    training_ids = split_record.get("training_sequences")
    validation_ids = split_record.get("validation_sequences")
    if not isinstance(training_ids, list) or not isinstance(validation_ids, list):
        raise DatasetError(f"{split_path}: expected training_sequences and validation_sequences lists")
    combined = training_ids + validation_ids
    if len(set(combined)) != len(combined) or set(combined) != set(development_ids):
        raise DatasetError(f"{split_path}: split does not partition current development IDs")
    if len(validation_ids) != max(1, round(len(development_ids) * validation_fraction)):
        raise DatasetError(f"{split_path}: validation split has an unexpected size")
    return training_ids, validation_ids, split_path


def run_comic_detector_experiment(
    root: Path,
    *,
    artifact_dir: Path,
    model_path: Path,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    root = root.resolve()
    training_ids, validation_ids, split_path = _load_fixed_split(
        root,
        artifact_dir,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    experiments_dir = artifact_dir / "experiments"
    baseline_path = experiments_dir / "validation_psm11_conf70_raw_rtl_geometry.jsonl"
    if not baseline_path.is_file():
        raise DatasetError(f"frozen RTL baseline predictions are missing: {baseline_path}")
    baseline_records = _read_jsonl(baseline_path)
    if [record.get("sequence_id") for record in baseline_records] != validation_ids:
        raise DatasetError(
            "saved RTL baseline IDs/order do not match the persisted fixed validation split"
        )

    detector_records: list[dict[str, Any]] = []
    for sequence_id in validation_ids:
        prediction_path = run_comic_text_sequence(
            root,
            sequence_id,
            artifact_dir=artifact_dir,
            model_path=model_path,
        )
        records = _read_jsonl(prediction_path)
        if len(records) != 1 or records[0].get("sequence_id") != sequence_id:
            raise DatasetError(f"invalid resumable detector prediction for {sequence_id}")
        detector_records.append(records[0])

    labels = _labels(root, _manifest(root))
    references = [
        {"sequence_id": sequence_id, "pages": labels[sequence_id]}
        for sequence_id in validation_ids
    ]
    reference_path = experiments_dir / "comic_detector_validation_references.jsonl"
    _write_jsonl(reference_path, references)

    baseline_report = _official_score(
        root,
        artifact_dir,
        reference_path=reference_path,
        prediction_path=baseline_path,
        name="comic_detector_baseline_recheck",
    )
    detector_path = experiments_dir / "validation_comic_text_detector.jsonl"
    _write_jsonl(detector_path, detector_records)
    detector_report = _official_score(
        root,
        artifact_dir,
        reference_path=reference_path,
        prediction_path=detector_path,
        name="comic_detector_candidate",
    )

    summary = {
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "fixed_split_file": str(split_path),
        "training_sequence_count": len(training_ids),
        "validation_sequence_count": len(validation_ids),
        "validation_sequences": validation_ids,
        "baseline_prediction_file": str(baseline_path),
        "detector_prediction_file": str(detector_path),
        "model_file": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "inference": {
            "detector": "dmMaze/comic-text-detector ONNX via OpenCV DNN CPU",
            "region_map_threshold": 0.3,
            "minimum_region_score": 0.6,
            "duplicate_overlap_threshold": 0.75,
            "tesseract_language": "eng",
            "tesseract_psm": 7,
            "crop_upscale": 3,
            "reading_order": "rtl_geometry",
            "inference_reads_development_labels": False,
        },
        "configurations": {
            "raw_tesseract_rtl_baseline": {
                "macro": baseline_report["macro"],
                "empty_pages": _empty_page_metrics(references, baseline_records),
            },
            "comic_detector_onnx_tesseract": {
                "macro": detector_report["macro"],
                "empty_pages": _empty_page_metrics(references, detector_records),
            },
        },
    }
    summary_path = experiments_dir / "comic_detector_validation_metrics.json"
    _atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    comparison_path = experiments_dir / "comic_detector_validation_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "configuration",
                "text_order_score",
                "balanced_joint_f1",
                "matched_token_coverage",
                "joint_f1",
                "speaker_accuracy_on_matched",
                "gold_empty_pages",
                "correctly_empty_gold_pages",
                "false_positive_gold_empty_pages",
            ]
        )
        for name, configuration in summary["configurations"].items():
            writer.writerow(
                [
                    name,
                    configuration["macro"]["text_order_score"],
                    configuration["macro"]["balanced_joint_f1"],
                    configuration["macro"]["matched_token_coverage"],
                    configuration["macro"]["joint_f1"],
                    configuration["macro"]["speaker_accuracy_on_matched"],
                    configuration["empty_pages"]["gold_empty_pages"],
                    configuration["empty_pages"]["correctly_empty_gold_pages"],
                    configuration["empty_pages"]["false_positive_gold_empty_pages"],
                ]
            )
    print(f"Fixed-split comic detector metrics: {summary_path}")
    print(f"Comparison table: {comparison_path}")
    return summary


def run_validation_experiment(
    root: Path,
    *,
    artifact_dir: Path,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    if not 0 < validation_fraction < 1:
        raise DatasetError("validation fraction must be strictly between 0 and 1")
    root = root.resolve()
    manifest = _manifest(root)
    labels = _labels(root, manifest)
    development_ids = [record["sequence_id"] for record in manifest if record["split"] == "development"]
    training_ids, validation_ids, split_path = _load_fixed_split(
        root,
        artifact_dir,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    validation_empty_pages = sum(
        not page for sequence_id in validation_ids for page in labels[sequence_id]
    )

    experiments_dir = artifact_dir / "experiments"
    reference_path = experiments_dir / "validation_references.jsonl"
    _write_jsonl(
        reference_path,
        [
            {"sequence_id": sequence_id, "pages": labels[sequence_id]}
            for sequence_id in validation_ids
        ],
    )
    summary: dict[str, Any] = {
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "stratification": "presence of an empty page; all pages stay with their sequence",
        "validation_gold_empty_pages": validation_empty_pages,
        "fixed_split_file": str(split_path),
        "training_sequences": training_ids,
        "validation_sequences": validation_ids,
        "configurations": {},
    }
    configurations = (
        (11, 70, 1, "raw", "tesseract"),
        (11, 70, 1, "gray2x", "tesseract"),
        (11, 70, 1, "threshold2x", "tesseract"),
        (11, 70, 1, "raw", "rtl_geometry"),
    )
    for psm, min_confidence, min_lines, preprocessing, reading_order in configurations:
        name = f"psm{psm}_conf{min_confidence}"
        if min_lines > 1:
            name += f"_minlines{min_lines}"
        name += f"_{preprocessing}_{reading_order}"
        predictions = [
            {
                "sequence_id": sequence_id,
                "pages": run_tesseract_sequence(
                    root,
                    sequence_id,
                    artifact_dir=artifact_dir,
                    psm=psm,
                    min_confidence=min_confidence,
                    min_lines=min_lines,
                    preprocessing=preprocessing,
                    reading_order=reading_order,
                ),
            }
            for sequence_id in validation_ids
        ]
        prediction_path = experiments_dir / f"validation_{name}.jsonl"
        _write_jsonl(prediction_path, predictions)
        report = _official_score(
            root,
            artifact_dir,
            reference_path=reference_path,
            prediction_path=prediction_path,
            name=f"validation_{name}",
        )
        reference_records = [
            {"sequence_id": sequence_id, "pages": labels[sequence_id]}
            for sequence_id in validation_ids
        ]
        summary["configurations"][name] = {
            "psm": psm,
            "minimum_confidence": min_confidence,
            "minimum_lines_per_page": min_lines,
            "preprocessing": preprocessing,
            "reading_order": reading_order,
            "prediction_file": str(prediction_path),
            "macro": report["macro"],
            "empty_pages": _empty_page_metrics(reference_records, predictions),
        }
    summary["recommended_configuration"] = max(
        summary["configurations"],
        key=lambda name: (
            summary["configurations"][name]["macro"]["balanced_joint_f1"] or 0.0,
            summary["configurations"][name]["macro"]["text_order_score"] or 0.0,
            summary["configurations"][name]["macro"]["matched_token_coverage"] or 0.0,
        ),
    )
    summary_path = experiments_dir / "validation_metrics.json"
    _atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    comparison_path = experiments_dir / "validation_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "configuration",
                "preprocessing",
                "reading_order",
                "text_order_score",
                "balanced_joint_f1",
                "matched_token_coverage",
                "joint_f1",
                "speaker_accuracy_on_matched",
                "gold_empty_pages",
                "correctly_empty_gold_pages",
                "false_positive_gold_empty_pages",
            ]
        )
        for name, configuration in summary["configurations"].items():
            writer.writerow(
                [
                    name,
                    configuration["preprocessing"],
                    configuration["reading_order"],
                    configuration["macro"]["text_order_score"],
                    configuration["macro"]["balanced_joint_f1"],
                    configuration["macro"]["matched_token_coverage"],
                    configuration["macro"]["joint_f1"],
                    configuration["macro"]["speaker_accuracy_on_matched"],
                    configuration["empty_pages"]["gold_empty_pages"],
                    configuration["empty_pages"]["correctly_empty_gold_pages"],
                    configuration["empty_pages"]["false_positive_gold_empty_pages"],
                ]
            )
    print(f"Sequence-level split and validation metrics: {summary_path}")
    print(f"Comparison table: {comparison_path}")
    print(
        f"Training sequences: {len(training_ids)}; "
        f"validation sequences: {len(validation_ids)}; seed: {seed}"
    )
    return summary


def run_speaker_validation_experiment(
    root: Path,
    *,
    artifact_dir: Path,
    model_path: Path,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    root = root.resolve()
    training_ids, validation_ids, split_path = _load_fixed_split(
        root,
        artifact_dir,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    experiments_dir = artifact_dir / "experiments"
    baseline_path = (
        experiments_dir
        / "weak_text_filter_groupcv_seed42"
        / "final_validation_predictions.jsonl"
    )
    filter_model_path = (
        experiments_dir
        / "weak_text_filter_groupcv_seed42"
        / "final_training_model.json"
    )
    if not baseline_path.is_file() or not filter_model_path.is_file():
        raise DatasetError(
            "selected threshold-0.7 validation predictions or filter model are missing"
        )
    baseline_records = _read_jsonl(baseline_path)
    if [record.get("sequence_id") for record in baseline_records] != validation_ids:
        raise DatasetError(
            "selected OCR prediction IDs/order do not match the persisted validation split"
        )
    filter_model = _read_json(filter_model_path)
    if filter_model.get("threshold") != 0.7:
        raise DatasetError("speaker experiment requires the selected threshold-0.7 OCR filter")
    if not model_path.is_dir() or not (model_path / "model.safetensors").is_file():
        raise DatasetError(f"local MLX-VLM model directory is incomplete or unavailable: {model_path}")
    if (model_path / "model.safetensors").stat().st_size < 100_000_000:
        raise DatasetError(f"local MLX-VLM weight file is incomplete: {model_path}")
    try:
        from mlx_vlm import load

        loaded_model = load(str(model_path.resolve()))
    except ImportError as error:
        raise DatasetError("MLX-VLM is required for local speaker attribution") from error
    except (OSError, ValueError, RuntimeError, TypeError, MemoryError) as error:
        raise DatasetError(
            f"could not load local MLX-VLM model {model_path}: {type(error).__name__}: {error}"
        ) from error

    from weak_text_filter import (
        _load_candidate_pages,
        _make_predictions,
        _validation_references,
        line_features,
        predict_probability,
    )

    all_unknown_records = [
        {
            "sequence_id": record["sequence_id"],
            "pages": [
                [
                    {"speaker": "UNKNOWN", "text": entry["text"]}
                    for entry in page
                ]
                for page in record["pages"]
            ],
        }
        for record in baseline_records
    ]
    all_unknown_path = experiments_dir / "validation_qwen25_filtered_all_unknown.jsonl"
    speaker_path = experiments_dir / "validation_qwen25_filtered_speaker.jsonl"
    _write_jsonl(all_unknown_path, all_unknown_records)
    validate_submission(all_unknown_path, validation_ids)

    speaker_records: list[dict[str, Any]] = []
    for sequence_id, baseline in zip(validation_ids, baseline_records, strict=True):
        sequence = load_sequence(root, sequence_id, include_labels=False)
        if sequence.split != "development":
            raise DatasetError(f"validation inference received non-development ID {sequence_id}")
        candidate_pages = _load_candidate_pages(root, sequence_id, artifact_dir)
        selected_by_filter = _make_predictions(
            sequence_id, candidate_pages, filter_model
        )
        if selected_by_filter != baseline:
            raise DatasetError(
                f"saved selected OCR does not match threshold-0.7 inference for {sequence_id}"
            )
        line_pages: list[list[OCRLine]] = []
        for page in candidate_pages:
            selected_lines = [
                line
                for line in page["lines"]
                if predict_probability(
                    filter_model,
                    line_features(line.text, line.confidence, line.box, page["size"]),
                )
                >= filter_model["threshold"]
            ]
            line_pages.append(selected_lines)

        prediction_path = run_speaker_pilot(
            root,
            sequence_id,
            model_path=model_path,
            artifact_dir=artifact_dir,
            loaded_model=loaded_model,
            candidate_pages=line_pages,
            expected_pages=baseline["pages"],
            allow_validation=True,
        )
        predictions = _read_jsonl(prediction_path)
        if len(predictions) != 1 or predictions[0].get("sequence_id") != sequence_id:
            raise DatasetError(f"invalid resumable speaker prediction for {sequence_id}")
        if [
            [entry["text"] for entry in page] for page in predictions[0]["pages"]
        ] != [
            [entry["text"] for entry in page] for page in baseline["pages"]
        ]:
            raise DatasetError(
                f"speaker-only inference changed OCR text/order for {sequence_id}"
            )
        speaker_records.append(predictions[0])

    if [record["sequence_id"] for record in speaker_records] != validation_ids:
        raise DatasetError("speaker inference did not produce all validation IDs in order")
    for unknown, attributed in zip(all_unknown_records, speaker_records, strict=True):
        if unknown["sequence_id"] != attributed["sequence_id"] or [
            [entry["text"] for entry in page] for page in unknown["pages"]
        ] != [
            [entry["text"] for entry in page] for page in attributed["pages"]
        ]:
            raise DatasetError(
                f"speaker inference changed selected OCR text/order for "
                f"{unknown['sequence_id']}"
            )
    _write_jsonl(speaker_path, speaker_records)
    validate_submission(speaker_path, validation_ids)

    # References are loaded only after the complete inference output is frozen.
    references = _validation_references(root, validation_ids)
    reference_path = experiments_dir / "validation_qwen25_speaker_references.jsonl"
    _write_jsonl(reference_path, references)
    baseline_report = _official_score(
        root,
        artifact_dir,
        reference_path=reference_path,
        prediction_path=all_unknown_path,
        name="speaker_qwen25_filtered_unknown_baseline",
    )
    speaker_report = _official_score(
        root,
        artifact_dir,
        reference_path=reference_path,
        prediction_path=speaker_path,
        name="speaker_qwen25_filtered_attributed",
    )
    assignment_counts: dict[str, int] = {}
    character_counts_by_sequence: dict[str, int] = {}
    for record in speaker_records:
        labels = [
            entry["speaker"]
            for page in record["pages"]
            for entry in page
        ]
        for label in labels:
            assignment_counts[label] = assignment_counts.get(label, 0) + 1
        character_counts_by_sequence[record["sequence_id"]] = len(
            {label for label in labels if re.fullmatch(r"character[1-9][0-9]*", label)}
        )
    sequences_with_multiple_characters = sum(
        count >= 2 for count in character_counts_by_sequence.values()
    )
    summary = {
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "fixed_split_file": str(split_path),
        "training_sequence_count": len(training_ids),
        "validation_sequence_count": len(validation_ids),
        "validation_sequences": validation_ids,
        "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        "model_path": str(model_path),
        "inference": {
            "prompt": "speaker-only indexed batches of 10, with prior visual character anchors",
            "response_parser": (
                "indexed JSON; malformed, duplicate-fence, missing, duplicate-index, "
                "or invalid assignment abstains to UNKNOWN"
            ),
            "images_per_sequence": PAGES_PER_SEQUENCE,
            "image_order": "manifest page order",
            "ocr": "selected threshold-0.7 weak filter over Tesseract PSM 11/confidence 70/geometric RTL",
            "candidate_boxes_in_prompt": True,
            "batch_size": 10,
            "max_tokens": "min(1024, max(384, batch_size * 64))",
            "temperature": 0.0,
            "repetition_penalty": 1.05,
            "fallback": "UNKNOWN for malformed, missing, duplicate, or invalid assignment indices",
            "ocr_text_and_order_unchanged": True,
            "validation_labels_in_inference": False,
        },
        "baseline_prediction_file": str(baseline_path),
        "all_unknown_prediction_file": str(all_unknown_path),
        "speaker_prediction_file": str(speaker_path),
        "speaker_assignment_counts": dict(sorted(assignment_counts.items())),
        "character_label_counts_by_sequence": character_counts_by_sequence,
        "sequences_with_multiple_character_labels": sequences_with_multiple_characters,
        "configurations": {
            "all_unknown_baseline": {
                "macro": baseline_report["macro"],
                "empty_pages": _empty_page_metrics(references, baseline_records),
            },
            "qwen2vl_speaker_only": {
                "macro": speaker_report["macro"],
                "empty_pages": _empty_page_metrics(references, speaker_records),
            },
        },
        "selection": (
            "selected all-UNKNOWN: Qwen speaker labels are evaluated only after inference, "
            "and are rejected if held-out metrics do not improve and labels are not meaningful"
        ),
    }
    summary_path = experiments_dir / "speaker_attribution_validation_metrics_qwen25_filtered.json"
    _atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    comparison_path = experiments_dir / "speaker_attribution_validation_comparison_qwen25_filtered.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "configuration",
                "text_order_score",
                "balanced_joint_f1",
                "matched_token_coverage",
                "joint_f1",
                "speaker_accuracy_on_matched",
                "gold_empty_pages",
                "correctly_empty_gold_pages",
                "false_positive_gold_empty_pages",
            ]
        )
        for name, configuration in summary["configurations"].items():
            writer.writerow(
                [
                    name,
                    configuration["macro"]["text_order_score"],
                    configuration["macro"]["balanced_joint_f1"],
                    configuration["macro"]["matched_token_coverage"],
                    configuration["macro"]["joint_f1"],
                    configuration["macro"]["speaker_accuracy_on_matched"],
                    configuration["empty_pages"]["gold_empty_pages"],
                    configuration["empty_pages"]["correctly_empty_gold_pages"],
                    configuration["empty_pages"]["false_positive_gold_empty_pages"],
                ]
            )
    print(f"Fixed-split speaker metrics: {summary_path}")
    print(f"Comparison table: {comparison_path}")
    return summary


def generate_test_submission(
    root: Path,
    *,
    artifact_dir: Path,
    output_path: Path,
    psm: int = 11,
    min_confidence: int = 70,
    min_lines: int = 1,
    reading_order: str = "rtl_geometry",
    weak_filter_model: Path | None = None,
) -> list[dict[str, Any]]:
    root = root.resolve()
    validate_dataset(root)
    template = _read_jsonl(root / "sample_submission.jsonl")
    if weak_filter_model is not None:
        if (psm, min_confidence, min_lines, reading_order) != (
            11, 70, 1, "rtl_geometry"
        ):
            raise DatasetError(
                "the saved weak filter requires PSM 11, confidence 70, "
                "minimum one OCR line, and geometric RTL order"
            )
        if not weak_filter_model.is_file():
            raise DatasetError(f"weak filter model file does not exist: {weak_filter_model}")
        from weak_text_filter import _load_candidate_pages, _make_predictions

        model = _read_json(weak_filter_model)
        records = [
            _make_predictions(
                record["sequence_id"],
                _load_candidate_pages(root, record["sequence_id"], artifact_dir),
                model,
            )
            for record in template
        ]
    else:
        records = [
            {
                "sequence_id": record["sequence_id"],
                "pages": run_tesseract_sequence(
                    root,
                    record["sequence_id"],
                    artifact_dir=artifact_dir,
                    psm=psm,
                    min_confidence=min_confidence,
                    min_lines=min_lines,
                    reading_order=reading_order,
                ),
            }
            for record in template
        ]
    _write_jsonl(output_path, records)
    validated = validate_submission(output_path, [record["sequence_id"] for record in template])
    print(f"Submission generated and validated: {output_path} ({len(validated)} sequences)")
    return validated


def score_development_predictions(
    root: Path,
    *,
    artifact_dir: Path,
    prediction_path: Path,
    name: str,
) -> dict[str, Any]:
    root = root.resolve()
    labels = _labels(root, _manifest(root))
    predictions = _read_jsonl(prediction_path)
    if not predictions:
        raise DatasetError("prediction file contains no records")
    seen: set[str] = set()
    references: list[dict[str, Any]] = []
    for index, record in enumerate(predictions, 1):
        where = f"{prediction_path}, record {index}"
        if set(record) != {"sequence_id", "pages"}:
            raise DatasetError(f"{where}: expected exactly sequence_id and pages")
        sequence_id = _sequence_id(record["sequence_id"], where)
        if sequence_id not in labels:
            raise DatasetError(f"{where}: not a labeled development sequence")
        if sequence_id in seen:
            raise DatasetError(f"{where}: duplicate sequence_id {sequence_id}")
        seen.add(sequence_id)
        _validate_pages(record["pages"], where)
        references.append({"sequence_id": sequence_id, "pages": labels[sequence_id]})
    reference_path = artifact_dir / "experiments" / f"{name}_references.jsonl"
    _write_jsonl(reference_path, references)
    return _official_score(
        root,
        artifact_dir,
        reference_path=reference_path,
        prediction_path=prediction_path,
        name=name,
    )


def _page_cache(value: Any, where: str) -> list[dict[str, str]]:
    return list(_validate_pages([value, [], []], where)[0])


def _parse_model_page(response: str, where: str) -> list[dict[str, str]]:
    content = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        content = fenced.group(1).strip()
    try:
        parsed = json.loads(content, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, DatasetError) as error:
        raise DatasetError(f"{where}: local model did not return valid JSON: {error}") from error
    if not isinstance(parsed, list):
        raise DatasetError(f"{where}: local model response must be a JSON list")
    return _page_cache(parsed, where)


def _page_prompt(page_number: int) -> str:
    return (
        "Look only at the attached manga image, page "
        f"{page_number}. Transcribe the exact legible text in speech balloons and "
        "narration boxes in reading order. Never list character names or describe "
        "the picture. Return only a JSON array; every item must have exactly the "
        'string fields "speaker" and "text". For dialogue use speaker "UNKNOWN" '
        'rather than guessing identity; use "NARRATION" only for text visibly '
        "outside a speech balloon. Keep each balloon as one item. Do not translate, "
        "continue, or invent text. If there is no readable text, return []."
    )


def parse_speaker_assignments(response: str, line_count: int) -> list[str]:
    """Parse index-only speaker decisions; invalid, missing, or repeated indices abstain."""
    if line_count < 0:
        raise DatasetError("line_count cannot be negative")
    speakers = ["UNKNOWN"] * line_count
    content = response.strip()
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if len(fenced_blocks) > 1:
        return speakers
    if fenced_blocks:
        content = fenced_blocks[0].strip()
    try:
        parsed = json.loads(content, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, DatasetError):
        return speakers
    if isinstance(parsed, dict) and set(parsed) == {"assignments"}:
        assignments = parsed["assignments"]
    elif isinstance(parsed, list):
        assignments = parsed
    else:
        return speakers
    if not isinstance(assignments, list):
        return speakers

    valid_speakers = re.compile(r"(?:UNKNOWN|NARRATION|character[1-9][0-9]*)\Z")
    assigned: dict[int, str] = {}
    ambiguous: set[int] = set()
    seen: set[int] = set()
    for item in assignments:
        if not isinstance(item, dict):
            continue
        index = item.get("line_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < line_count:
            continue
        if index in seen:
            ambiguous.add(index)
            continue
        seen.add(index)
        if set(item) != {"line_index", "speaker"}:
            ambiguous.add(index)
            continue
        speaker = item["speaker"]
        if not isinstance(speaker, str) or not valid_speakers.fullmatch(speaker):
            ambiguous.add(index)
            continue
        assigned[index] = speaker
    for index, speaker in assigned.items():
        if index not in ambiguous:
            speakers[index] = speaker
    return speakers


def apply_speaker_assignments(
    pages: list[list[OCRLine]],
    speakers: list[str],
) -> list[list[dict[str, str]]]:
    lines = [line for page in pages for line in page]
    if len(lines) != len(speakers):
        raise DatasetError(
            f"speaker assignments count ({len(speakers)}) does not match OCR lines ({len(lines)})"
        )
    assigned_pages: list[list[dict[str, str]]] = []
    offset = 0
    for page in pages:
        assigned_pages.append(
            [
                {"speaker": speakers[offset + index], "text": line.text}
                for index, line in enumerate(page)
            ]
        )
        offset += len(page)
    return assigned_pages


def _speaker_prompt(
    pages: list[list[OCRLine]],
    *,
    candidate_indices: list[int] | None = None,
    known_assignments: dict[int, str] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    page_lines: list[str] = []
    line_locations: list[tuple[int, int]] = []
    flat_lines: list[tuple[int, OCRLine]] = []
    global_index = 0
    for page_number, page in enumerate(pages, 1):
        for line in page:
            line_locations.append((page_number, global_index))
            flat_lines.append((page_number, line))
            global_index += 1
    selected_indices = (
        list(range(global_index)) if candidate_indices is None else candidate_indices
    )
    if (
        len(set(selected_indices)) != len(selected_indices)
        or any(index < 0 or index >= global_index for index in selected_indices)
    ):
        raise DatasetError("speaker prompt candidate indices must be unique and in range")
    assignments = known_assignments or {}
    valid_speakers = re.compile(r"(?:NARRATION|character[1-9][0-9]*)\Z")
    if any(
        index < 0
        or index >= global_index
        or index in selected_indices
        or not valid_speakers.fullmatch(speaker)
        for index, speaker in assignments.items()
    ):
        raise DatasetError("speaker prompt anchors must be valid earlier assignments")
    first_anchor_by_speaker: dict[str, int] = {}
    for index, speaker in sorted(assignments.items()):
        first_anchor_by_speaker.setdefault(speaker, index)
    anchors = {index: speaker for speaker, index in first_anchor_by_speaker.items()}
    selected_set = set(selected_indices)
    for page_number, page in enumerate(pages, 1):
        page_lines.append(f"Page {page_number}:")
        page_indices = [
            index
            for index, (line_page_number, _) in enumerate(flat_lines)
            if line_page_number == page_number
        ]
        visible_indices = [
            index for index in page_indices if index in selected_set or index in anchors
        ]
        if not visible_indices:
            page_lines.append("(no OCR candidates)")
        for line_index in visible_indices:
            _, line = flat_lines[line_index]
            x, y, width, height = line.box
            label = (
                f"Known speaker anchor {line_index} ({anchors[line_index]})"
                if line_index in anchors
                else f"Candidate line_index {line_index}"
            )
            page_lines.append(
                f"{label}: box=[{x},{y},{width},{height}] "
                f"ocr_text={json.dumps(line.text, ensure_ascii=False)}"
            )
    prompt = (
        "Assign speakers only to the candidate OCR lines listed below in this one manga sequence. "
        "The three attached images are pages 1, 2, and 3 in order. Each OCR candidate "
        "has one global line_index and a page-relative pixel box. Previously assigned "
        "lines are included as visual identity anchors; reuse their speaker label only "
        "when the same character is visible. Use the images, "
        "balloon tails, visual continuity, and the ordered dialogue only to identify "
        "the likely speaker. Keep one stable label for the same character throughout "
        "all pages: use character1, character2, etc., assigning a label only when "
        "reasonably confident. Do not create labels from names mentioned in dialogue. "
        "Use NARRATION only when the candidate is clearly narration rather than "
        "dialogue; otherwise use UNKNOWN whenever uncertain. Return exactly "
        f"{len(selected_indices)} decisions, one for each candidate line_index "
        "shown. Do not assign the known speaker anchor lines. Do not change, quote, rewrite, add, "
        "delete, or reorder any OCR text. Return only a JSON object with an "
        '"assignments" array; each array item must contain integer "line_index" '
        'and string "speaker" fields. Use only candidate indices '
        f"{selected_indices}. "
        "Allowed speaker values are UNKNOWN, NARRATION, or character followed by a "
        "positive integer.\n\n"
        + "\n".join(page_lines)
        + "\n\nImage page order: <image> page 1 <image> page 2 <image> page 3"
    )
    return prompt, line_locations


def run_speaker_pilot(
    root: Path,
    sequence_id: str,
    *,
    model_path: Path,
    artifact_dir: Path,
    loaded_model: tuple[Any, Any] | None = None,
    force: bool = False,
    candidate_pages: list[list[OCRLine]] | None = None,
    expected_pages: list[list[dict[str, str]]] | None = None,
    allow_validation: bool = False,
) -> Path:
    sequence = load_sequence(root, sequence_id, include_labels=False)
    if sequence.split != "development" or (
        sequence.split == "validation" and not allow_validation
    ):
        raise DatasetError("speaker pilot sequence must belong to development or explicit validation")
    if not model_path.is_dir() or not (model_path / "model.safetensors").is_file():
        raise DatasetError(
            f"local MLX-VLM model directory is incomplete or unavailable: {model_path}"
        )
    if (model_path / "model.safetensors").stat().st_size < 100_000_000:
        raise DatasetError(f"local MLX-VLM weight file is incomplete: {model_path}")

    if candidate_pages is None:
        if expected_pages is not None:
            raise DatasetError("expected_pages require supplied OCR candidate_pages")
        baseline_pages = run_tesseract_sequence(
            root,
            sequence_id,
            artifact_dir=artifact_dir,
            psm=11,
            min_confidence=70,
            preprocessing="raw",
            reading_order="rtl_geometry",
            force=force,
        )
        line_pages = [
            _run_tesseract_ocr_lines(
                image.path,
                psm=11,
                min_confidence=70,
                reading_order="rtl_geometry",
            )
            for image in sequence.images
        ]
    else:
        if len(candidate_pages) != PAGES_PER_SEQUENCE:
            raise DatasetError("supplied OCR candidate pages must contain exactly three pages")
        if expected_pages is None or len(expected_pages) != PAGES_PER_SEQUENCE:
            raise DatasetError("supplied OCR candidate pages require a three-page frozen output")
        line_pages = candidate_pages
        baseline_pages = expected_pages
    candidate_texts = [
        [{"speaker": "UNKNOWN", "text": line.text} for line in page]
        for page in line_pages
    ]
    if candidate_texts != baseline_pages:
        raise DatasetError(
            "OCR line-box extraction does not exactly match the frozen selected predictions; "
            "speaker inference was stopped to protect OCR text/order"
        )

    _, line_locations = _speaker_prompt(line_pages, candidate_indices=[])
    if len(line_locations) != sum(map(len, line_pages)):
        raise DatasetError("speaker prompt line-index construction is inconsistent")
    input_hash = hashlib.sha256(
        json.dumps(
            [
                [
                    {"text": line.text, "box": line.box, "confidence": line.confidence}
                    for line in page
                ]
                for page in line_pages
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    sequence_key = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()
    model_key = hashlib.sha256(str(model_path.resolve()).encode("utf-8")).hexdigest()[:16]
    cache_dir = (
        artifact_dir
        / "intermediate"
        / sequence_key
        / f"speaker_only_v5_{model_key}"
    )
    assignment_cache = cache_dir / "assignments.json"
    if assignment_cache.is_file() and not force:
        cache = _read_json(assignment_cache)
        if not isinstance(cache, dict) or cache.get("input_sha256") != input_hash:
            raise DatasetError(
                f"speaker cache input no longer matches OCR candidates: {assignment_cache}; "
                "rerun with --force"
            )
        cached_speakers = cache.get("speakers")
        speaker_count = sum(map(len, line_pages))
        valid_speakers = re.compile(r"(?:UNKNOWN|NARRATION|character[1-9][0-9]*)\Z")
        if (
            not isinstance(cached_speakers, list)
            or len(cached_speakers) != speaker_count
            or any(
                not isinstance(speaker, str) or not valid_speakers.fullmatch(speaker)
                for speaker in cached_speakers
            )
        ):
            raise DatasetError(f"invalid resumable speaker cache: {assignment_cache}")
        speakers = cached_speakers
        raw_response_path = cache_dir / "response.txt"
        raw_response = raw_response_path.read_text(encoding="utf-8") if raw_response_path.is_file() else ""
    else:
        if loaded_model is None:
            try:
                from mlx_vlm import load
            except ImportError as error:
                raise DatasetError(
                    "MLX-VLM is not installed; the speaker-only backend requires local MLX-VLM."
                ) from error
            try:
                model, processor = load(str(model_path.resolve()))
            except (OSError, ValueError, RuntimeError, TypeError, MemoryError) as error:
                raise DatasetError(
                    f"could not load local MLX-VLM model {model_path}: "
                    f"{type(error).__name__}: {error}"
                ) from error
        else:
            model, processor = loaded_model
        from mlx_vlm import generate

        speakers = ["UNKNOWN"] * len(line_locations)
        batch_size = 10
        batches: list[dict[str, Any]] = []
        for batch_start in range(0, len(line_locations), batch_size):
            batch_indices = list(
                range(batch_start, min(batch_start + batch_size, len(line_locations)))
            )
            prompt, _ = _speaker_prompt(
                line_pages,
                candidate_indices=batch_indices,
                known_assignments={
                    index: speaker
                    for index, speaker in enumerate(speakers)
                    if speaker != "UNKNOWN"
                },
            )
            try:
                raw_response = generate(
                    model,
                    processor,
                    prompt,
                    image=[str(image.path) for image in sequence.images],
                    max_tokens=min(1024, max(384, len(batch_indices) * 64)),
                    temperature=0.0,
                    repetition_penalty=1.05,
                    repetition_context_size=64,
                    verbose=False,
                ).text
            except (OSError, ValueError, RuntimeError, TypeError, MemoryError) as error:
                raise DatasetError(
                    f"local speaker-only VLM inference failed for {sequence_id}, "
                    f"lines {batch_indices[0]}-{batch_indices[-1]}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            parsed_speakers = parse_speaker_assignments(raw_response, len(line_locations))
            for index in batch_indices:
                speakers[index] = parsed_speakers[index]
            batches.append(
                {
                    "line_indices": batch_indices,
                    "response": raw_response,
                }
            )
        raw_response = json.dumps(batches, ensure_ascii=False, indent=2)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(cache_dir / "response.txt", raw_response)
        _atomic_write(
            assignment_cache,
            json.dumps(
                {"input_sha256": input_hash, "speakers": speakers},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    pages = apply_speaker_assignments(line_pages, speakers)
    if [[line["text"] for line in page] for page in pages] != [
        [line["text"] for line in page] for page in baseline_pages
    ]:
        raise DatasetError("speaker assignment changed OCR text or line order")
    output_path = artifact_dir / "pilot" / f"{sequence_id}_speaker_only_v5.jsonl"
    _write_jsonl(output_path, [{"sequence_id": sequence_id, "pages": pages}])
    validate_submission(output_path, [sequence_id])
    counts = {
        label: sum(speaker == label for speaker in speakers)
        for label in sorted(set(speakers))
    }
    print(
        f"Speaker assignments: {len(speakers)} OCR lines; "
        f"labels={counts}; local_model={model_path}"
    )
    print(f"Raw model response: {cache_dir / 'response.txt'}")
    return output_path


def _default_local_speaker_model() -> Path:
    model_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--mlx-community--Qwen2-VL-2B-Instruct-4bit"
        / "snapshots"
    )
    complete = [
        snapshot
        for snapshot in model_root.iterdir()
        if snapshot.is_dir()
        and (snapshot / "model.safetensors").is_file()
        and (snapshot / "model.safetensors").stat().st_size >= 100_000_000
    ] if model_root.is_dir() else []
    if not complete:
        raise DatasetError(
            "No complete local Qwen2-VL-2B-Instruct-4bit snapshot was found. "
            "The speaker pilot requires local MLX-VLM weights and will not download them."
        )
    return sorted(complete)[-1]


def run_local_pilot(
    root: Path,
    sequence_id: str,
    *,
    model_id: str,
    artifact_dir: Path,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    force: bool = False,
) -> Path:
    sequence = load_sequence(root, sequence_id, include_labels=False)
    if sequence.split != "development":
        raise DatasetError("pilot sequence must belong to the development split")

    sequence_key = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()
    model_key = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:16]
    intermediate_dir = artifact_dir / "intermediate" / sequence_key / model_key
    pages: list[list[dict[str, str]]] = []
    pending_pages: list[int] = []
    for page_index in range(PAGES_PER_SEQUENCE):
        cache_path = intermediate_dir / f"page_{page_index + 1:02d}.json"
        if cache_path.is_file() and not force:
            cached = _read_json(cache_path)
            pages.append(_page_cache(cached, f"cached page {page_index + 1}"))
        else:
            pages.append([])
            pending_pages.append(page_index)

    if pending_pages:
        try:
            from mlx_vlm import generate, load
        except ImportError as error:
            raise DatasetError(
                "MLX-VLM is not installed in this Python environment; install it with "
                "`python -m pip install -U mlx-vlm`."
            ) from error
        model, processor = load(model_id)
        for page_index in pending_pages:
            result = generate(
                model,
                processor,
                _page_prompt(page_index + 1),
                image=str(sequence.images[page_index].path),
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            raw_path = intermediate_dir / f"page_{page_index + 1:02d}.response.txt"
            _atomic_write(raw_path, result.text)
            page = _parse_model_page(result.text, f"model output for page {page_index + 1}")
            pages[page_index] = page
            _atomic_write(
                intermediate_dir / f"page_{page_index + 1:02d}.json",
                json.dumps(page, ensure_ascii=False, indent=2) + "\n",
            )

    prediction = {"sequence_id": sequence_id, "pages": pages}
    output_path = artifact_dir / "pilot" / f"{sequence_key}.jsonl"
    _atomic_write(output_path, json.dumps(prediction, ensure_ascii=False) + "\n")
    validate_submission(
        output_path,
        [sequence_id],
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate dataset and template")
    validate_parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="dataset root containing sequences.json",
    )
    pilot_parser = subparsers.add_parser("pilot", help="run local inference on one sequence")
    pilot_parser.add_argument("--sequence-id", required=True)
    pilot_parser.add_argument("--model", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    pilot_parser.add_argument("--max-tokens", type=int, default=1024)
    pilot_parser.add_argument("--temperature", type=float, default=0.0)
    pilot_parser.add_argument("--force", action="store_true", help="regenerate cached pages")
    pilot_parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="dataset root containing sequences.json",
    )
    pilot_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="directory for resumable model outputs",
    )
    ocr_parser = subparsers.add_parser("ocr-pilot", help="run local Tesseract on one sequence")
    ocr_parser.add_argument("--sequence-id", required=True)
    ocr_parser.add_argument("--psm", type=int, choices=(6, 11), default=6)
    ocr_parser.add_argument("--min-confidence", type=int, default=0)
    ocr_parser.add_argument("--min-lines", type=int, default=1)
    ocr_parser.add_argument("--force", action="store_true")
    ocr_parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="dataset root containing sequences.json",
    )
    ocr_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="directory for resumable OCR outputs",
    )
    comic_parser = subparsers.add_parser(
        "comic-detector-pilot",
        help="run the local comic-text detector and Tesseract on one development sequence",
    )
    comic_parser.add_argument("--sequence-id", required=True)
    comic_parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "models"
        / "comictextdetector.pt.onnx",
    )
    comic_parser.add_argument("--force", action="store_true")
    comic_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    comic_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    comic_experiment_parser = subparsers.add_parser(
        "evaluate-comic-detector",
        help="compare comic-detector OCR with the frozen validation RTL baseline",
    )
    comic_experiment_parser.add_argument("--seed", type=int, default=42)
    comic_experiment_parser.add_argument("--validation-fraction", type=float, default=0.2)
    comic_experiment_parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "models"
        / "comictextdetector.pt.onnx",
    )
    comic_experiment_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    comic_experiment_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    speaker_parser = subparsers.add_parser(
        "speaker-pilot",
        help="assign speakers to fixed local OCR lines for one development sequence",
    )
    speaker_parser.add_argument("--sequence-id", required=True)
    speaker_parser.add_argument("--model", type=Path)
    speaker_parser.add_argument("--force", action="store_true")
    speaker_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    speaker_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    speaker_experiment_parser = subparsers.add_parser(
        "evaluate-speakers",
        help="compare speaker-only assignments with UNKNOWN on the frozen split",
    )
    speaker_experiment_parser.add_argument("--seed", type=int, default=42)
    speaker_experiment_parser.add_argument("--validation-fraction", type=float, default=0.2)
    speaker_experiment_parser.add_argument("--model", type=Path)
    speaker_experiment_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    speaker_experiment_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    weak_filter_parser = subparsers.add_parser(
        "evaluate-weak-filter",
        help="train a training-only weak-label OCR line filter and score it on the frozen split",
    )
    weak_filter_parser.add_argument("--seed", type=int, default=42)
    weak_filter_parser.add_argument("--validation-fraction", type=float, default=0.2)
    weak_filter_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    weak_filter_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    calibrate_filter_parser = subparsers.add_parser(
        "calibrate-weak-filter",
        help="select a weak OCR filter threshold by grouped training-only OOF, then score fixed validation",
    )
    calibrate_filter_parser.add_argument("--seed", type=int, default=42)
    calibrate_filter_parser.add_argument("--validation-fraction", type=float, default=0.2)
    calibrate_filter_parser.add_argument("--folds", type=int, default=4)
    calibrate_filter_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    calibrate_filter_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    final_filter_parser = subparsers.add_parser(
        "fit-final-filter",
        help="fit the OOF-selected weak OCR filter on all labeled development sequences",
    )
    final_filter_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    final_filter_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    final_filter_parser.add_argument(
        "--output-dir",
        type=Path,
        help="write a fresh final filter artifact set to this new directory",
    )
    experiment_parser = subparsers.add_parser(
        "evaluate", help="compare Tesseract configurations on a sequence-level holdout"
    )
    experiment_parser.add_argument("--seed", type=int, default=42)
    experiment_parser.add_argument("--validation-fraction", type=float, default=0.2)
    experiment_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    experiment_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    submission_parser = subparsers.add_parser(
        "generate-test", help="run local OCR on test images and validate the submission"
    )
    submission_parser.add_argument("--psm", type=int, choices=(6, 11), default=6)
    submission_parser.add_argument("--min-confidence", type=int, default=70)
    submission_parser.add_argument("--min-lines", type=int, default=1)
    submission_parser.add_argument(
        "--reading-order",
        choices=("tesseract", "rtl_geometry"),
        default="rtl_geometry",
    )
    submission_parser.add_argument(
        "--weak-filter-model",
        type=Path,
        help="optional locally fitted weak OCR-line classifier JSON",
    )
    submission_parser.set_defaults(psm=11)
    submission_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    submission_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    submission_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "submission.jsonl",
    )
    submission_validator = subparsers.add_parser(
        "validate-submission", help="validate a submission against the supplied template"
    )
    submission_validator.add_argument("submission", type=Path)
    submission_validator.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    score_parser = subparsers.add_parser(
        "score-development",
        help="score predictions for labeled development sequence IDs only",
    )
    score_parser.add_argument("predictions", type=Path)
    score_parser.add_argument("--name", default="development_predictions")
    score_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    score_parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )

    args = parser.parse_args()
    try:
        if args.command == "validate":
            counts = validate_dataset(args.root)
            print("Dataset validation passed.")
            for name, count in counts.items():
                print(f"{name}: {count}")
        elif args.command == "pilot":
            output_path = run_local_pilot(
                args.root,
                args.sequence_id,
                model_id=args.model,
                artifact_dir=args.artifacts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                force=args.force,
            )
            print(f"Pilot prediction written to {output_path}")
        elif args.command == "ocr-pilot":
            sequence = load_sequence(args.root, args.sequence_id, include_labels=False)
            if sequence.split != "development":
                raise DatasetError("pilot sequence must belong to the development split")
            pages = run_tesseract_sequence(
                args.root,
                args.sequence_id,
                artifact_dir=args.artifacts,
                psm=args.psm,
                min_confidence=args.min_confidence,
                min_lines=args.min_lines,
                force=args.force,
            )
            output_path = args.artifacts / "pilot" / f"{args.sequence_id}_tesseract_psm{args.psm}.jsonl"
            _write_jsonl(output_path, [{"sequence_id": args.sequence_id, "pages": pages}])
            validate_submission(output_path, [args.sequence_id])
            print(f"Local Tesseract OCR pilot written to {output_path}")
            print(f"Recognized entries by page: {[len(page) for page in pages]}")
        elif args.command == "comic-detector-pilot":
            output_path = run_comic_text_sequence(
                args.root,
                args.sequence_id,
                artifact_dir=args.artifacts,
                model_path=args.model,
                force=args.force,
            )
            print(f"Local comic-detector OCR pilot written to {output_path}")
            print(
                "Visual overlays: "
                f"{args.artifacts / 'diagnostics' / 'comic_text_detector'}"
            )
        elif args.command == "evaluate-comic-detector":
            run_comic_detector_experiment(
                args.root,
                artifact_dir=args.artifacts,
                model_path=args.model,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
        elif args.command == "speaker-pilot":
            model_path = args.model or _default_local_speaker_model()
            output_path = run_speaker_pilot(
                args.root,
                args.sequence_id,
                model_path=model_path,
                artifact_dir=args.artifacts,
                force=args.force,
            )
            print(f"Local speaker-only prediction written to {output_path}")
        elif args.command == "evaluate-speakers":
            model_path = args.model or _default_local_speaker_model()
            run_speaker_validation_experiment(
                args.root,
                artifact_dir=args.artifacts,
                model_path=model_path,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
        elif args.command == "evaluate-weak-filter":
            from weak_text_filter import run_weak_filter_experiment

            run_weak_filter_experiment(
                args.root,
                artifact_dir=args.artifacts,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
        elif args.command == "calibrate-weak-filter":
            from weak_text_filter import run_grouped_cv_and_final_validation

            run_grouped_cv_and_final_validation(
                args.root,
                artifact_dir=args.artifacts,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
                folds_count=args.folds,
            )
        elif args.command == "fit-final-filter":
            from weak_text_filter import fit_final_development_filter

            fit_final_development_filter(
                args.root,
                artifact_dir=args.artifacts,
                output_dir=args.output_dir,
            )
        elif args.command == "evaluate":
            run_validation_experiment(
                args.root,
                artifact_dir=args.artifacts,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
        elif args.command == "generate-test":
            generate_test_submission(
                args.root,
                artifact_dir=args.artifacts,
                output_path=args.output,
                psm=args.psm,
                min_confidence=args.min_confidence,
                min_lines=args.min_lines,
                reading_order=args.reading_order,
                weak_filter_model=args.weak_filter_model,
            )
        else:
            if args.command == "validate-submission":
                root = args.root.resolve()
                template = _read_jsonl(root / "sample_submission.jsonl")
                records = validate_submission(
                    args.submission,
                    [record["sequence_id"] for record in template],
                )
                print(f"Submission validation passed: {len(records)} sequences")
            else:
                report = score_development_predictions(
                    args.root,
                    artifact_dir=args.artifacts,
                    prediction_path=args.predictions,
                    name=args.name,
                )
                print(f"Official development score saved for {len(report['sequences'])} sequences")
    except DatasetError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
