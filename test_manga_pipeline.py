import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from manga_pipeline import (
    DatasetError,
    OCRLine,
    PNG_SIGNATURE,
    TextRegion,
    _speaker_prompt,
    _decode_comic_text_regions,
    _deduplicate_text_regions,
    _ocr_text_to_page,
    _empty_page_metrics,
    _parse_model_page,
    _parse_tesseract_tsv,
    _sequence_level_split,
    apply_speaker_assignments,
    load_sequence,
    order_ocr_lines,
    parse_speaker_assignments,
    preprocess_image,
    validate_dataset,
    validate_submission,
)
from weak_text_filter import (
    _training_labels,
    aligned_candidate_coverages,
    development_sequence_ids,
    filter_candidate_pages,
    grouped_sequence_folds,
    line_features,
    weak_candidate_label,
)


class MangaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.development_id = "seq_development"
        self.test_id = "seq_test"
        manifest = []
        for split, sequence_id in (
            ("development", self.development_id),
            ("test", self.test_id),
        ):
            image_dir = self.root / split / "images" / sequence_id
            image_dir.mkdir(parents=True)
            names = ("03.png", "01.png", "02.png")
            for name in names:
                (image_dir / name).write_bytes(PNG_SIGNATURE + b"fixture")
            manifest.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "images": [
                        f"{split}/images/{sequence_id}/{name}" for name in names
                    ],
                }
            )
        (self.root / "sequences.json").write_text(json.dumps(manifest), encoding="utf-8")
        label = {
            "sequence_id": self.development_id,
            "pages": [
                [{"speaker": "character1", "text": "page one"}],
                [],
                [{"speaker": "UNKNOWN", "text": "page three"}],
            ],
        }
        (self.root / "development" / "labels.jsonl").write_text(
            json.dumps(label) + "\n", encoding="utf-8"
        )
        template = {
            "sequence_id": self.test_id,
            "pages": [[], [], []],
        }
        (self.root / "sample_submission.jsonl").write_text(
            json.dumps(template) + "\n", encoding="utf-8"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_validate_dataset_counts_records(self):
        self.assertEqual(
            validate_dataset(self.root),
            {
                "development_sequences": 1,
                "test_sequences": 1,
                "image_files": 6,
                "development_text_entries": 2,
            },
        )

    def test_load_sequence_preserves_manifest_image_order(self):
        sequence = load_sequence(self.root, self.development_id)
        self.assertEqual(
            [image.path.name for image in sequence.images],
            ["03.png", "01.png", "02.png"],
        )
        self.assertEqual([len(page) for page in sequence.pages or ()], [1, 0, 1])
        self.assertTrue(all(image.content.startswith(PNG_SIGNATURE) for image in sequence.images))

    def test_inference_loader_can_skip_development_labels(self):
        sequence = load_sequence(self.root, self.development_id, include_labels=False)
        self.assertIsNone(sequence.pages)

    def test_sequence_split_stratifies_blank_pages_without_splitting_pages(self):
        ids = ["empty", "filled1", "filled2", "filled3", "filled4"]
        labels = {
            "empty": ((), (), ()),
            "filled1": (({"speaker": "c1", "text": "a"},), (), ()),
            "filled2": (({"speaker": "c1", "text": "a"},),) * 3,
            "filled3": (({"speaker": "c1", "text": "a"},),) * 3,
            "filled4": (({"speaker": "c1", "text": "a"},),) * 3,
        }
        training, validation = _sequence_level_split(
            ids,
            labels,
            seed=42,
            validation_fraction=0.4,
        )
        self.assertEqual(len(validation), 2)
        self.assertTrue(any(any(not page for page in labels[item]) for item in validation))
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(set(training) | set(validation), set(ids))

    def test_empty_page_metrics_measure_gold_empty_pages_only(self):
        references = [{"pages": [[], [{"speaker": "c1", "text": "hi"}], []]}]
        predictions = [{"pages": [[{"speaker": "UNKNOWN", "text": "noise"}], [], []]}]
        self.assertEqual(
            _empty_page_metrics(references, predictions),
            {
                "gold_empty_pages": 2,
                "correctly_empty_gold_pages": 1,
                "false_positive_gold_empty_pages": 1,
            },
        )

    def test_submission_requires_template_id_order(self):
        path = self.root / "sample_submission.jsonl"
        record = {
            "sequence_id": self.test_id,
            "pages": [[], [], []],
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        self.assertEqual(validate_submission(path, [self.test_id])[0]["sequence_id"], self.test_id)
        with self.assertRaisesRegex(DatasetError, "template order"):
            validate_submission(path, [self.development_id])

    def test_submission_rejects_non_string_speakers(self):
        path = self.root / "invalid.jsonl"
        record = {
            "sequence_id": self.test_id,
            "pages": [[{"speaker": 1, "text": "words"}], [], []],
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "speaker must be"):
            validate_submission(path, [self.test_id])

    def test_ocr_lines_are_not_presented_as_attributed_speakers(self):
        self.assertEqual(
            _ocr_text_to_page(" first line \n\nsecond line\n"),
            [
                {"speaker": "UNKNOWN", "text": "first line"},
                {"speaker": "UNKNOWN", "text": "second line"},
            ],
        )

    def test_tesseract_confidence_filter_groups_words_into_lines(self):
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\t"
            "width\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\treadable\n"
            "5\t1\t1\t1\t1\t2\t11\t0\t10\t10\t70\tline\n"
            "5\t1\t1\t1\t2\t1\t0\t20\t10\t10\t20\tnoise\n"
        )
        self.assertEqual(
            _parse_tesseract_tsv(tsv, 50),
            [{"speaker": "UNKNOWN", "text": "readable line"}],
        )

    def test_grayscale_upscale_and_threshold_preprocessing_preserve_dimensions_ratio(self):
        source = Image.new("RGB", (20, 10), "white")
        self.assertEqual(preprocess_image(source, "raw").size, (20, 10))
        self.assertEqual(preprocess_image(source, "gray2x").size, (40, 20))
        thresholded = preprocess_image(source, "threshold2x")
        self.assertEqual(thresholded.size, (40, 20))
        self.assertEqual(thresholded.mode, "L")

    def test_rtl_geometric_sort_reads_right_column_first(self):
        lines = [
            OCRLine("left", 90, (10, 10, 30, 12)),
            OCRLine("right", 90, (100, 11, 30, 12)),
            OCRLine("lower", 90, (10, 80, 30, 12)),
        ]
        self.assertEqual(
            [line.text for line in order_ocr_lines(lines, "rtl_geometry")],
            ["right", "left", "lower"],
        )

    def test_detector_decoder_maps_regions_and_filters_low_confidence(self):
        import numpy as np

        line_map = np.zeros((100, 100), dtype=np.float32)
        line_map[20:28, 10:50] = 0.9
        line_map[60:64, 60:80] = 0.5
        regions = _decode_comic_text_regions(
            line_map,
            (100, 100),
            pad_x=0,
            pad_y=0,
            input_size=100,
        )
        self.assertEqual(len(regions), 1)
        xs = [point[0] for point in regions[0].polygon]
        ys = [point[1] for point in regions[0].polygon]
        self.assertLessEqual(min(xs), 10)
        self.assertGreaterEqual(max(xs), 49)
        self.assertLessEqual(min(ys), 20)
        self.assertGreaterEqual(max(ys), 27)
        self.assertGreater(regions[0].confidence, 0.8)

    def test_detector_removes_contained_overlapping_duplicate_regions(self):
        regions = [
            TextRegion(((0, 0), (100, 0), (100, 20), (0, 20)), 0.8),
            TextRegion(((2, 2), (98, 2), (98, 18), (2, 18)), 0.9),
            TextRegion(((120, 0), (160, 0), (160, 20), (120, 20)), 0.7),
        ]
        unique = _deduplicate_text_regions(regions)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0].confidence, 0.9)

    def test_speaker_assignments_map_only_valid_line_indices(self):
        self.assertEqual(
            parse_speaker_assignments(
                '{"assignments":[{"line_index":0,"speaker":"character1"},'
                '{"line_index":1,"speaker":"NARRATION"},'
                '{"line_index":2,"speaker":"UNKNOWN"}]}',
                4,
            ),
            ["character1", "NARRATION", "UNKNOWN", "UNKNOWN"],
        )
        self.assertEqual(
            parse_speaker_assignments(
                '[{"line_index":0,"speaker":"character1"},'
                '{"line_index":1,"speaker":"NARRATION"}]',
                2,
            ),
            ["character1", "NARRATION"],
        )

    def test_malformed_speaker_response_abstains_for_every_line(self):
        self.assertEqual(
            parse_speaker_assignments(
                '```json\n{"assignments":[{"line_index":0,"speaker":"character0"}',
                2,
            ),
            ["UNKNOWN", "UNKNOWN"],
        )
        self.assertEqual(
            parse_speaker_assignments(
                '{"assignments":[{"line_index":0,"speaker":"character1"},'
                '{"line_index":0,"speaker":"character2"}]}',
                1,
            ),
            ["UNKNOWN"],
        )
        self.assertEqual(
            parse_speaker_assignments(
                '[{"line_index":0,"speaker":"character1"},'
                '{"line_index":0,"speaker":"character2"}]',
                1,
            ),
            ["UNKNOWN"],
        )
        repeated_blocks = (
            '```json\n{"assignments":[{"line_index":0,"speaker":"character1"}]}```'
            '```json\n{"assignments":[{"line_index":0,"speaker":"character1"}]}```'
        )
        self.assertEqual(parse_speaker_assignments(repeated_blocks, 1), ["UNKNOWN"])

    def test_speaker_assignment_preserves_ocr_text_order_and_cross_page_identity(self):
        pages = [
            [OCRLine("Hello.", 90, (10, 20, 30, 10)), OCRLine("Who?", 90, (50, 20, 30, 10))],
            [OCRLine("Hello again.", 90, (12, 25, 40, 10))],
            [],
        ]
        assigned = apply_speaker_assignments(
            pages,
            ["character1", "character2", "character1"],
        )
        self.assertEqual(
            assigned,
            [
                [
                    {"speaker": "character1", "text": "Hello."},
                    {"speaker": "character2", "text": "Who?"},
                ],
                [{"speaker": "character1", "text": "Hello again."}],
                [],
            ],
        )

    def test_batched_speaker_prompt_uses_global_indices_and_prior_anchor(self):
        pages = [
            [OCRLine("Hello.", 90, (10, 20, 30, 10))],
            [OCRLine("Again?", 90, (50, 25, 30, 10))],
            [],
        ]
        prompt, locations = _speaker_prompt(
            pages,
            candidate_indices=[1],
            known_assignments={0: "character1"},
        )
        self.assertEqual(locations, [(1, 0), (2, 1)])
        self.assertIn("Known speaker anchor 0 (character1)", prompt)
        self.assertIn("Candidate line_index 1", prompt)
        self.assertIn("Return exactly 1 decisions", prompt)

    def test_weak_filter_features_use_prediction_time_text_and_page_geometry(self):
        features = line_features(
            "HELLO 12!",
            80,
            (10, 20, 40, 10),
            (100, 100),
        )
        self.assertEqual(set(features), {
            "confidence", "log_character_count", "log_token_count",
            "alphanumeric_ratio", "letter_ratio", "digit_ratio",
            "punctuation_ratio", "uppercase_ratio", "log_width_ratio",
            "log_height_ratio", "log_area_ratio", "log_aspect_ratio",
            "center_x", "center_y", "vertical_orientation",
        })
        self.assertEqual(features["confidence"], 0.8)
        self.assertEqual(features["center_x"], 0.3)
        self.assertEqual(features["center_y"], 0.25)
        self.assertEqual(features["vertical_orientation"], 0.0)
        self.assertEqual(features["uppercase_ratio"], 1.0)

    def test_weak_alignment_labels_positive_negative_and_ambiguous_separately(self):
        candidates = [
            OCRLine("We are here.", 90, (0, 0, 50, 12)),
            OCRLine("???", 90, (0, 20, 10, 10)),
            OCRLine("unmatched dialogue", 90, (0, 40, 60, 12)),
        ]
        coverage = aligned_candidate_coverages(
            [{"speaker": "character1", "text": "We are here."}],
            candidates,
        )
        self.assertGreaterEqual(coverage[0], 0.6)
        self.assertEqual(weak_candidate_label(candidates[0].text, coverage[0]), "positive")
        self.assertEqual(weak_candidate_label(candidates[1].text, coverage[1]), "negative")
        self.assertEqual(weak_candidate_label(candidates[2].text, coverage[2]), "ambiguous")

    def test_training_label_loader_excludes_nontraining_sequence_ids(self):
        labels_path = self.root / "development" / "labels.jsonl"
        labels_path.write_text(
            "\n".join(
                json.dumps(
                    {"sequence_id": sequence_id, "pages": [[], [], []]}
                )
                for sequence_id in ("train_sequence", "validation_sequence")
            ) + "\n",
            encoding="utf-8",
        )
        loaded = _training_labels(self.root, ["train_sequence"])
        self.assertEqual(list(loaded), ["train_sequence"])

    def test_final_filter_training_ids_include_development_only_in_manifest_order(self):
        self.assertEqual(
            development_sequence_ids(self.root),
            [self.development_id],
        )
        self.assertNotIn(self.test_id, development_sequence_ids(self.root))

    def test_grouped_cross_validation_is_reproducible_and_partitions_sequences(self):
        ids = [f"seq_{index}" for index in range(13)]
        first = grouped_sequence_folds(ids, 4, seed=42)
        self.assertEqual(first, grouped_sequence_folds(ids, 4, seed=42))
        self.assertEqual(set().union(*(set(fold) for fold in first)), set(ids))
        self.assertEqual(sum(map(len, first)), len(ids))
        self.assertTrue(all(fold for fold in first))
        self.assertEqual(len({sequence_id for fold in first for sequence_id in fold}), len(ids))

    def test_weak_filter_uses_only_features_and_preserves_empty_pages(self):
        model = {
            "feature_names": ["log_character_count"],
            "means": [0.0],
            "scales": [1.0],
            "coefficients": [1.0],
            "intercept": -1.0,
            "threshold": 0.5,
        }
        pages = [
            {
                "size": (100, 100),
                "lines": [
                    OCRLine("HELLO", 90, (10, 10, 30, 10)),
                    OCRLine("x", 90, (10, 20, 5, 10)),
                ],
            },
            {"size": (100, 100), "lines": []},
        ]
        self.assertEqual(
            filter_candidate_pages(pages, model),
            [[{"speaker": "UNKNOWN", "text": "HELLO"}], []],
        )

    def test_vlm_parser_accepts_a_single_json_fence_and_rejects_name_lists(self):
        page = _parse_model_page(
            'Page 1:\n```json\n[{"speaker":"UNKNOWN","text":"visible"}]\n```',
            "fixture",
        )
        self.assertEqual(page, [{"speaker": "UNKNOWN", "text": "visible"}])
        with self.assertRaisesRegex(DatasetError, "exactly speaker and text"):
            _parse_model_page('["character1", "character2"]', "fixture")


if __name__ == "__main__":
    unittest.main()
