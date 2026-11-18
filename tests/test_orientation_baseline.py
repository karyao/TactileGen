"""Validate archived evidence without loading YOLO/OCR or enforcing old routing rules."""

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from src.contract_io import load_detection_frame


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/orientation"
CASES = json.loads((FIXTURES / "cases.json").read_text())
CAPTURE = json.loads((FIXTURES / "capture.json").read_text())
OBSERVATIONS = {case["id"]: case for case in CAPTURE["cases"]}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_archived_prediction_matches_its_source_image(case):
    observation = OBSERVATIONS[case["id"]]
    image_path = ROOT / case["image"]
    frame_path = FIXTURES / f'{case["id"]}.detections.json'
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == observation["image_sha256"]
    assert hashlib.sha256(frame_path.read_bytes()).hexdigest() == observation["detections_sha256"]
    frame = load_detection_frame(frame_path)  # Also validates geometry, roles and confidences.
    with Image.open(image_path) as image:
        assert image.size == (frame.width, frame.height) == tuple(observation["image_size"])
    assert len({d.id for d in frame.detections}) == len(frame.detections)
    assert observation["cli"]["raw_frame_equal"]


def test_corpus_contains_real_mixed_evidence_and_both_clean_orientations():
    assert len(OBSERVATIONS) == len(CASES)
    frame = load_detection_frame(FIXTURES / "mixed_predictions.detections.json")
    assert {d.role for d in frame.detections if d.confidence >= 0.30
            and d.role.endswith("_bar")} == {"vertical_bar", "horizontal_bar"}
    assert {case["expected_orientation"] for case in CASES
            if case["id"].startswith("clean_")} == {"vertical", "horizontal"}


@pytest.mark.parametrize("name", ["clean_vertical", "clean_horizontal"])
def test_archived_clean_results_agree_with_authored_ground_truth(name):
    expected = next(case for case in CASES if case["id"] == name)
    observed = OBSERVATIONS[name]
    for spec in (observed["app"]["spec"], observed["cli"]["spec"]):
        assert spec["orientation"] == expected["expected_orientation"]
        assert spec["categories"] == expected["expected_categories"]
        assert len(spec["series"]) == 1
        assert spec["series"][0]["values"] == pytest.approx(expected["expected_values"], abs=1)
    assert observed["app"]["completeness"]["status"] == "ready"
    assert observed["cli"]["exit_code"] == 0
