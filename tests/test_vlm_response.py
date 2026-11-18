"""Offline VLM response adaptation and contract handoff tests."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.baseline.vlm_response import parse_chart_response
from src.contract_io import load_bar_chart, parse_bar_chart, write_bar_chart
from src.contracts import HorizontalBarChart, VerticalBarChart


FIXTURES = Path(__file__).parent / "fixtures" / "vlm"
TITLE_DEFAULTS = {"title": "Untitled chart", "x_axis_title": "X axis", "y_axis_title": "Y axis"}


class VLMResponseTest(unittest.TestCase):
    def setUp(self):
        self.raw = (FIXTURES / "vertical_response.json").read_text(encoding="utf-8")
        self.data = json.loads(self.raw)

    def test_response_fixtures_and_round_trips(self):
        cases = (
            ("vertical_response.json", VerticalBarChart, (), ("2024", "2025"), "Year", "Sales"),
            ("horizontal_response.json", HorizontalBarChart, ("title", "y_axis_title"),
             ("North", "South"), "Sales", "Y axis"),
        )
        for filename, kind, placeholders, categories, x_title, y_title in cases:
            raw = (FIXTURES / filename).read_text(encoding="utf-8")
            for response in (raw, f" \n```json\n{raw}\n```\t\n"):
                with self.subTest(filename=filename, response=response):
                    result = parse_chart_response(response, image=Path("charts/example.png"))
                    self.assertIs(type(result.chart), kind)
                    self.assertEqual(result.placeholder_fields, placeholders)
                    self.assertEqual(result.chart.source_key, "vlm/example.png")
                    self.assertEqual(result.chart.categories, categories)
                    self.assertEqual(result.chart.x_axis_title, x_title)
                    self.assertEqual(result.chart.y_axis_title, y_title)
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "chart.json"
                        write_bar_chart(path, result.chart)
                        self.assertNotIn("placeholder_fields", json.loads(path.read_text()))
                        self.assertEqual(load_bar_chart(path), result.chart)

    def test_source_identity_is_always_caller_owned(self):
        for model_key in ("model/chosen.png", None, [], 1):
            response = json.dumps({**self.data, "source_key": model_key})
            for supplied, expected in ((None, "vlm/input.png"), ("eval/custom.png", "eval/custom.png")):
                with self.subTest(model_key=model_key, supplied=supplied):
                    result = parse_chart_response(response, image="absent/input.png", source_key=supplied)
                    self.assertEqual(result.chart.source_key, expected)
        for key in ("", "image.png", "../bad.png", "vlm/../bad.png", False):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "source_key"):
                parse_chart_response(self.raw, image="input.png", source_key=key)

    def test_each_title_placeholder_and_invalid_types(self):
        for field, default in TITLE_DEFAULTS.items():
            for value in (None, "", " \t\n", "missing"):
                with self.subTest(field=field, value=value):
                    data = {**self.data, field: value}
                    if value == "missing":
                        del data[field]
                    result = parse_chart_response(json.dumps(data), image="input.png")
                    self.assertEqual(getattr(result.chart, field), default)
                    self.assertEqual(result.placeholder_fields, (field,))
            for value in (False, 0, [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    parse_chart_response(json.dumps({**self.data, field: value}), image="input.png")

    def test_all_placeholders_have_stable_order_and_preserve_nonblank_titles(self):
        data = {key: value for key, value in self.data.items() if key not in TITLE_DEFAULTS}
        result = parse_chart_response(json.dumps(data), image="input.png")
        self.assertEqual(result.placeholder_fields, tuple(TITLE_DEFAULTS))
        for field, default in TITLE_DEFAULTS.items():
            self.assertEqual(getattr(result.chart, field), default)
        data.update({field: "  Printed title  " for field in TITLE_DEFAULTS})
        result = parse_chart_response(json.dumps(data), image="input.png")
        self.assertEqual(result.placeholder_fields, ())
        for field in TITLE_DEFAULTS:
            self.assertEqual(getattr(result.chart, field), "  Printed title  ")

    def test_crlf_fence(self):
        response = "```json\r\n" + self.raw + "\r\n```"
        self.assertEqual(parse_chart_response(response, image="input.png").placeholder_fields, ())

    def test_invalid_json_and_wrappers(self):
        for response in (
            "", "Here is the chart: " + self.raw, self.raw + " Done.",
            self.raw + self.raw, self.raw[:-3], "{invalid}",
            "```json\n" + self.raw, self.raw + "\n```",
            "```\n" + self.raw + "\n```", "```python\n" + self.raw + "\n```",
            "```json\n" + self.raw + "\n```\n```json\n" + self.raw + "\n```",
            "Prose\n```json\n" + self.raw + "\n```",
            "```json\n" + self.raw + "\n```\nProse",
            "[]", "null", "true", "42", '"chart"',
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_chart_response(response, image="input.png")
        for response in (None, {}, b"{}"):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "must be text"):
                parse_chart_response(response, image="input.png")

    def test_duplicate_keys_and_nonstandard_numbers(self):
        duplicate_chart = self.raw.replace('"title": "Annual totals"', '"title": "Annual totals", "title": "Other"')
        duplicate_series = self.raw.replace('"name": "North"', '"name": "North", "name": "North"')
        for response in (duplicate_chart, duplicate_series):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                parse_chart_response(response, image="input.png")
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            response = self.raw.replace("[4, 6]", f"[{token}, 6]")
            with self.subTest(token=token), self.assertRaises(ValueError):
                parse_chart_response(response, image="input.png")

    def test_model_extraction_error(self):
        for response in ('{"error": "unreadable or unsupported chart"}', '{"error": null}'):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "VLM extraction failed"):
                parse_chart_response(response, image="input.png")

    def test_contract_errors_are_not_repaired(self):
        invalid = [
            {**self.data, "chart_type": "line"}, {**self.data, "unknown": True},
            {**self.data, "categories": []}, {**self.data, "categories": "2024"},
            {**self.data, "series": [{"name": None, "values": [1]}]},
            {**self.data, "series": [{"name": None, "values": [True, 2]}]},
            {**self.data, "series": [{"name": None, "values": [None, 2]}]},
            {**self.data, "series": [{"name": None, "values": [1, 2], "extra": 0}]},
        ]
        for field in ("categories", "series"):
            invalid.append({key: value for key, value in self.data.items() if key != field})
        invalid.append({**self.data, "series": [{"name": None}]})
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_chart_response(json.dumps(data), image="input.png")

    def test_prepared_payload_delegates_and_preserves_optional_fields(self):
        self.data["omit_legend"] = True
        self.data["series"][0]["rescued"] = [False, True]
        self.data["series"][1] = {"name": "South", "values": [], "has_bars": False}
        expected = {**self.data, "schema_version": 1, "source_key": "vlm/input.png"}
        with patch("src.baseline.vlm_response.parse_bar_chart", wraps=parse_bar_chart) as parser:
            result = parse_chart_response(json.dumps(self.data), image="input.png")
        parser.assert_called_once_with(expected)
        self.assertEqual(result.chart, parse_bar_chart(expected))

    def test_schema_version_is_supplied_internally(self):
        expected = parse_chart_response(self.raw, image="input.png")
        for version in (1, 2, None, "model-chosen", False):
            with self.subTest(version=version):
                response = json.dumps({**self.data, "schema_version": version})
                self.assertEqual(parse_chart_response(response, image="input.png"), expected)

    def test_import_without_third_party_dependencies(self):
        subprocess.run(
            [sys.executable, "-S", "-c", "from src.baseline.vlm_response import parse_chart_response"],
            cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True,
        )
