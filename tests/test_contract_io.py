"""In-memory bar chart parsing and JSON boundary validation."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.contract_io import load_bar_chart, parse_bar_chart, write_bar_chart
from src.contracts import HorizontalBarChart, VerticalBarChart


FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


class ParseBarChartTest(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((FIXTURES / "vertical_bar_chart.json").read_text())

    def test_both_orientations_preserve_order_titles_and_input(self):
        for kind, expected in (("vbar_categorical", VerticalBarChart),
                               ("hbar_categorical", HorizontalBarChart)):
            with self.subTest(kind=kind):
                self.data["chart_type"] = kind
                original = deepcopy(self.data)
                chart = parse_bar_chart(self.data)
                self.assertIs(type(chart), expected)
                self.assertEqual(chart.categories, ("A", "B"))
                self.assertEqual(chart.x_axis_title, "X")
                self.assertEqual(chart.y_axis_title, "Y")
                self.assertEqual(chart.series[0].values, (4, 6))
                self.assertEqual(chart.series[1].values, (3, 5))
                self.assertTrue(chart.series[0].has_bars)
                self.assertEqual(chart.series[0].rescued, ())
                self.assertFalse(chart.omit_legend)
                self.assertEqual(self.data, original)

    def test_round_trip_with_optional_fields(self):
        for kind in ("vbar_categorical", "hbar_categorical"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                self.data["chart_type"] = kind
                self.data["omit_legend"] = True
                self.data["series"][0]["rescued"] = [False, True]
                self.data["series"][1] = {"name": "S", "values": [], "has_bars": False}
                chart = parse_bar_chart(self.data)
                path = Path(tmp) / "chart.json"
                write_bar_chart(path, chart)
                self.assertEqual(load_bar_chart(path), chart)
                self.assertEqual(parse_bar_chart(json.loads(path.read_text())), chart)

    def test_loader_delegates_to_parser(self):
        with patch("src.contract_io.parse_bar_chart") as parser:
            result = load_bar_chart(FIXTURES / "vertical_bar_chart.json")
        parser.assert_called_once_with(self.data)
        self.assertIs(result, parser.return_value)

    def test_required_fields(self):
        for field in self.data:
            with self.subTest(field=field):
                data = deepcopy(self.data)
                del data[field]
                with self.assertRaisesRegex(ValueError, field):
                    parse_bar_chart(data)
        for field in ("name", "values"):
            with self.subTest(series_field=field):
                data = deepcopy(self.data)
                del data["series"][0][field]
                with self.assertRaisesRegex(ValueError, field):
                    parse_bar_chart(data)

    def test_unknown_fields(self):
        for target in (self.data, self.data["series"][0]):
            target["unexpected"] = 1
            with self.assertRaisesRegex(ValueError, "unknown fields: unexpected"):
                parse_bar_chart(self.data)
            del target["unexpected"]

    def test_invalid_envelopes_and_chart_types(self):
        for data in (None, [], "chart", 1):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "JSON object"):
                parse_bar_chart(data)
        for field, values in (
            ("schema_version", (True, 1.0, "1", None, 2)),
            ("chart_type", ([], {}, None, True, "line")),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    data = {**self.data, field: value}
                    with self.assertRaises(ValueError):
                        parse_bar_chart(data)

    def test_array_and_series_object_boundaries(self):
        for field in ("categories", "series", "values", "rescued"):
            for value in (None, "AB", {}, (True, False), False):
                with self.subTest(field=field, value=value):
                    data = deepcopy(self.data)
                    target = data if field in ("categories", "series") else data["series"][0]
                    target[field] = value
                    with self.assertRaisesRegex(ValueError, "JSON array"):
                        parse_bar_chart(data)
        for value in (None, "series", [], 1):
            with self.subTest(series=value), self.assertRaisesRegex(ValueError, "JSON object"):
                parse_bar_chart({**self.data, "series": [value]})

    def test_invalid_booleans(self):
        for field in ("has_bars", "omit_legend"):
            for value in (0, 1, "false", None, []):
                with self.subTest(field=field, value=value):
                    data = deepcopy(self.data)
                    target = data if field == "omit_legend" else data["series"][0]
                    target[field] = value
                    with self.assertRaisesRegex(ValueError, "boolean"):
                        parse_bar_chart(data)

    def test_dataclass_validation_is_preserved(self):
        invalid = [
            ("categories", []), ("categories", ["A", None]),
            ("series", []), ("title", None), ("x_axis_title", ""),
            ("y_axis_title", 1), ("source_key", "image.png"),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                parse_bar_chart({**self.data, field: value})
        for field, value in (
            ("values", []), ("values", [1]), ("values", [True, 2]),
            ("values", [None, 2]), ("values", ["1", 2]),
            ("values", [float("nan"), 2]), ("values", [float("inf"), 2]),
            ("values", [-float("inf"), 2]),
            ("values", [10 ** 400, 2]), ("values", [-10 ** 400, 2]),
            ("rescued", [True]), ("rescued", [0, 1]),
            ("name", ""), ("has_bars", False),
        ):
            with self.subTest(series_field=field, value=value):
                data = deepcopy(self.data)
                data["series"][0][field] = value
                with self.assertRaises(ValueError):
                    parse_bar_chart(data)
