"""Artifact generation tests; no detector, OCR, or Gradio runs."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree
from zipfile import ZipFile

from src.contracts import SeriesSpec, VerticalBarSpec, YAxisSpec
from src.tactile.generation import (
    GenerationError,
    cleanup_generation,
    generate_tactile_artifacts,
)
from src.tactile.rules import RULES, Violation


def make_spec(**changes):
    fields = dict(
        source_key="input/chart.png",
        title="Sales",
        x_axis_title="Year",
        y_axis_title="Revenue",
        y_axis=YAxisSpec(0, 100, [0, 50, 100]),
        categories=["2001", "2002", "2003"],
        series=[SeriesSpec(None, "#112233", [10.0, 20.0, 30.0])],
        bars_adjacent=False,
        flags=[],
        confidence={},
    )
    fields.update(changes)
    return VerticalBarSpec(**fields)


def divided_spec():
    categories = [f"Category number {index}" for index in range(12)]
    return make_spec(
        categories=categories,
        series=[
            SeriesSpec("Alpha", "#112233", [float(i + 1) for i in range(12)]),
            SeriesSpec("Beta", "#445566", [float(i + 2) for i in range(12)]),
            SeriesSpec("Gamma", "#778899", [float(i + 3) for i in range(12)]),
        ],
    )


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_single_page_artifacts_and_zip(self):
        result = generate_tactile_artifacts(make_spec(), self.root)

        self.assertEqual(len(result.pages), 1)
        page = result.pages[0]
        self.assertEqual(
            (page.page_number, page.page_count, page.page_kind, page.section, page.caption),
            (1, 1, "graph", None, "Tactile graphic"),
        )
        self.assertEqual(page.review_svg.name, "chart-review.svg")
        self.assertEqual(page.clean_svg.name, "chart.svg")
        self.assertIn("PRINT LABELS FOR REVIEW ONLY", page.review_svg.read_text())
        self.assertNotIn('id="print"', page.clean_svg.read_text())
        self.assertNotIn('id="violations"', page.clean_svg.read_text())
        with ZipFile(result.zip_path) as archive:
            self.assertEqual(archive.namelist(), ["chart.svg"])
            self.assertEqual(archive.read("chart.svg"), page.clean_svg.read_bytes())

    def test_multi_page_metadata_order_and_overview_finding(self):
        result = generate_tactile_artifacts(divided_spec(), self.root)

        self.assertGreater(len(result.pages), 2)
        self.assertEqual(result.pages[0].page_kind, "key")
        self.assertEqual(result.pages[0].caption, f"Page 1 of {len(result.pages)} — key")
        graph_pages = [page for page in result.pages if page.page_kind == "graph"]
        self.assertEqual(
            [page.section for page in graph_pages],
            [(i, len(graph_pages)) for i in range(1, len(graph_pages) + 1)],
        )
        self.assertIn("overview_first", {v.rule.id for v in graph_pages[0].violations})
        self.assertEqual(
            graph_pages[0].caption,
            f"Page 2 of {len(result.pages)} — part 1 of {len(graph_pages)}",
        )
        with ZipFile(result.zip_path) as archive:
            self.assertEqual(archive.namelist(), [p.clean_svg.name for p in result.pages])
            self.assertFalse(any("review" in name for name in archive.namelist()))

    def test_review_overlay_only_boxes_addressable_violations(self):
        addressable = Violation(RULES["bar_width"], "too narrow", ("bar-0-0",))
        with mock.patch("src.tactile.generation.check_layout", return_value=[addressable]):
            result = generate_tactile_artifacts(make_spec(), self.root)
        review = result.pages[0].review_svg.read_text()
        self.assertIn('id="violations"', review)
        self.assertIn("too narrow", review)
        self.assertNotIn('id="violations"', result.pages[0].clean_svg.read_text())

        report_only = Violation(RULES["overview_first"], "overview needed", ())
        with mock.patch("src.tactile.generation.check_layout", return_value=[report_only]):
            result = generate_tactile_artifacts(make_spec(), self.root)
        svg = ElementTree.fromstring(result.pages[0].review_svg.read_text().split("\n", 1)[1])
        ns = {"svg": "http://www.w3.org/2000/svg"}
        group = svg.find(".//svg:g[@id='violations']", ns)
        self.assertIsNotNone(group)
        self.assertEqual(list(group), [])

    def test_global_notes_are_separate_from_page_review_information(self):
        spec = make_spec(
            flags=["y_ticks_not_arithmetic"],
            series=[SeriesSpec(None, "#112233", [-0.1, 20.0, 30.0])],
        )
        result = generate_tactile_artifacts(spec, self.root)
        joined = " ".join(result.notes)
        self.assertIn("not evenly spaced", joined)
        self.assertIn("treated as 0", joined)
        self.assertIsInstance(result.pages[0].notes, tuple)
        self.assertIsInstance(result.pages[0].needs_review, tuple)

    def test_non_ready_and_invalid_root_are_validation_errors(self):
        before = set(self.root.iterdir())
        with self.assertRaises(GenerationError) as raised:
            generate_tactile_artifacts(make_spec(title=None), self.root)
        self.assertEqual(raised.exception.stage, "validation")
        self.assertEqual(set(self.root.iterdir()), before)

        with self.assertRaises(GenerationError) as raised:
            generate_tactile_artifacts(make_spec(), self.root / "missing")
        self.assertEqual(raised.exception.stage, "validation")

    def test_real_layout_failure_removes_job_directory(self):
        spec = make_spec(series=[SeriesSpec(None, "#112233", [-10.0, 20.0, 30.0])])
        with self.assertRaises(GenerationError) as raised:
            generate_tactile_artifacts(spec, self.root)
        self.assertEqual(raised.exception.stage, "layout")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_render_write_and_zip_failures_remove_partial_job(self):
        cases = (
            ("render", mock.patch("src.tactile.generation.render_svg", side_effect=ValueError("x"))),
            ("write", mock.patch("pathlib.Path.write_text", side_effect=OSError("x"))),
            ("zip", mock.patch("src.tactile.generation.ZipFile", side_effect=OSError("x"))),
        )
        for expected_stage, patcher in cases:
            with self.subTest(expected_stage), patcher:
                with self.assertRaises(GenerationError) as raised:
                    generate_tactile_artifacts(make_spec(), self.root)
                self.assertEqual(raised.exception.stage, expected_stage)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_later_page_write_failure_and_cleanup_failure_keep_primary_error(self):
        with mock.patch(
            "pathlib.Path.write_text",
            side_effect=[None, None, OSError("second page failed")],
        ):
            with self.assertRaises(GenerationError) as raised:
                generate_tactile_artifacts(divided_spec(), self.root)
        self.assertEqual(raised.exception.stage, "write")
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(list(self.root.iterdir()), [])

        with (
            mock.patch("src.tactile.generation.render_svg", side_effect=ValueError("render")),
            mock.patch(
                "src.tactile.generation._remove_job_dir",
                side_effect=OSError("cleanup"),
            ),
        ):
            with self.assertRaises(GenerationError) as raised:
                generate_tactile_artifacts(make_spec(), self.root)
        self.assertEqual(raised.exception.stage, "render")
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIsInstance(raised.exception.cleanup_error, OSError)

    def test_names_are_sanitized_limited_and_isolated(self):
        names = {
            "input/a name.png": "a_name.svg",
            "input/ééé.png": "chart.svg",
            "input/---.png": "chart.svg",
            f"input/{'a' * 300}.png": f"{'a' * 120}.svg",
        }
        for source_key, expected in names.items():
            with self.subTest(source_key):
                result = generate_tactile_artifacts(make_spec(source_key=source_key), self.root)
                self.assertEqual(result.pages[0].clean_svg.name, expected)

        first = generate_tactile_artifacts(make_spec(), self.root)
        second = generate_tactile_artifacts(make_spec(), self.root)
        self.assertNotEqual(first.output_dir, second.output_dir)
        self.assertTrue(first.pages[0].clean_svg.exists())
        self.assertTrue(second.pages[0].clean_svg.exists())

    def test_cleanup_is_idempotent_and_preserves_siblings(self):
        first = generate_tactile_artifacts(make_spec(), self.root)
        second = generate_tactile_artifacts(make_spec(), self.root)
        first.pages[0].review_svg.unlink()
        cleanup_generation(first, self.root)
        cleanup_generation(first, self.root)
        self.assertFalse(first.output_dir.exists())
        self.assertTrue(second.output_dir.exists())

        invalid = replace(second, output_dir=self.root / "not-a-generation")
        with self.assertRaises(ValueError):
            cleanup_generation(invalid, self.root)
        self.assertTrue(second.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
