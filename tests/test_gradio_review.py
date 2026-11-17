"""Review drafts, completeness messages, and tactile generation."""

import tempfile
import unittest
from pathlib import Path

from src.contracts import SeriesSpec
from src.gradio_app import (
    ManualGenerationBlocked,
    alert_message,
    analyzing_status,
    build_review_model,
    complete_status,
    completeness_summary,
    generate_review_submission,
    generation_report,
    ready_status,
    review_overrides,
    review_text_rows,
    review_value_rows,
    update_review_draft,
)
from src.tactile.completeness import (
    CANNOT_BUILD,
    NEEDS_INPUT,
    READY,
    Completeness,
    MissingItem,
)

from tests.gradio_helpers import GradioTestCase, make_spec


class GradioReviewTest(GradioTestCase):
    def test_accessible_status_helpers_cover_ready_busy_success_zero_and_error(self):
        self.assertIn('role="status"', ready_status())
        self.assertIn('aria-live="polite"', analyzing_status())
        self.assertIn("Analysis complete", complete_status())
        self.assertIn("Analysis complete", complete_status(0))
        self.assertIn("Analysis complete", complete_status(3, 2))
        self.assertIn('role="alert"', alert_message("Try again"))

    def test_completeness_summary_covers_all_states_and_details(self):
        self.assertIn("Reconstruction: Ready", completeness_summary(Completeness(READY)))
        needs_input = completeness_summary(Completeness(
            NEEDS_INPUT,
            notes=["Check the inferred value."],
            missing=[MissingItem("title", "No title was read.", "--title")],
        ))
        self.assertIn("Reconstruction: Needs input", needs_input)
        self.assertIn("Check the inferred value", needs_input)
        self.assertIn("`title`: No title was read", needs_input)
        cannot_build = completeness_summary(Completeness(
            CANNOT_BUILD, blockers=["The axis scale is unavailable."]
        ))
        self.assertIn("Reconstruction: Cannot build", cannot_build)
        self.assertIn("The axis scale is unavailable", cannot_build)

    def test_review_model_has_stable_order_and_read_only_values(self):
        spec = make_spec(
            categories=["2023", "2024"],
            series=[
                SeriesSpec("North", "#112233", [10.0, 20.0]),
                SeriesSpec("South", "#445566", [30.0, 40.0]),
            ],
        )
        model = build_review_model(spec, Completeness(READY))

        self.assertEqual(
            [binding.key for binding in model.bindings],
            [
                "title", "x_axis_title", "y_axis_title",
                "category[0]", "category[1]",
                "series_name[0]", "series_name[1]",
            ],
        )
        self.assertEqual(review_text_rows(model)[3], ["Category 0", "2023"])
        self.assertEqual(
            review_value_rows(spec),
            [
                [0, "2023", 0, "North", 10.0],
                [1, "2024", 0, "North", 20.0],
                [0, "2023", 1, "South", 30.0],
                [1, "2024", 1, "South", 40.0],
            ],
        )

    def test_review_mapping_ignores_submitted_labels_and_preserves_spec(self):
        spec = make_spec(
            categories=["2023", "2024"],
            series=[SeriesSpec(None, "#112233", [25.0, 50.0])],
        )
        model = build_review_model(spec, Completeness(READY))
        rows = review_text_rows(model)
        rows[0] = ["category[1]", "Corrected sales"]
        rows[3] = ["title", "FY 2023"]
        rows[4] = ["anything", "FY 2024"]

        updated = update_review_draft(model, rows)
        overrides = review_overrides(updated)

        self.assertEqual(overrides.title, "Corrected sales")
        self.assertEqual(overrides.categories, {0: "FY 2023", 1: "FY 2024"})
        self.assertEqual(spec.title, "Sales")
        self.assertEqual(spec.categories, ["2023", "2024"])
        self.assertEqual(review_text_rows(updated)[0][0], "Chart title")
        self.assertEqual(review_text_rows(updated)[3][0], "Category 0")

    def test_review_confirmation_and_blank_validation(self):
        spec = make_spec(flags=["title_cut_off"])
        completeness = Completeness(
            NEEDS_INPUT,
            missing=[MissingItem("title", "Confirm title.", "--title")],
        )
        model = build_review_model(spec, completeness)
        self.assertTrue(model.bindings[0].confirmation_required)
        self.assertEqual(review_overrides(model).title, "Sales")

        rows = review_text_rows(model)
        rows[1][1] = "   "
        draft = update_review_draft(model, rows)
        self.assertEqual(draft.draft_values[1], "   ")
        with self.assertRaisesRegex(ValueError, "X-axis title cannot be blank"):
            review_overrides(draft)

        unnamed = build_review_model(make_spec(), Completeness(READY))
        self.assertFalse(unnamed.bindings[-1].required)
        self.assertEqual(review_overrides(unnamed).series_names, {})

    def test_review_draft_rejects_malformed_table_shapes_and_cells(self):
        model = build_review_model(make_spec(), Completeness(READY))
        rows = review_text_rows(model)
        invalid = (
            rows[:-1],
            [*rows, ["extra", "value"]],
            [["only one column"], *rows[1:]],
            [["Chart title", 42], *rows[1:]],
        )
        for submitted in invalid:
            with self.subTest(submitted=submitted):
                with self.assertRaises(ValueError):
                    update_review_draft(model, submitted)
        self.assertEqual(model.spec.title, "Sales")
        self.assertEqual(model.draft_values[0], "Sales")

    def test_manual_generation_applies_and_commits_review_text(self):
        spec = make_spec(flags=["title_cut_off"])
        assessment = Completeness(
            NEEDS_INPUT,
            missing=[MissingItem("title", "Confirm title.", "--title")],
        )
        model = build_review_model(spec, assessment)
        rows = review_text_rows(model)
        rows[0][1] = "Confirmed sales"
        rows[3][1] = "FY 2024"

        with tempfile.TemporaryDirectory() as root:
            result = generate_review_submission(model, rows, root)
            self.assertEqual(result.completeness.status, READY)
            self.assertEqual(result.review_model.spec.title, "Confirmed sales")
            self.assertEqual(result.review_model.spec.categories, ["FY 2024"])
            self.assertNotIn("title_cut_off", result.review_model.spec.flags)
            self.assertTrue(result.generation.zip_path.is_file())
            report = generation_report(result)

        self.assertIn("chart title supplied by user", report)
        self.assertIn("category 0 supplied by user", report)
        self.assertIn("Tactile graphic", report)
        self.assertIn("Checked", report)
        self.assertIn("annotated review copies", report)

    def test_manual_generation_blocks_invalid_text_and_structural_uncertainty(self):
        model = build_review_model(make_spec(), Completeness(READY))
        rows = review_text_rows(model)
        rows[0][1] = "Sales 😀"
        with tempfile.TemporaryDirectory() as root, self.assertRaises(
            ManualGenerationBlocked
        ) as raised:
            generate_review_submission(model, rows, root)
        self.assertEqual(raised.exception.completeness.status, NEEDS_INPUT)
        self.assertIn("cannot be written in braille",
                      raised.exception.completeness.missing[0].reason)

        blocked_spec = make_spec(flags=["category_ambiguous"])
        blocked = build_review_model(
            blocked_spec,
            Completeness(CANNOT_BUILD, blockers=["Ambiguous categories."]),
        )
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ManualGenerationBlocked) as raised:
                generate_review_submission(blocked, review_text_rows(blocked), root)
            self.assertEqual(raised.exception.completeness.status, CANNOT_BUILD)
            self.assertFalse(any(Path(root).iterdir()))

        blank = review_text_rows(model)
        blank[1][1] = "   "
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(
            ValueError, "X-axis title cannot be blank"
        ):
            generate_review_submission(model, blank, root)


if __name__ == "__main__":
    unittest.main()
