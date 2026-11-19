import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_e2e.py"
spec = importlib.util.spec_from_file_location("audit_e2e", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def bbox(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


def record(chart_type, models, value_ticks):
    axis = "y_axis" if chart_type == "vbar_categorical" else "x_axis"
    other = "x_axis" if axis == "y_axis" else "y_axis"
    return {"type": chart_type, "image_index": 1, "models": models, "general_figure_info": {
        axis: {"major_ticks": {"values": value_ticks}}, other: {"major_ticks": {"values": [0, 1]}}}}


class GroundTruthTest(unittest.TestCase):
    def test_vertical_categories_read_left_to_right(self):
        models = [{"name": "A", "colors": "#000000", "x": ["2001", "2000"], "y": [5.0, 2.0],
                   "bboxes": [bbox(200, 50, 20, 150), bbox(100, 140, 20, 60)]}]
        gt = audit.ground_truth(record("vbar_categorical", models, [0, 2, 4, 6]))
        self.assertEqual(gt["orientation"], "vertical")
        self.assertEqual(gt["categories"], ["2000", "2001"])
        self.assertEqual(gt["series"][0]["values"], [2.0, 5.0])
        self.assertEqual(gt["axis_max"], 6.0)
        self.assertEqual([b["thickness"] for b in gt["bars"]], [60, 150])

    def test_horizontal_values_and_categories_swap_and_read_top_to_bottom(self):
        # PlotQA lists horizontal categories bottom to top, values in "x".
        models = [{"name": "A", "colors": "#000000", "y": ["1990", "2000"], "x": [63.2, 35.3],
                   "bboxes": [bbox(62, 550, 626, 34), bbox(62, 320, 349, 34)]}]
        gt = audit.ground_truth(record("hbar_categorical", models, [0, 10, 70]))
        self.assertEqual(gt["orientation"], "horizontal")
        self.assertEqual(gt["categories"], ["2000", "1990"])
        self.assertEqual(gt["series"][0]["values"], [35.3, 63.2])
        self.assertEqual(gt["axis_max"], 70.0)
        self.assertEqual([b["thickness"] for b in gt["bars"]], [349, 626])

    def test_zero_size_bar_is_kept_with_zero_thickness(self):
        models = [{"name": "A", "colors": "#000000", "x": ["a", "b"], "y": [0.0, 3.0],
                   "bboxes": [bbox(10, 100, 20, 0), bbox(40, 70, 20, 30)]}]
        gt = audit.ground_truth(record("vbar_categorical", models, [0, 3]))
        self.assertEqual(gt["bars"][0]["thickness"], 0)


class ReasonsAndGroupsTest(unittest.TestCase):
    def test_block_reasons_read_flags_missing_text_and_bana_rules(self):
        blocked = "This chart cannot be turned into a tactile graphic:\n  - Legend mismatch. (series_ambiguous) Try\n"
        self.assertEqual(audit.block_reasons(blocked, 3), (["series_ambiguous"], [], []))
        needs = "needs text:\n  title: cut off. Supply it with --title\n  series_name[2]: x\n"
        self.assertEqual(audit.block_reasons(needs, 2), ([], ["title", "series_name[n]"], []))
        bana = "Checked 24 BANA rules: 1 problem(s) found.\n  [bar_width] 6.6.6.7: too thin\n"
        self.assertEqual(audit.block_reasons(bana, 1), ([], [], ["bar_width"]))

    def test_missed_bars_split_into_tiny_and_real(self):
        self.assertEqual(audit.failure_group("detector_missing_bars", 2, 2), "tiny_bars_undetected")
        self.assertEqual(audit.failure_group("detector_missing_bars", 2, 1), "real_bars_undetected")
        self.assertEqual(audit.failure_group("series_separation", 0, 0), "series_legend_colour_match")
        self.assertEqual(audit.failure_group("bar_rescue_wrong", 0, 0), "other_rare")
        self.assertEqual(audit.failure_group("other_wrong_values", 0, 0), "bar_values_off")
        self.assertEqual(audit.failure_group("none_correct", 0, 0), "")


if __name__ == "__main__":
    unittest.main()
