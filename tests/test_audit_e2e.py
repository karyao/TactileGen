import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


class SelectTest(unittest.TestCase):
    def test_an_orientation_gets_the_same_charts_alone_or_with_the_other(self):
        records = [{"type": t, "image_index": i} for i in range(50)
                   for t in ("vbar_categorical", "hbar_categorical", "line")]
        from unittest import mock
        with mock.patch("src.dataset.prepare_plotqa.iter_annotations", return_value=records):
            both = audit._select(None, ["vbar_categorical", "hbar_categorical"], 10, 0)
            alone = audit._select(None, ["hbar_categorical"], 10, 0)
        self.assertEqual(alone, [r for r in both if r["type"] == "hbar_categorical"])
        self.assertEqual(len(alone), 10)


class ClassificationTest(unittest.TestCase):
    def analyse(self, values, name="A", missing=False, rescued=None, bana=False):
        models = [{"name": "A", "colors": "#000000", "x": ["a", "b"], "y": [20.0, 40.0],
                   "bboxes": [bbox(10, 80, 10, 20), bbox(30, 60, 10, 40)]}]
        chart = record("vbar_categorical", models, [0, 100])
        series = SimpleNamespace(name=name, values=values, value_rescued=rescued, source_color=None)
        chart_spec = SimpleNamespace(categories=["a", "b"], series=[series], flags=[],
                                     y_axis=SimpleNamespace(tick_values=[0, 100]))
        row = {"id": "1", "record": chart, "spec": chart_spec, "frame": object(),
               "exit": 1 if bana else 0, "stdout": "  [bar_width] problem\n" if bana else "", "stderr": ""}
        with mock.patch.object(audit, "surviving_bars", return_value=([], [])), \
             mock.patch.object(audit, "match_bars", return_value=({}, [0] if missing else [])):
            return audit.analyse(row)

    def test_name_mismatch_with_exact_values(self):
        row = self.analyse([20, 40], name="wrong")
        self.assertEqual(row["group"], "series_name_mismatch")
        self.assertTrue(row["values_ok"])

    def test_unread_and_numeric_errors(self):
        unread = self.analyse([None, 40])
        self.assertEqual((unread["group"], unread["n_unread_values"], unread["n_numeric_value_errors"]),
                         ("bar_value_unread", 1, 0))
        mixed = self.analyse([None, 42])
        self.assertEqual((mixed["group"], mixed["n_bad_values"], mixed["n_unread_values"],
                          mixed["n_numeric_value_errors"]), ("bar_values_off", 2, 1, 1))

    def test_tolerance_boundary(self):
        self.assertTrue(self.analyse([21, 40])["values_ok"])
        self.assertEqual(self.analyse([21.01, 40])["group"], "bar_values_off")

    def test_earlier_causes_remain_priorities(self):
        row = self.analyse([22, 40], missing=True)
        self.assertEqual(row["primary_cause"], "detector_missing_bars")
        self.assertEqual(row["n_numeric_value_errors"], 1)
        self.assertEqual(self.analyse([None, 40], rescued=[True, False])["primary_cause"], "bar_rescue_wrong")
        self.assertEqual(self.analyse([20, 40], bana=True)["primary_cause"], "false_block")


if __name__ == "__main__":
    unittest.main()
