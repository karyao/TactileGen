"""End-to-end check of the legend collision repair on real PlotQA charts.

In these charts two series have similar colours and the template match gave both series' bars in a slot to the
same legend entry, so one series had no bars and the chart was blocked (category_ambiguous). Reassigning the bars
by median colour reads them correctly. Skips like test_probe_charts_e2e.py when the data, weights or OCR engines
are missing.
"""

import pytest

from tests.test_probe_charts_e2e import analyse, annotations, assert_matches_ground_truth  # noqa: F401  (fixtures)

REPAIRED = ("24665", "25321", "23935")


@pytest.mark.parametrize("chart_id", REPAIRED)
def test_chart_with_two_bars_per_slot_matched_to_one_series_is_read_correctly(chart_id, annotations, analyse):
    from src.tactile.completeness import check_completeness

    spec = analyse(chart_id)
    assert not any(f.split(":")[0] in ("category_ambiguous", "category_count_mismatch") for f in spec.flags)
    assert not check_completeness(spec).blockers
    assert_matches_ground_truth(annotations[chart_id], spec)
