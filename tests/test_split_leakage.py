"""Leakage checks on the real train/val/test splits; skipped when data is absent.

The datasets are not in the repository (PlotQA is downloaded separately; PMC is
CC BY-NC-SA and gitignored). Each class below skips with the missing path.
Hard failures: shared file names, byte-identical images, papers in several
splits. Near-duplicate (dHash) hits only warn.
"""

import unittest
import warnings
from pathlib import Path

from src.dataset.check_splits import list_images, load_splits, run_checks


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


class SplitLeakageCase:
    """Base for one dataset; not collected itself (no TestCase base)."""

    image_root: Path
    group_regex = None
    check_file_names = True

    @classmethod
    def setUpClass(cls):
        missing = [s for s in SPLITS
                   if not (cls.image_root / s).is_dir() or not list_images(cls.image_root / s)]
        if missing:
            raise unittest.SkipTest(
                f"{cls.image_root} has no images for split(s) {', '.join(missing)}; "
                "dataset not available on this machine"
            )
        cls.report = run_checks(
            load_splits({s: cls.image_root / s for s in SPLITS}),
            group_regex=cls.group_regex,
        )

    def test_no_shared_file_names(self):
        if not self.check_file_names:
            self.skipTest("file-name check disabled for this dataset (see class comment)")
        self.assertEqual(self.report.shared_names[:10], [])

    def test_no_byte_identical_images(self):
        pairs = [(a, pa.name, b, pb.name) for a, pa, b, pb in self.report.exact_duplicates[:10]]
        self.assertEqual(pairs, [])

    def test_near_duplicates_only_warn(self):
        near = self.report.near_duplicates
        if near:
            pairs = "; ".join(f"{sa}:{pa.name} ~ {sb}:{pb.name} (dist {d})"
                              for sa, pa, sb, pb, d in near[:10])
            warnings.warn(f"{len(near)} near-duplicate pair(s) across splits: {pairs}")


class PmcSplitLeakageTest(SplitLeakageCase, unittest.TestCase):
    image_root = ROOT / "data" / "pmc_yolo" / "images"
    group_regex = r"(PMC\d+)"

    def test_no_paper_in_several_splits(self):
        self.assertEqual(list(self.report.shared_groups.items())[:10], [])
        self.assertGreater(self.report.n_groups, 0, "regex matched no file names")


# Built by src/dataset/pmc_bars_to_yolo.py from plots_detection. Its papers share
# one split map with data/pmc_yolo, so papers must not repeat across splits here either.
class PmcBarsSplitLeakageTest(SplitLeakageCase, unittest.TestCase):
    image_root = ROOT / "data" / "pmc_yolo_bars" / "images"
    group_regex = r"(PMC\d+)"

    def test_no_paper_in_several_splits(self):
        self.assertEqual(list(self.report.shared_groups.items())[:10], [])
        self.assertGreater(self.report.n_groups, 0, "regex matched no file names")


# The deduplicated copy written by src/dataset/remove_split_leakage.py, which
# the redone baseline used; not the raw prepare_plotqa.py output.
# PlotQA reuses numeric image IDs across splits (docs/contracts.md), so the same
# file name in two splits is not a leak; byte-identical images (SHA-256) are.
class PlotQASplitLeakageTest(SplitLeakageCase, unittest.TestCase):
    image_root = ROOT / "dataset_yolo_vbars_clean" / "images"
    check_file_names = False


if __name__ == "__main__":
    unittest.main()
