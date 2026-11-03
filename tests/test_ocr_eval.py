"""Text normalization, number parsing and edit distance used by the OCR scoring."""

import unittest

from src.baseline.ocr_decimal_diagnosis import classify, has_decimal
from src.baseline.ocr_eval import allow_chars, edit_distance, normalize, parse_number, prepare


class OcrScoringTest(unittest.TestCase):
    def test_normalize_unifies_unicode_and_whitespace(self):
        self.assertEqual(normalize("Speciﬁc  ’coping’\nstrategy"), "Specific 'coping' strategy")
        self.assertEqual(normalize("ADA-GEL—nBG"), "ADA-GEL-nBG")
        self.assertEqual(normalize("−2"), "-2")

    def test_literal_backslash_n_is_whitespace_in_normalized_form_only(self):
        self.assertEqual(normalize("Aza \\n +H"), "Aza +H")
        self.assertEqual(normalize("NHR:\\nRegurg"), "NHR: Regurg")
        self.assertEqual(normalize("a\nb"), "a b")
        self.assertEqual(normalize(r"CFP\alpha"), r"CFP\alpha")  # other backslashes are untouched

    def test_parse_number(self):
        for text, value in (("12", 12.0), ("0.12", 0.12), ("−3.5", -3.5), ("45%", 45.0),
                            ("1,000", 1000.0), ("0,5", 0.5), ("1e3", 1000.0), (" 7 ", 7.0)):
            self.assertEqual(parse_number(text), value, text)
        for text in ("", "abc", "1~2", "<0.05", "1.2.3", "nan"):
            self.assertIsNone(parse_number(text), text)

    def test_has_decimal_counts_points_and_decimal_commas_but_not_thousands(self):
        for text in ("0.5", "4,5", "-0.25", "8,0%", "0.012"):
            self.assertTrue(has_decimal(text), text)
        for text in ("45", "1,000", "12,345,678", "50%", "", "a.b"):
            self.assertFalse(has_decimal(text), text)

    def test_classify_decimal_cases(self):
        self.assertEqual(classify({"text": "4.5", "pred": "4.5"}), "retained")
        self.assertEqual(classify({"text": "4.5", "pred": "45"}), "dropped")
        self.assertEqual(classify({"text": "45", "pred": "4.5"}), "invented")
        self.assertEqual(classify({"text": "45", "pred": "45"}), "no decimal")
        self.assertEqual(classify({"text": "0.5", "pred": "0,5"}), "retained")  # comma for point still a decimal

    def test_allow_chars_and_prepare_options(self):
        self.assertIsNone(allow_chars({}))
        self.assertTrue(allow_chars({"whitelist": True}))
        self.assertEqual(allow_chars({"allow": "0123"}), "0123")
        import numpy as np
        crop = np.full((10, 20), 200, np.uint8)
        crop[4:6, 8:12] = 20
        plain, sharp = prepare(crop, 40, False), prepare(crop, 40, False, sharpen=True)
        self.assertEqual(plain.shape, (40, 80))
        self.assertFalse((plain == sharp).all())  # sharpening changed the pixels
        self.assertTrue(set(np.unique(prepare(crop, 10, True))) <= {0, 255})

    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("same", "same"), 0)


if __name__ == "__main__":
    unittest.main()
