"""Text normalization, number parsing and edit distance used by the OCR scoring."""

import unittest

from src.baseline.ocr_eval import edit_distance, normalize, parse_number


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

    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("same", "same"), 0)


if __name__ == "__main__":
    unittest.main()
