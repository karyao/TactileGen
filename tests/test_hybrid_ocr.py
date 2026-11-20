"""Per-region choice between the RapidOCR and Tesseract readings."""

from src.hybrid_ocr import choose
from src.ocr import OCRBackendResult


def result(text, confidence):
    return OCRBackendResult(text, confidence)


def test_a_dollar_only_rapidocr_read_wins_even_when_tesseract_is_more_confident():
    assert choose("legend_label", result("US$", 0.90), result("USS", 0.95)).text == "US$"


def test_margin_applies_per_role():
    assert choose("legend_label", result("Peru", 0.97), result("Pern", 0.95)).text == "Pern"
    assert choose("x_tick_label", result("Peru", 0.97), result("Pern", 0.95)).text == "Peru"


def test_roles_without_a_margin_keep_rapidocr_and_missing_readings_fall_back():
    assert choose("chart_title", result("A", 0.5), result("B", 0.99)).text == "A"
    assert choose("legend_label", None, result("B", 0.5)).text == "B"
