"""Per-region choice between the RapidOCR and Tesseract readings."""

from src.ocr.hybrid import choose
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


def test_rapidocr_that_dropped_letters_or_added_a_space_loses_to_tesseract_on_word_roles():
    assert choose("x_tick_label", result("Adolescent ferility rate", 0.98),
                  result("Adolescent fertility rate", 0.96)).text == "Adolescent fertility rate"
    assert choose("legend_label", result("Age 65 (male)", 0.97),
                  result("Age 65(male)", 0.93)).text == "Age 65(male)"


def test_the_slip_rule_ignores_punctuation_extras_other_roles_and_larger_differences():
    # Tesseract's extra "|" is not a dropped letter; y ticks and titles are untouched
    assert choose("legend_label", result("1992", 0.99), result("1992 |", 0.80)).text == "1992"
    assert choose("y_tick_label", result("ferility", 0.99), result("fertility", 0.5)).text == "ferility"
    assert choose("x_tick_label", result("Pru", 0.99), result("Peruvian", 0.6)).text == "Pru"


LONG_NAME = "Net savings(excluding particulate emission damage"


def test_an_unclosed_parenthesis_is_closed_on_word_roles():
    assert choose("legend_label", result(LONG_NAME, 0.98), result(LONG_NAME, 0.90)).text == LONG_NAME + ")"
    assert choose("x_tick_label", result("Age 65(female", 0.98), result("Age 65(female", 0.9)).text == "Age 65(female)"
    # the chosen reading is closed, whichever engine it came from, and keeps its confidence
    closed = choose("legend_label", result("Pern (all", 0.80), result("Peru (all", 0.95))   # Tesseract is chosen
    assert (closed.text, closed.confidence) == ("Peru (all)", 0.95)


def test_balanced_or_closing_only_parentheses_are_left_alone():
    assert choose("legend_label", result("Age 65(male)", 0.98), result("Age 65(male)", 0.9)).text == "Age 65(male)"
    assert choose("legend_label", result("male)", 0.98), result("male)", 0.9)).text == "male)"
    assert choose("legend_label", result("a (b) (c", 0.98), result("a (b) (c", 0.9)).text == "a (b) (c)"


def test_parenthesis_closing_ignores_other_roles_and_missing_readings():
    assert choose("chart_title", result("Savings (excluding", 0.9), result("Savings (excluding", 0.9)).text == "Savings (excluding"
    assert choose("y_tick_label", result("(5", 0.9), result("(5", 0.9)).text == "(5"
    assert choose("legend_label", None, None) is None
    assert choose("legend_label", None, result("Peru (all", 0.5)).text == "Peru (all)"
