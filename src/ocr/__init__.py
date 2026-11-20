"""OCR orchestration and engine adapters."""

from src.ocr.pipeline import OCRBackend, OCRBackendResult, run_ocr

__all__ = ["OCRBackend", "OCRBackendResult", "run_ocr"]
