"""App-owned, lazy OCR backends shared by serialized analysis requests."""

from src.gradio_ui.settings import DEFAULT_OCR_ENGINE


OCR_ENGINES = ("rapidocr", "tesseract")


class OCRBackendResolver:
    """Cache successful initializations; leave failures retryable."""

    def __init__(self, initial_engine=DEFAULT_OCR_ENGINE, backend=None):
        self.validate(initial_engine)
        self._backends = {} if backend is None else {initial_engine: backend}

    @staticmethod
    def validate(engine):
        if engine not in OCR_ENGINES:
            raise ValueError(f"Unsupported OCR engine: {engine}")

    def resolve(self, engine):
        self.validate(engine)
        if engine not in self._backends:
            if engine == "rapidocr":
                from src.ocr.rapidocr import RapidOCRBackend

                backend = RapidOCRBackend()
            else:
                from src.ocr.tesseract import TesseractBackend

                backend = TesseractBackend()
            self._backends[engine] = backend
        return self._backends[engine]
