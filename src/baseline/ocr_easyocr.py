"""EasyOCR engine for the shared OCR protocol (src/baseline/ocr_eval.py), on GPU.

  ~/.venvs/tactilegen-easyocr/bin/python -m src.baseline.ocr_easyocr {tune,test,rescore} --out <dir>

Two reading modes are tuned: ``recognize`` (the whole crop is one text line, no
detector) and ``readtext`` (EasyOCR's own text detector, then recognition, so
multi-line crops work). Install notes are in docs/ocr_baselines.md.
"""

import itertools
import os
import statistics
from pathlib import Path

from src.baseline.ocr_eval import WHITELIST, main

MODEL_DIR = Path(os.environ.get("EASYOCR_MODEL_DIR", Path.home() / ".venvs/tactilegen-easyocr/models"))


class EasyOCR:
    name = "easyocr"
    workers = 1  # one GPU model; calls are sequential

    def __init__(self):
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(["en"], gpu=True, model_storage_directory=str(MODEL_DIR),
                                          user_network_directory=str(MODEL_DIR / "user"), verbose=False)
        return self._reader

    def grid(self):
        return [dict(height=h, binarize=b, mode=m)
                for h, b, m in itertools.product((32, 64), (False, True), ("recognize", "readtext"))]

    def read(self, gray, cfg):
        allow = WHITELIST if cfg.get("whitelist") else None
        if cfg["mode"] == "recognize":
            results = self.reader.recognize(gray, allowlist=allow, detail=1)
        else:
            results = self.reader.readtext(gray, allowlist=allow, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        # Group boxes into lines top to bottom, words left to right.
        boxes = sorted(((statistics.fmean(p[1] for p in b), min(p[0] for p in b), max(p[1] for p in b) - min(p[1] for p in b), t, c)
                        for b, t, c in results))
        height = statistics.median(h for _, _, h, _, _ in boxes) or 1
        lines, last = [], None
        for cy, x, _, text, conf in boxes:
            if last is None or cy - last > 0.6 * height:
                lines.append([])
            lines[-1].append((x, text))
            last = cy
        text = "\n".join(" ".join(t for _, t in sorted(line)) for line in lines)
        return text, statistics.fmean(c for *_, c in boxes)


if __name__ == "__main__":
    main(EasyOCR())
