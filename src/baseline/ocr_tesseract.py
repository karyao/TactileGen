"""Tesseract engine for the shared OCR protocol (src/baseline/ocr_eval.py), CPU only.

  python -m src.baseline.ocr_tesseract {tune,test,rescore} --out <dir>

Tesseract is found via $TESSERACT_CMD, then PATH, then the tactilegen-ocr conda
env (see docs/ocr_baselines.md for how to create it).
"""

import itertools
import os
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

from src.baseline.ocr_eval import WHITELIST, main

FALLBACK_CMD = Path.home() / "miniconda3/envs/tactilegen-ocr/bin/tesseract"


def tesseract_cmd() -> str:
    found = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract") or (
        str(FALLBACK_CMD) if FALLBACK_CMD.is_file() else None)
    if not found:
        raise SystemExit("tesseract not found: set TESSERACT_CMD or install it in the tactilegen-ocr conda env")
    return found


def run_tesseract(gray, psm, whitelist, cmd):
    """-> (text, confidence 0-1). Text lines are joined with newlines."""
    ok, png = cv2.imencode(".png", gray)
    args = ["nice", "-n", "19", cmd, "stdin", "stdout", "--psm", str(psm), "-l", "eng"]
    if whitelist:
        args += ["-c", f"tessedit_char_whitelist={WHITELIST}"]
    out = subprocess.run(args + ["tsv"], input=png.tobytes(), capture_output=True,
                         env={**os.environ, "OMP_THREAD_LIMIT": "1"}).stdout.decode("utf-8", "replace")
    lines, confs = defaultdict(list), []
    for row in out.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) == 12 and cols[0] == "5" and cols[11].strip():
            lines[tuple(cols[2:5])].append(cols[11].strip())
            confs.append(float(cols[10]))
    return "\n".join(" ".join(words) for words in lines.values()), (statistics.fmean(confs) / 100 if confs else 0.0)


class Tesseract:
    name = "tesseract"
    workers = 6

    def __init__(self):
        self.cmd = tesseract_cmd()

    def grid(self):
        return [dict(height=h, binarize=b, psm=p) for h, b, p in itertools.product((32, 64), (False, True), (7, 6))]

    def read(self, gray, cfg):
        return run_tesseract(gray, cfg["psm"], cfg.get("whitelist"), self.cmd)


if __name__ == "__main__":
    main(Tesseract())
