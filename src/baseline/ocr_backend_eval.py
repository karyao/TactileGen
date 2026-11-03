"""Score an OCRBackend-protocol implementation with the shared OCR protocol (ground-truth crops).

The backend is loaded unmodified from a directory (default ``output/karen_ocr``, which is
not part of the repository) holding ``ocr.py`` (``run_ocr``, ``OCRBackendResult``) and
``tesseract_ocr.py`` (``TesseractBackend``), and ``pylib/`` with ``pytesseract``. Boxes are
the ground-truth text boxes of the same crops, normalization and per-class table as
src/baseline/ocr_eval.py; each image goes through the backend's own ``run_ocr`` as a
DetectionFrame, as it would from a detector's JSON. The backend's settings are fixed, so
nothing is tuned: val is a clean score, and test is scored once.

  python -m src.baseline.ocr_backend_eval --split val  --out output/ocr/karen
  python -m src.baseline.ocr_backend_eval --split test --out output/ocr/karen
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from src.baseline.ocr_eval import NUMERIC, load_crops, summarize
from src.baseline.ocr_tesseract import tesseract_cmd
from src.contracts import Box, Detection, DetectionFrame

ROLE_BY_CLASS = {"x_title": "x_axis_title", "y_title": "y_axis_title", "chart_title": "chart_title",
                 "xlabel": "x_tick_label", "ylabel": "y_tick_label", "legend_label": "legend_label",
                 "legend_title": "legend_label",  # no role of its own; any text role is read the same way
                 "value_label": "value_label"}


def load_backend(directory: Path):
    """-> (run_ocr, TesseractBackend) from files that are not part of this repository."""
    sys.path.insert(0, str(directory / "pylib"))
    for name, filename in (("src.ocr", "ocr.py"), ("src.tesseract_ocr", "tesseract_ocr.py")):
        spec = importlib.util.spec_from_file_location(name, directory / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd()  # where the binary is; the backend is not touched
    return sys.modules["src.ocr"].run_ocr, sys.modules["src.tesseract_ocr"].TesseractBackend


def clip(box, width, height):
    x, y, w, h = box
    x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(width), x + w), min(float(height), y + h)
    return Box(x1, y1, x2, y2) if x2 - x1 >= 1 and y2 - y1 >= 1 else None


def read_crops(crops, run_ocr, backend, language="eng", workers=6):
    by_image = defaultdict(list)
    for c in crops:
        by_image[c["image"]].append(c)

    def one(item):
        path, cs = item
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        detections, owner = [], {}
        for i, c in enumerate(cs):
            box = clip(c["box"], width, height)
            if box is None:
                continue
            detections.append(Detection(f"c{i}", ROLE_BY_CLASS[c["klass"]], box, 1.0))
            owner[f"c{i}"] = c
        frame = DetectionFrame(f"read/{Path(path).name}", width, height, tuple(detections))
        ocr = run_ocr(image, frame, backend, language)
        return [dict(owner[r.detection_id], pred=r.text, conf=r.confidence) for r in ocr.regions]

    with ThreadPoolExecutor(workers) as pool:
        return [row for rows in pool.map(one, by_image.items()) for row in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/pmc_yolo_bars"))
    ap.add_argument("--backend-dir", type=Path, default=Path("output/karen_ocr"))
    args = ap.parse_args()
    result = args.out / f"ocr_{args.split}.json"
    if args.split == "test" and result.exists():
        raise SystemExit(f"test already scored once: {result}")
    os.nice(19)
    os.environ["OMP_THREAD_LIMIT"] = "1"  # one thread per Tesseract process, as in the other scoring runs
    run_ocr, backend_class = load_backend(args.backend_dir)
    crops, charts = load_crops(args.root, args.split)
    rows = read_crops(crops, run_ocr, backend_class())
    scope = [r for r in rows if charts[r["chart"]]["scope"]]
    out = {"all_vertical": summarize(rows, charts), "scope": summarize(scope, charts),
           "charts": {"all_vertical": len(charts), "scope": sum(v["scope"] for v in charts.values())}, "crops": len(crops),
           "read": len(rows)}
    args.out.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(out, indent=1))
    (args.out / f"ocr_{args.split}_predictions.json").write_text(json.dumps(
        [{k: (str(v) if k == "image" else v) for k, v in r.items()} for r in rows]))
    print(f"{args.split}: {len(crops)} crops, {len(rows)} reads, {out['charts']}")
    for klass, m in out["all_vertical"].items():
        print(f"  {klass:<13}{m['crops']:>5}  exact raw {m['exact_raw']:.3f}  norm {m['exact_normalized']:.3f}  CER {m['cer_micro']:.3f}"
              + (f"  parse {m['parse_rate']:.3f}  correct {m['correct_parse_rate']:.3f}" if klass in NUMERIC else "")
              + (f"  charts all-correct {m['charts_all_ylabels_correct']}" if "charts_all_ylabels_correct" in m else ""))


if __name__ == "__main__":
    main()
