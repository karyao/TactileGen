"""Cache one full pipeline run per PlotQA vbar test chart, for offline failure analysis.

Runs ``src.tactile.pipeline.run`` (``--yes --strict``, no overrides, non-tty) in-process
from a given source tree, so exit codes follow the real code path. For each chart it
writes ``<cache>/<id>.pkl`` holding the detections, OCR, legend mapping, reconstructed
spec (or the interpretation error), exit code, stdout/stderr and the annotation record.
Existing cache files are skipped, so the job is resumable.

usage: python scripts/audit_cache.py TREE CACHE_DIR IDFILE WORKERS
  TREE   source tree to import ``src`` from (e.g. a worktree at commit 2412bb2e); cwd is set to it
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
import traceback
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

ROOT = Path.home() / "lsi-code/TactileGen"
PNG = ROOT / "data/png_test/png"
ANNOTATIONS = ROOT / "data/annotations_test.json"


def _worker(args):
    tree, cache, ids, shard = args
    os.chdir(tree)
    sys.path.insert(0, str(tree))
    import cv2

    from src.contracts import ChartInterpretationError
    from src.detection import load_model, run_detection
    from src.legend_matcher import match_legend_to_bars
    from src.ocr import run_ocr
    from src.reconstruct import reconstruct_vbar
    from src.tactile import pipeline
    from src.tactile.completeness import cut_off_flags
    from src.tesseract_ocr import TesseractBackend

    model, backend = load_model(pipeline.DEFAULT_WEIGHTS), TesseractBackend()
    records = {str(r["image_index"]): r for r in json.loads(ANNOTATIONS.read_text())
               if r["type"] == "vbar_categorical"}
    tmp = Path(tempfile.mkdtemp(prefix=f"audit{shard}_"))
    for cid in ids:
        target = cache / f"{cid}.pkl"
        if target.exists():
            continue
        captured: dict = {}

        def analyse(image_path, weights):
            # Same steps as pipeline.analyse, but keeping the intermediates.
            image = cv2.imread(str(image_path))
            t = time.time()
            frame = run_detection(model, image_path)
            captured["frame"], captured["t_detect"] = frame, time.time() - t
            ocr = run_ocr(image, frame, backend, "eng")
            captured["ocr"] = ocr
            try:
                mapping = match_legend_to_bars(image, frame, ocr)
                captured["legend_mapping"] = mapping
                spec = reconstruct_vbar(frame, ocr, image, legend_mapping=mapping)
                spec = replace(spec, flags=[*spec.flags, *cut_off_flags(frame, image)])
                captured["spec"] = spec
                return spec, None, frame
            except ChartInterpretationError as error:
                captured["error"] = str(error)
                return None, error, frame

        ns = Namespace(image=PNG / f"{cid}.png", output=tmp / f"{cid}.svg", weights=pipeline.DEFAULT_WEIGHTS,
                       title=None, x_title=None, y_title=None, category=[], series=[], overrides=None,
                       yes=True, force=False, strict=True, show_print=False, x_labels="auto")
        out, err = io.StringIO(), io.StringIO()
        t0 = time.time()
        try:
            code = pipeline.run(ns, analyse_fn=analyse, stdin_is_tty=False, out=out, err=err)
        except Exception:  # a crash in the old run was exit 1 with a traceback
            code = 1
            err.write(traceback.format_exc())
        row = {"id": cid, "exit": code, "seconds": round(time.time() - t0, 2),
               "stdout": out.getvalue(), "stderr": err.getvalue(),
               "record": records[cid], **captured}
        for f in tmp.glob("*"):
            f.unlink()
        part = target.with_suffix(".tmp")
        part.write_bytes(pickle.dumps(row))
        part.rename(target)


def main():
    tree, cache, idfile, workers = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    cache.mkdir(parents=True, exist_ok=True)
    ids = [str(i) for i in json.load(open(idfile))]
    shards = [(tree, cache, ids[k::workers], k) for k in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        pool.map(_worker, shards, chunksize=1)


if __name__ == "__main__":
    main()
