"""Write uniquely named JSON downloads into the application artifact directory."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path

from src.contract_io import write_ocr_frame
from src.contracts import OCRFrame, VerticalBarSpec


def write_ocr_download(root: Path, image_path: str | Path, ocr: OCRFrame) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(image_path).stem).strip("._-")
    path = root / f"{stem or 'chart'}_ocr_{uuid.uuid4().hex}.json"
    write_ocr_frame(path, ocr)
    return path


def write_spec_download(
    root: Path, image_path: str | Path, spec: VerticalBarSpec
) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(image_path).stem).strip("._-")
    path = root / f"{stem or 'chart'}_spec_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(asdict(spec), indent=2) + "\n", encoding="utf-8")
    return path
