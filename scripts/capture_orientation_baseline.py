"""Author fixtures or capture the unchanged orientation pipeline.

Run from the repository root; see docs/bar_orientation_baseline.md.
No detections are synthesized, relabelled, or filtered by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


FIXTURES = Path("tests/fixtures/orientation")
OUTPUT = Path("output/orientation_baseline")
WEIGHTS = Path("models/yolo_bars_hv/best.pt")
SETTINGS = {"confidence": 0.25, "nms_iou": 0.70, "imgsz": 640,
            "device": "cpu", "ocr_engine": "rapidocr", "language": "en"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def author_chart(path, orientation, *, bordered=False, names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12}):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        values = (40, 70, 100)
        names = names or ["Alpha", "Beta", "Gamma"]
        if orientation == "vertical":
            ax.bar(names, values, width=0.6, color="#477bad")
            ax.set(xlabel="Class", ylabel="Books", ylim=(0, 120))
            ax.set_yticks(range(0, 121, 20))
        else:
            ax.barh(names, values, height=0.6, color="#477bad")
            ax.invert_yaxis()
            ax.set(xlabel="Books", ylabel="Class", xlim=(0, 120))
            ax.set_xticks(range(0, 121, 20))
        ax.set_title("Books by class", pad=16)
        ax.spines[["top", "right"]].set_visible(bordered)
        fig.tight_layout(pad=2)
        fig.savefig(path, metadata={"Software": "TactileGen orientation fixture"})
        plt.close(fig)


def prepare():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for orientation in ("vertical", "horizontal"):
        author_chart(FIXTURES / f"clean_{orientation}.png", orientation)
    author_chart(FIXTURES / "mixed_predictions.png", "horizontal", bordered=True,
                 names=["Class A", "Class B", "Class C"])
    # Preserve the user's original image; the snapshot makes the corpus portable.
    target = FIXTURES / "stacked_horizontal.png"
    if not target.exists():
        shutil.copyfile("images.png", target)


def bar_evidence(frame):
    from src.reconstruct_hbar import bar_orientation

    bars = [d for d in frame.detections
            if d.role in ("vertical_bar", "horizontal_bar")]
    return {
        "counts": dict(Counter(d.role for d in bars)),
        "counts_at_0_30": dict(Counter(d.role for d in bars if d.confidence >= 0.30)),
        "confidence_sums_at_0_30": {
            role: sum(d.confidence for d in bars if d.role == role and d.confidence >= 0.30)
            for role in ("vertical_bar", "horizontal_bar")
        },
        "bars": [{"id": d.id, "role": d.role, "confidence": d.confidence} for d in bars],
        "current_helper_decision": bar_orientation(frame),
    }


def capture():
    import cv2
    from src.contract_io import load_detection_frame, write_detection_frame
    from src.detection import load_model
    from src.diagnostics import AnalysisDiagnostics
    from src.gradio_ui.analysis import analyze_image
    from src.rapidocr_ocr import RapidOCRBackend
    from src.tactile.pipeline import analyse, parse_args, run

    cases = json.loads((FIXTURES / "cases.json").read_text())
    model = load_model(WEIGHTS)
    backend = RapidOCRBackend()
    capture_info = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], text=True),
        "python": platform.python_version(), "platform": platform.platform(),
        "weights": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS), "settings": SETTINGS,
        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "ocr_models_sha256": {
            path.name: sha256(path)
            for path in Path(metadata.distribution("rapidocr").locate_file("rapidocr/models")).glob("*.onnx")
        },
        "cases": [],
    }
    for case in cases:
        name = case["id"]
        image_path = Path(case["image"])
        destination = OUTPUT / name
        destination.mkdir(parents=True, exist_ok=True)
        diagnostics = AnalysisDiagnostics(settings=SETTINGS)
        diagnostics.capture_metadata(model, backend, "en")
        result = analyze_image(model, image_path, 0.25, 0.70, imgsz=640, device="cpu",
                               ocr_backend=backend, language="en", diagnostics=diagnostics)
        diagnostics.finish()
        diagnostics.persist(destination)
        archive = diagnostics.export_zip(destination)
        if archive is None:
            raise RuntimeError(diagnostics.export_error)
        frame = result.detection_frame
        assert diagnostics.record["outputs"]["detections"] == asdict(frame)
        frame_path = FIXTURES / f"{name}.detections.json"
        write_detection_frame(frame_path, frame)
        assert load_detection_frame(frame_path) == frame
        pixels = cv2.imread(str(image_path))
        assert pixels.shape[:2] == (frame.height, frame.width)
        assert len({d.id for d in frame.detections}) == len(frame.detections)
        if not cv2.imwrite(str(destination / "annotated.png"),
                           cv2.cvtColor(result.overlay_rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Could not write overlay")

        # Exercise the public CLI runner without redirecting its detector/backend.
        # Its callback is the real analyse(); retain that result for comparison.
        cli_observation = {}

        def observe_cli(*args, **kwargs):
            spec, error, cli_frame = analyse(*args, ocr_engine="rapidocr", **kwargs)
            cli_observation.update(
                spec=asdict(spec) if spec is not None else None,
                error=str(error) if error is not None else None,
                raw_frame_equal=cli_frame == frame,
                bar_evidence=bar_evidence(cli_frame),
            )
            write_detection_frame(destination / "cli.detections.json", cli_frame)
            return spec, error, cli_frame

        args = parse_args(["--image", str(image_path), "--weights", str(WEIGHTS),
                           "--ocr", "rapidocr", "--yes",
                           "--output", str(destination / "cli.svg")])
        with (destination / "cli.stdout.txt").open("w") as out, \
                (destination / "cli.stderr.txt").open("w") as err:
            code = run(args, analyse_fn=observe_cli, stdin_is_tty=False, out=out, err=err)
        cli_observation["exit_code"] = code
        observation = {
            "id": name, "image_sha256": sha256(image_path),
            "image_size": [frame.width, frame.height],
            "detections_sha256": sha256(frame_path),
            "bar_evidence": bar_evidence(frame),
            "inference": diagnostics.record["metadata"]["detection"],
            "model": diagnostics.record["metadata"]["model"],
            "app": {"spec": asdict(result.spec) if result.spec is not None else None,
                    "completeness": asdict(result.completeness) if result.completeness else None,
                    "ocr_warning": result.ocr_warning, "legend_warning": result.legend_warning,
                    "reconstruction_warning": result.reconstruction_warning,
                    "summary": result.summary},
            "cli": cli_observation,
            "artifacts": str(destination),
            "diagnostics_archive": str(archive),
        }
        capture_info["cases"].append(observation)
        print(name, json.dumps(observation["bar_evidence"]["counts"]),
              "app:", result.completeness.status if result.completeness else "no spec",
              "cli:", code, flush=True)
    write_json(FIXTURES / "capture.json", capture_info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "capture"))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((OUTPUT / "matplotlib").resolve()))
    os.environ.setdefault("YOLO_CONFIG_DIR", str((OUTPUT / "ultralytics").resolve()))
    {"prepare": prepare, "capture": capture}[args.action]()


if __name__ == "__main__":
    main()
