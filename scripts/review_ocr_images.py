"""Generate the Plan 3 checkpoint with deterministic fake-engine evidence.

Run from the repository root: .venv/bin/python -m scripts.review_ocr_images
No models are loaded and no real recognition accuracy is claimed.
"""

import base64
import html
import json
from pathlib import Path
import types
from unittest.mock import patch

import cv2
import numpy as np

from src.contracts import Box, Detection, DetectionFrame
from src.diagnostics import AnalysisDiagnostics
from src.ocr import run_ocr
from src.rapidocr_ocr import RapidOCRBackend
from src.tesseract_ocr import TesseractBackend
from tests.test_ocr import fake_rapidocr, output, fake_pytesseract, tesseract_data


def gallery(diagnostics):
    def escaped(value):
        return html.escape(json.dumps(value, ensure_ascii=False, indent=2))

    def picture(evidence):
        if evidence['availability'] != 'available':
            return '<pre>' + escaped(evidence) + '</pre>'
        data = base64.b64encode(diagnostics.assets[evidence['filename']]).decode()
        return f'<img alt="{html.escape(evidence["filename"])}" src="data:image/png;base64,{data}">'

    sections = ['<!doctype html><meta charset="utf-8"><title>OCR Plan 3 review</title>',
                '<style>body{font:16px system-ui;margin:32px;max-width:1200px}'
                'article{border-top:1px solid #aaa;margin-top:24px;padding-top:16px}'
                'img{max-width:100%;border:1px solid #bbb;background:#eee}'
                'pre{white-space:pre-wrap}section{padding:16px;background:#f4f6f8;margin:12px 0}'
                '</style><h1>OCR image evidence — Plan 3</h1>',
                '<p>Deterministic fake engines. Text results are fixtures, not accuracy measurements. '
                'Coordinates are source pixels; right/bottom bounds are exclusive.</p>',
                '<p>Run: ' + html.escape(diagnostics.run_id) + '</p>']
    for source in diagnostics.record['ocr_source_crops']:
        sections += ['<article><h2>' + html.escape(source['detection_id']) + '</h2>',
                     picture(source['image']), '<pre>' + escaped(source) + '</pre>']
        for attempt in diagnostics.record['ocr_trace']['attempts']:
            if attempt['source_crop_id'] != source['source_crop_id']:
                continue
            sections += ['<section><h3>' + html.escape(attempt['attempt_id']) + '</h3>',
                         picture(attempt['recognition_image']),
                         '<pre>' + escaped(attempt) + '</pre></section>']
        sections.append('</article>')
    return '\n'.join(sections)


def main():
    root = Path('output/ocr-plan3-review')
    root.mkdir(parents=True, exist_ok=True)
    image = np.full((160, 320, 3), 255, np.uint8)
    cv2.putText(image, 'Sales 2026', (75, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 40, 80), 1)
    cv2.putText(image, 'Key', (260, 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.putText(image, '10', (0, 152), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    label = np.full((24, 85, 3), 255, np.uint8)
    cv2.putText(label, 'Revenue', (1, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    image[48:133, 30:54] = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
    detections = (
        Detection('title', 'chart_title', Box(88, 10, 170, 30), .9),
        Detection('legend', 'legend_label', Box(260, 12, 295, 30), .9),
        Detection('edge', 'y_tick_label', Box(0, 137, 20, 159), .9),
        Detection('vertical', 'y_axis_title', Box(30, 48, 54, 133), .9),
    )
    frame = DetectionFrame('input/review.png', 320, 160, detections)
    for backend_name in ('rapidocr', 'tesseract'):
        diagnostics = AnalysisDiagnostics(settings={'fixture': True})
        destination = root / backend_name
        destination.mkdir(exist_ok=True)
        if backend_name == 'rapidocr':
            module, engine, _ = fake_rapidocr([output('fixture label', .9)] * 6)
            def classify(images):
                return ([cv2.rotate(candidate, cv2.ROTATE_180) for candidate in images],
                        types.SimpleNamespace(cls_res=[('180', .99)] * len(images), elapse=0.0))
            engine.cls_and_rotate = classify
            with patch('src.rapidocr_ocr._load_rapidocr', return_value=module):
                backend = RapidOCRBackend()
            run_ocr(image, frame, backend, 'eng', diagnostics=diagnostics)
            submitted = engine.rec_calls[0]
        else:
            module = fake_pytesseract(results=[])
            submitted = []
            def recognize(path, **kwargs):
                from PIL import Image, ImageSequence
                with Image.open(path) as tiff:
                    submitted.extend(np.array(page) for page in ImageSequence.Iterator(tiff))
                return tesseract_data([[('fixture label', '90')]] * len(submitted))
            module.image_to_data.side_effect = recognize
            with patch('src.tesseract_ocr._load_pytesseract', return_value=module):
                run_ocr(image, frame, TesseractBackend(), 'eng', diagnostics=diagnostics)
        attempts = diagnostics.record['ocr_trace']['attempts']
        assert len(submitted) == len(attempts)
        for attempt, candidate in zip(attempts, submitted, strict=True):
            saved = diagnostics.assets[attempt['recognition_image']['filename']]
            decoded = cv2.imdecode(np.frombuffer(saved, np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, candidate)
        diagnostics.finish()
        record_path = diagnostics.persist(destination)
        archive_path = diagnostics.export_zip(destination)
        diagnostics.log_summary(record_path=record_path, archive_path=archive_path)
        (destination / 'gallery.html').write_text(gallery(diagnostics), encoding='utf-8')
    (root / 'REVIEW.md').write_text(
        '# Plan 3 review checkpoint\n\n'
        'These are fake-engine fixtures, not real-engine accuracy results.\n\n'
        '- [RapidOCR gallery](rapidocr/gallery.html): 4 source crops, 6 recognition inputs; '
        'includes classifier-applied 180° rotation after adapter rotations.\n'
        '- [Tesseract gallery](tesseract/gallery.html): 4 source crops, 12 recognition inputs; '
        'includes grayscale, Otsu, upscaling, and rotated labels.\n\n'
        'All saved recognition PNGs were decoded and compared pixel-for-pixel with '
        'the submitted RapidOCR arrays or actual Tesseract TIFF pages. '
        'Each backend directory also contains diagnostic JSON and ZIP evidence.\n', encoding='utf-8')
    print(root / 'REVIEW.md')


if __name__ == '__main__':
    main()
