"""Lossless evidence at the source-crop and recognition API boundaries."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import cv2
import numpy as np

from src.contracts import Box, Detection, DetectionFrame
from src.diagnostics import AnalysisDiagnostics
from src.ocr import run_ocr
from src.ocr_crop import crop_text_bounds, crop_text_region
from src.rapidocr_ocr import RapidOCRBackend
from tests import test_ocr_trace as helpers
from tests.test_ocr_trace import SHORT, TALL, IMAGE, tesseract_data
from tests.test_rapidocr_ocr import fake_rapidocr, output


def decode(diagnostics, evidence):
    return cv2.imdecode(np.frombuffer(diagnostics.assets[evidence['filename']], np.uint8),
                        cv2.IMREAD_UNCHANGED)


class OCRImageEvidenceTest(unittest.TestCase):
    def test_source_coordinates_pixels_and_app_collection_is_idempotent(self):
        title = Detection('../../title', 'chart_title', Box(25, 3, 35, 12), .9)
        neighbor = Detection('neighbor', 'legend_label', Box(55, 2, 59, 14), .9)
        edge = Detection('edge', 'x_tick_label', Box(.2, 34, 5.1, 39.9), .9)
        frame = DetectionFrame('input/chart.png', 60, 40, (title, neighbor, edge, TALL))
        diagnostics = AnalysisDiagnostics(settings={})
        diagnostics.source_images(IMAGE, frame)
        backend = types.SimpleNamespace(recognize=lambda *args, **kwargs: None)
        run_ocr(IMAGE, frame, backend, 'eng', diagnostics=diagnostics)
        sources = diagnostics.record['ocr_source_crops']
        self.assertEqual(len(sources), 4)
        self.assertEqual(sources[0]['crop_policy'], 'neighbor_aware_title_strip')
        self.assertEqual(sources[2]['crop_bounds'], [0, 32, 8, 40])
        for source, detection in zip(sources, frame.detections):
            bounds = crop_text_bounds(IMAGE, detection, frame.detections)
            self.assertEqual(source['crop_bounds'], list(bounds))
            left, top, right, bottom = bounds
            np.testing.assert_array_equal(decode(diagnostics, source['image']),
                                          IMAGE[top:bottom, left:right])
            np.testing.assert_array_equal(crop_text_region(IMAGE, detection, frame.detections),
                                          IMAGE[top:bottom, left:right])
            self.assertNotIn('..', source['image']['filename'])
            self.assertEqual(source['detection_box']['x1'], detection.box.x1)

    def test_rapid_images_are_after_classifier_rotation(self):
        module, engine, _ = fake_rapidocr([output()] * 4)
        def classify(images):
            oriented = [cv2.rotate(image, cv2.ROTATE_180) for image in images]
            return oriented, types.SimpleNamespace(cls_res=[('180', .99)] * len(images),
                                                   elapse=.01)
        engine.cls_and_rotate = Mock(side_effect=classify)
        diagnostics = AnalysisDiagnostics(settings={})
        with patch('src.rapidocr_ocr._load_rapidocr', return_value=module):
            backend = RapidOCRBackend()
        frame = DetectionFrame('input/chart.png', 60, 40, (SHORT, TALL))
        run_ocr(IMAGE, frame, backend, 'eng', diagnostics=diagnostics)
        attempts = diagnostics.record['ocr_trace']['attempts']
        self.assertEqual(len(attempts), 4)
        for attempt, submitted in zip(attempts, engine.rec_calls[0]):
            np.testing.assert_array_equal(decode(diagnostics, attempt['recognition_image']), submitted)
            self.assertEqual(attempt['preprocessing']['classifier_orientation'], ['180', .99])
            self.assertIsNotNone(attempt['source_crop_id'])
        self.assertEqual(engine.cls_and_rotate.call_count, 1)
        self.assertEqual(len(engine.rec_calls), 1)
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics.finish()
            path = diagnostics.export_zip(Path(temporary))
            with ZipFile(path) as archive:
                record = json.loads(archive.read('diagnostics.json'))
                self.assertNotIn('input.png', archive.namelist())
                for attempt in record['ocr_trace']['attempts']:
                    self.assertIn(attempt['recognition_image']['filename'], archive.namelist())

    def test_tesseract_pngs_equal_actual_multipage_tiff_pixels(self):
        diagnostics = AnalysisDiagnostics(settings={})
        raw = tesseract_data([[('label', '90')]] * 8)
        _, module = helpers.OCRTraceTest().tess(raw, (SHORT, TALL), diagnostics=diagnostics)
        attempts = diagnostics.record['ocr_trace']['attempts']
        self.assertEqual(len(attempts), 8)
        for attempt, submitted in zip(attempts, module.submitted_pages):
            np.testing.assert_array_equal(decode(diagnostics, attempt['recognition_image']), submitted)
            self.assertEqual(attempt['recognition_image']['channel_layout'], 'GRAY')
        self.assertEqual({a['preprocessing']['variant'] for a in attempts},
                         {'grayscale', 'otsu_binary'})
        self.assertEqual(module.image_to_data.call_count, 1)

    def test_encoding_failure_retains_result_and_explicit_unavailability(self):
        diagnostics = AnalysisDiagnostics(settings={})
        helper = helpers.OCRTraceTest()
        with patch('cv2.imencode', side_effect=RuntimeError('encoder unavailable')):
            result, _ = helper.rapid(output('label'), diagnostics=diagnostics)
        self.assertEqual(result.regions[0].text, 'label')
        attempt = diagnostics.record['ocr_trace']['attempts'][0]
        self.assertEqual(attempt['recognition_image']['availability'], 'unavailable')
        self.assertIsNone(attempt['recognition_image']['filename'])
        self.assertEqual(diagnostics.assets, {})
        self.assertTrue(diagnostics.record['exceptions'])

    def test_failures_retain_prepared_images_without_inventing_submission(self):
        helper = helpers.OCRTraceTest()
        diagnostics = AnalysisDiagnostics(settings={})
        with self.assertRaisesRegex(RuntimeError, 'classification failed'):
            helper.rapid(output(), diagnostics=diagnostics,
                         cls_error=RuntimeError('classification failed'))
        attempt = diagnostics.record['ocr_trace']['attempts'][0]
        self.assertEqual(attempt['recognition_image']['availability'], 'unavailable')
        self.assertEqual(attempt['recognition_submission'], 'not_submitted')
        self.assertEqual(len(diagnostics.record['ocr_source_crops']), 1)
        diagnostics = AnalysisDiagnostics(settings={})
        with self.assertRaisesRegex(RuntimeError, 'recognition failed'):
            helper.rapid(RuntimeError('recognition failed'), diagnostics=diagnostics)
        attempt = diagnostics.record['ocr_trace']['attempts'][0]
        self.assertEqual(attempt['recognition_image']['availability'], 'available')
        self.assertEqual(attempt['recognition_submission'], 'submitted')

    def test_repeated_calls_and_runs_have_distinct_input_paths(self):
        helper = helpers.OCRTraceTest()
        diagnostics = AnalysisDiagnostics(settings={})
        helper.rapid(output(), diagnostics=diagnostics)
        helper.rapid(output(), diagnostics=diagnostics)
        attempts = diagnostics.record['ocr_trace']['attempts']
        paths = [a['recognition_image']['filename'] for a in attempts]
        self.assertEqual(len(set(paths)), 2)
        other = AnalysisDiagnostics(settings={})
        self.assertEqual(other.assets, {})
        self.assertNotEqual(diagnostics.run_id, other.run_id)

    def test_grayscale_sources_and_unavailable_classifier_metadata(self):
        gray = IMAGE[:, :, 0].copy()
        module, engine, _ = fake_rapidocr([output()])
        with patch('src.rapidocr_ocr._load_rapidocr', return_value=module):
            backend = RapidOCRBackend()
        diagnostics = AnalysisDiagnostics(settings={})
        frame = DetectionFrame('input/gray.png', 60, 40, (SHORT,))
        run_ocr(gray, frame, backend, 'eng', diagnostics=diagnostics)
        source = diagnostics.record['ocr_source_crops'][0]
        np.testing.assert_array_equal(decode(diagnostics, source['image']),
                                      crop_text_region(gray, SHORT, (SHORT,)))
        self.assertEqual(source['image']['channel_layout'], 'GRAY')
        trace = diagnostics.record['ocr_trace']
        self.assertEqual(trace['batches'][0]['classifier_output']['availability'], 'unavailable')
        self.assertEqual(trace['attempts'][0]['preprocessing']['classifier_orientation'], 'unavailable')
        np.testing.assert_array_equal(decode(diagnostics, trace['attempts'][0]['recognition_image']),
                                      engine.rec_calls[0][0])

    def test_unsupported_png_dtype_is_explicit_without_lossy_conversion(self):
        diagnostics = AnalysisDiagnostics(settings={})
        evidence = diagnostics.image_evidence('test.png', IMAGE.astype(np.float32),
                                              channel_layout='BGR')
        self.assertEqual(evidence['availability'], 'unavailable')
        self.assertEqual(evidence['dtype'], 'float32')
        self.assertIsNone(evidence['filename'])
        self.assertEqual(diagnostics.assets, {})


if __name__ == '__main__':
    unittest.main()
