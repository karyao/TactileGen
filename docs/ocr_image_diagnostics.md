# OCR image diagnostics (Plan 3)

For links from parsing and completeness findings to this evidence, see
[OCR warning provenance (Plan 4)](ocr_warning_provenance.md).

Diagnostics capture source crops and the actual image inputs used by both built-in
recognizers. Capture uses the original preprocessing pass; it adds no inference.
`OCRFrame`, chart contracts, backend signatures, and winner selection are unchanged.

`crop_text_bounds(image, detection, neighbors)` is the shared crop authority.
`crop_text_region` slices those bounds. Coordinates are source-image pixels with
top-left origin and exclusive right/bottom bounds. Titles retain the existing
neighbor-aware strip policy; other detections retain their padded boxes.

## Evidence records

`diagnostics.json` has an additive `ocr_source_crops` collection. Each record has
a run-scoped `source_crop_id`, run and detection IDs, the original `detection_box`,
actual integer `crop_bounds`, crop policy, and an `image` descriptor. Collection
works from both the app and direct `run_ocr(..., diagnostics=collector)` calls.
Collecting the same image object and detection frame twice reuses the source
records; additional attempts still get their own input images.

Each existing `ocr_trace.attempts` entry adds `source_crop_id` and a
`recognition_image` descriptor. These descriptors record availability, PNG path,
width, height, original array shape, dtype, and channel layout. Generated numeric
paths live under `ocr_crops/` and `ocr_inputs/`; user-provided detection IDs never
enter filenames. Paths are relative to the existing diagnostic ZIP root.

- RapidOCR captures BGR images returned by `cls_and_rotate`, before
  `recognize_txt`. The batch's `classifier_output` explicitly snapshots available
  `cls_res` and elapsed time. Each attempt retains its classifier label/score in
  `preprocessing.classifier_orientation`, independently of the adapter's requested
  rotation. A classifier prediction alone is not proof that a rotation was applied.
- Tesseract captures grayscale/Otsu candidates used to create the lossless
  multipage TIFF. `tiff_page_number` links each attempt to its submitted page;
  rotation, variant, and upscale factor remain in `preprocessing`.

Engine input means the recognition API image or TIFF page, before internal engine
resizing/normalization. The full uploaded `input.png` is omitted; the existing
annotated overlay remains available. Custom backends retain their existing
unavailable-internal-attempt behavior and still receive source-crop evidence.

Capture errors record unavailable evidence and exceptions without replacing OCR
results. Prepared images remain available even if recognition later fails.
Classification failure leaves final recognition images unavailable. Submission
state remains separate from image availability. PNG capture accepts uint8/uint16
pixels; unsupported dtypes are explicitly unavailable instead of silently converted.

## Validation and checkpoint

```sh
.venv/bin/python -m pytest tests/test_ocr_image_evidence.py tests/test_ocr_crop.py tests/test_ocr_trace.py tests/test_ocr.py tests/test_rapidocr_ocr.py tests/test_tesseract_ocr.py tests/test_diagnostics.py tests/test_analysis_evidence.py -q
.venv/bin/python -m scripts.review_ocr_images
```

The tests decode saved PNGs and compare them with submitted RapidOCR arrays and
actual Tesseract TIFF pages. They cover source coordinates, image-edge clipping,
neighbor-aware titles, preprocessing variants, rotations, failures, run isolation,
generated paths, and JSON/ZIP links. Existing tests cover diagnostics-on/off parity.

The generator creates [the review checkpoint](../output/ocr-plan3-review/REVIEW.md)
and separate self-contained HTML galleries for each backend, with source bounds,
all input variants, attempt descriptions, fixture output, and selection state.
It verifies all submitted input pixels before writing the checkpoint. These are
deterministic fake-engine fixtures, not real-engine accuracy measurements.
Artifacts live under ignored `output/`; regenerate them with the command above.

Plan 3 ends at review of this gallery. Final report formatting and download UI
changes remain deferred to Plan 5.
