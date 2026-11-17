# Add tactile SVG generation to Gradio

## Summary

Extend the existing Gradio analysis flow through reconstruction, tactile layout, validation, preview, and download. Successful analyses will automatically show sighted-review previews and provide clean, embosser-importable SVG pages in a ZIP.

## Implementation Changes

- Extend `src/gradio_app.py` to reconstruct a `VerticalBarSpec` from the existing detection, OCR, and legend results, including image-edge completeness checks.
- Preserve the current detection/OCR results when tactile reconstruction fails; show a plain-language blocker instead of failing the entire analysis.
- Add a guided review section:
  - Editable title, axis titles, categories, and series names.
  - Read-only reconstructed series/category values.
  - Stable internal field mapping so edited display labels cannot alter which specification field is updated.
  - Automatic generation when the reconstructed specification is already complete.
  - A “Generate tactile graphic” action for applying corrections and regenerating.
- Add tactile outputs:
  - A gallery containing every page as a sighted-review SVG with print labels and highlighted rule findings.
  - A clean SVG for each page containing only embossable geometry.
  - A ZIP download containing all clean pages in order.
  - Notes for inferred values, automated layout violations, and multi-page charts requiring a transcriber-created overview.
- Block tactile generation for uncertain numeric data, structural ambiguity, unreadable required text, or unsupported braille characters. Do not expose a “force ambiguity” option.
- Generate files in UUID-scoped temporary directories and sanitize source-derived filenames.
- Update the Gradio heading, scope text, progress messages, reset behavior, and accessibility statuses to describe the complete image-to-tactile workflow.
- Keep the existing tactile CLI and rendering contracts unchanged; reuse `reconstruct_vbar`, completeness/override helpers, paginated layout, rule checking, and `render_svg`.

## Interfaces and Behavior

- Extend the internal analysis result with the reconstructed specification and its completeness result; no new public HTTP API or CLI flags.
- Use an interactive Gradio dataframe with only the text column editable and a separate read-only values table.
- A ready chart generates immediately. A chart needing text remains editable until all required fields pass completeness checks.
- Layout findings do not block download. The preview highlights them, and the download area prominently states that transcriber review is required.
- The review SVG must contain the existing “do not emboss” print overlay; ZIP contents must omit print labels and violation overlays.

## Test Plan

- Verify ready, needs-input, and cannot-build reconstruction paths without regressing detection/OCR output.
- Verify text corrections, unchanged confirmations for cut-off labels, blank fields, unsupported characters, and row-shape validation.
- Verify inferred values remain read-only and numeric overrides are impossible.
- Verify single-page and multi-page generation, page ordering, sanitized filenames, and ZIP contents.
- Verify review SVGs contain print labels and violation annotations while embossing SVGs contain neither.
- Verify rule findings and missing-overview warnings remain downloadable but visibly require review.
- Verify Clear resets draft state, correction tables, previews, warnings, and downloads.
- Update Gradio wiring/configuration tests and run `tests.test_gradio_app`, tactile pipeline/layout tests, and a local smoke test with a real validation chart.

## Assumptions

- “Embosser ready” means the repository’s clean SVG format, which can be imported into graphics-embosser software or printed on swell paper; native embosser formats, BRF, and PDF remain out of scope.
- The current vertical categorical bar-chart limitation remains unchanged.
- Automated validation supports review but does not claim full BANA compliance or replace a certified tactile-graphics transcriber.
