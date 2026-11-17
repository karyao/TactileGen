"""Gradio layout, event wiring, and callback behavior."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.contract_io import load_ocr_frame
from src.contracts import (
    Box,
    Detection,
    DetectionFrame,
    LegendBarAssignment,
    LegendMappingFrame,
    OCRFrame,
    RecognizedText,
)
from src.gradio_app import (
    AnalysisResult,
    OCR_WARNING,
    RECONSTRUCTION_WARNING,
    ReviewModel,
    UploadValidationError,
    build_review_model,
    build_app,
    review_text_rows,
)
from src.tactile.completeness import CANNOT_BUILD, READY, Completeness

from tests.gradio_helpers import GradioTestCase, make_spec


class GradioUiTest(GradioTestCase):
    def test_size_control_uses_startup_default_forwards_selection_and_resets(self):
        demo = build_app(object(), ocr_backend=object(), imgsz=960)
        try:
            binding = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
            control = next(c for c in binding.inputs if c.label == "YOLO inference size (pixels)")
            self.assertEqual(control.value, 960)
            self.assertEqual(control.maximum, 1280)
            presets = [fn for fn in demo.fns.values()
                       if fn.outputs == [control] and not fn.inputs]
            self.assertEqual([fn.fn() for fn in presets], [640, 960, 1280])
            self.assertIsNone(control.precision)  # Never round fractional input into a valid size.
            for selected in (None, 1280):
                with patch("src.gradio_ui.callbacks.analyze_image", side_effect=UploadValidationError("Choose a chart.")) as analyze:
                    args = (None, .25, .7, "eng")
                    binding.fn(*args, *(() if selected is None else (selected,)))
                self.assertEqual(analyze.call_args.kwargs["imgsz"], selected or 960)
            clear = next(fn for fn in demo.fns.values() if fn.api_name == "clear_all")
            updates = clear.fn()
            index = clear.outputs.index(control)
            self.assertEqual(updates[index]["value"], 960)
            self.assertTrue(updates[index]["interactive"])
            finish = next(fn for fn in demo.fns.values() if fn.api_name == "finish_analysis")
            self.assertIn(control, finish.outputs)
            self.assertEqual(len(finish.fn()), len(finish.outputs))
        finally:
            demo.close(verbose=False)

    def test_invalid_size_is_reported_with_diagnostics_before_inference(self):
        from zipfile import ZipFile

        demo = build_app(object(), ocr_backend=object())
        try:
            binding = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
            archive_index = next(i for i, c in enumerate(binding.outputs)
                                 if getattr(c, "label", None) == "Download analysis diagnostics (ZIP)")
            for size in (640.5, 1281, 100000):
                with self.subTest(size=size):
                    with patch("src.gradio_ui.analysis.run_detection") as detect:
                        updates = binding.fn(None, .25, .7, "eng", size)
                    detect.assert_not_called()
                    with ZipFile(updates[archive_index]["value"]) as archive:
                        record = json.loads(archive.read("diagnostics.json"))
                    self.assertEqual(record["settings"]["imgsz"], size)
                    self.assertEqual(record["stages"]["validation"]["outcome"], "failed")
                    self.assertIn("imgsz", record["exceptions"][0]["message"])
                    self.assertIsNone(record["metadata"]["detection"]["tensor_shape"])
        finally:
            demo.close(verbose=False)

    def test_creator_workspace_disclosures_preserve_result_bindings(self):
        demo = build_app(object(), device="cpu")
        config = demo.get_config_file()
        components = {item["id"]: item for item in config["components"]}
        disclosures = {
            item["props"]["label"]: item
            for item in components.values() if item["type"] == "accordion"
        }

        def descendants(node):
            result = {node["id"]}
            for child in node.get("children", []):
                result.update(descendants(child))
            return result

        def layout_node(node, target):
            if node["id"] == target:
                return node
            for child in node.get("children", []):
                found = layout_node(child, target)
                if found is not None:
                    return found
            return None

        analysis_details = disclosures["Analysis details"]
        specification = disclosures["Chart specification"]
        for disclosure in (analysis_details, specification):
            self.assertFalse(disclosure["props"]["open"])
        detail_ids = descendants(layout_node(config["layout"], analysis_details["id"]))
        spec_ids = descendants(layout_node(config["layout"], specification["id"]))
        detail_labels = {components[key]["props"].get("label") for key in detail_ids}
        self.assertTrue({"Detection summary", "Recognized text", "Legend mappings"}
                        <= detail_labels)
        self.assertTrue(any(components[key]["props"].get("label") ==
                            "Reconstructed VerticalBarSpec" for key in spec_ids))
        secondary_ids = detail_ids | spec_ids
        for component in components.values():
            classes = component["props"].get("elem_classes", [])
            if any(name in classes for name in (
                "borealis-review", "borealis-error", "borealis-result-image",
                "borealis-tactile-gallery",
            )):
                self.assertNotIn(component["id"], secondary_ids)

        analyze = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
        ready = next(fn.fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        ready_updates = ready()
        self.assertEqual(len(ready_updates), len(analyze.outputs))
        self.assertIn("Your chart workspace", ready_updates[1]["value"])
        frame = DetectionFrame("input/chart.png", 20, 20, ())
        result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 0,
            frame, None, None, spec=make_spec(), completeness=Completeness(READY),
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=result):
            updates = analyze.fn("chart.png", 0.25, 0.70, "eng")
        by_class = {
            name: update
            for output, update in zip(analyze.outputs, updates)
            for name in output.elem_classes
        }
        self.assertTrue(by_class["borealis-result-image"]["visible"])
        self.assertTrue(by_class["borealis-review"]["visible"])
        self.assertTrue(by_class["borealis-generate"]["interactive"])
        self.assertTrue(by_class["borealis-spec-json"]["visible"])
        self.assertFalse(by_class["borealis-empty"]["visible"])

        blocked = AnalysisResult(
            "partial summary", result.overlay_rgb, 0, frame, None, None,
            spec=make_spec(),
            completeness=Completeness(CANNOT_BUILD, blockers=["Ambiguous structure."]),
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=blocked):
            blocked_updates = analyze.fn("chart.png", 0.25, 0.70, "eng")
        self.assertFalse(blocked_updates[13]["interactive"])
        self.assertTrue(blocked_updates[9]["visible"])
        with patch("src.gradio_ui.callbacks.analyze_image", side_effect=UploadValidationError("Choose a chart.")):
            error_updates = analyze.fn(None, 0.25, 0.70, "eng")
        self.assertTrue(error_updates[23]["visible"])
        self.assertIn('role="alert"', error_updates[23]["value"])
        self.assertEqual(len(ready()), len(analyze.outputs))

    def test_image_change_clears_diagnostics_with_all_analysis_results(self):
        demo = build_app(object(), device="cpu")
        reset = next(fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        analyze = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
        config = demo.get_config_file()
        image = next(component for component in config["components"]
                     if component["props"].get("label") == "Bar chart image")
        event = next(dependency for dependency in config["dependencies"]
                     if any(tuple(target) == (image["id"], "change")
                            for target in dependency["targets"]))

        self.assertEqual(event["outputs"], [output._id for output in reset.outputs])
        self.assertEqual(reset.outputs, analyze.outputs)
        self.assertFalse(event["queue"])
        self.assertEqual(event["api_visibility"], "private")
        updates = reset.fn()
        self.assertEqual(len(updates), len(reset.outputs))
        self.assertEqual(reset.outputs[-2].label, "Download analysis diagnostics (ZIP)")
        self.assertIsNone(updates[-2]["value"])
        self.assertFalse(updates[-2]["visible"])
        self.assertEqual(updates[-1]["value"], "")
        self.assertFalse(updates[-1]["visible"])

    def test_app_factory_wires_controls_states_and_private_events(self):
        with patch("src.gradio_app.load_model") as load, \
             patch("gradio.Blocks.launch") as launch:
            demo = build_app(object(), device="cpu")
        load.assert_not_called()
        launch.assert_not_called()
        config = demo.get_config_file()
        components = config["components"]
        labels = {component["props"].get("label") for component in components}
        self.assertIn("Bar chart image", labels)
        self.assertIn("Minimum detection confidence", labels)
        self.assertIn("NMS IoU for duplicate suppression", labels)
        self.assertIn("OCR language", labels)
        self.assertIn("Annotated detections", labels)
        self.assertIn("Download analysis diagnostics (ZIP)", labels)
        self.assertIn("Recognized text", labels)
        self.assertIn("Legend mappings", labels)
        self.assertIn("Reconstruction completeness", labels)
        self.assertIn("Reconstructed text", labels)
        self.assertIn("Reconstructed values (read-only)", labels)
        self.assertIn("Reconstructed VerticalBarSpec", labels)
        self.assertIn("Tactile review pages", labels)
        self.assertIn("Tactile generation report", labels)

        by_class = {}
        for component in components:
            for class_name in component["props"].get("elem_classes", []):
                by_class.setdefault(class_name, component)
        for class_name in (
            "borealis-scope", "borealis-control-rail", "borealis-workspace",
            "borealis-status", "borealis-empty", "borealis-summary",
            "borealis-error",
        ):
            self.assertIn(class_name, by_class)
        self.assertTrue(by_class["borealis-empty"]["props"]["visible"])
        self.assertFalse(by_class["borealis-summary"]["props"]["visible"])
        self.assertFalse(by_class["borealis-error"]["props"]["visible"])
        annotated = next(c for c in components
                         if c["props"].get("label") == "Annotated detections")
        self.assertFalse(annotated["props"]["visible"])
        recognized = next(c for c in components
                          if c["props"].get("label") == "Recognized text")
        self.assertFalse(recognized["props"]["visible"])

        inference = [dependency for dependency in config["dependencies"]
                     if dependency.get("api_name") == "analyze"]
        self.assertEqual(len(inference), 1)
        self.assertEqual(inference[0]["api_visibility"], "private")
        self.assertEqual(inference[0]["show_progress"], "full")
        inference_fn = next(fn for fn in demo.fns.values() if fn.api_name == "analyze")
        self.assertEqual(inference_fn.concurrency_limit, 1)
        self.assertEqual(inference_fn.concurrency_id, "tactilegen-inference")

        begin = next(fn.fn for fn in demo.fns.values() if fn.api_name == "begin_analysis")
        begin_updates = begin()
        self.assertEqual(len(begin_updates), 10 + len(inference_fn.outputs))
        self.assertTrue(all(update["interactive"] is False for update in begin_updates[:10]))
        finish = next(fn.fn for fn in demo.fns.values() if fn.api_name == "finish_analysis")
        self.assertTrue(all(update["interactive"] is True for update in finish()))
        ready = next(fn.fn for fn in demo.fns.values() if fn.api_name == "ready_results")
        ready_updates = ready()
        self.assertEqual(len(ready_updates), len(inference_fn.outputs))
        self.assertTrue(ready_updates[1]["visible"])
        self.assertIsNone(ready_updates[8])
        self.assertTrue(all(not update["visible"] for update in ready_updates[2:8]))
        self.assertFalse(ready_updates[9]["visible"])
        self.assertFalse(ready_updates[12]["visible"])
        self.assertFalse(ready_updates[13]["interactive"])
        self.assertTrue(all(not update["visible"] for update in ready_updates[14:24]))
        self.assertEqual(inference_fn.outputs[24].label, "Download analysis diagnostics (ZIP)")
        self.assertIsNone(ready_updates[24]["value"])
        self.assertFalse(ready_updates[24]["visible"])
        self.assertEqual(ready_updates[25]["value"], "")
        self.assertFalse(ready_updates[25]["visible"])

        analyze = inference_fn.fn
        display_frame = DetectionFrame("input/chart.png", 2, 2, (
            Detection("title", "chart_title", Box(0, 0, 2, 1), 0.9),
        ))
        display_ocr = OCRFrame("input/chart.png", (
            RecognizedText("title", "Sales", 0.8),
        ))
        success = AnalysisResult(
            "summary", np.zeros((2, 2, 3), dtype=np.uint8), 1,
            display_frame, display_ocr, None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=success):
            success_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertIn("Analysis complete", success_updates[0]["value"])
        self.assertEqual(success_updates[2]["value"], success.summary)
        self.assertTrue(success_updates[2]["visible"])
        self.assertTrue(success_updates[3]["visible"])
        self.assertTrue(success_updates[4]["visible"])
        self.assertEqual(success_updates[4]["value"][0][:4],
                         ["title", "chart title", "Sales", 0.8])
        self.assertTrue(success_updates[5]["visible"])
        self.assertEqual(load_ocr_frame(Path(success_updates[5]["value"])), display_ocr)
        self.assertFalse(success_updates[6]["visible"])
        self.assertFalse(success_updates[7]["visible"])
        self.assertIsNone(success_updates[8])
        self.assertFalse(success_updates[9]["visible"])
        self.assertFalse(success_updates[12]["visible"])
        self.assertFalse(success_updates[13]["interactive"])
        self.assertTrue(all(not success_updates[index]["visible"]
                            for index in range(14, 24)))
        self.assertIs(success_updates[3]["value"], success.overlay_rgb)
        self.assertTrue(success_updates[24]["visible"])
        self.assertTrue(Path(success_updates[24]["value"]).is_file())
        self.assertTrue(success_updates[25]["visible"])
        self.assertIn("Analysis run:", success_updates[25]["value"])
        self.assertEqual(begin_updates[-2:], ready_updates[-2:])

        mapped_frame = DetectionFrame("input/chart.png", 20, 20, (
            Detection("bar", "vertical_bar", Box(0, 5, 5, 20), 0.9),
            Detection("swatch", "legend_swatch", Box(8, 0, 12, 4), 0.9),
            Detection("label", "legend_label", Box(13, 0, 20, 4), 0.9),
        ))
        mapped = LegendMappingFrame("input/chart.png", (
            LegendBarAssignment(
                "bar", "swatch", "label", "North", 0.98, "matched"
            ),
        ))
        reconstructed = make_spec()
        mapped_result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 3,
            mapped_frame, None, None, mapped, None,
            reconstructed, Completeness(READY), None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=mapped_result):
            mapped_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertTrue(mapped_updates[6]["visible"])
        self.assertEqual(
            mapped_updates[6]["value"][0],
            ["bar", "North", 0.98, "swatch", "label", "matched"],
        )
        self.assertTrue(mapped_updates[7]["visible"])
        self.assertIn("Reconstruction: Ready", mapped_updates[7]["value"])
        self.assertIsInstance(mapped_updates[8], ReviewModel)
        self.assertTrue(mapped_updates[9]["visible"])
        self.assertEqual(mapped_updates[10]["value"][0], ["Chart title", "Sales"])
        self.assertEqual(mapped_updates[11]["value"],
                         [[0, "2024", 0, "(unnamed)", 50.0]])
        self.assertTrue(mapped_updates[13]["interactive"])
        self.assertTrue(mapped_updates[14]["visible"])
        self.assertEqual(mapped_updates[14]["value"], asdict(reconstructed))
        self.assertTrue(mapped_updates[15]["visible"])
        spec_path = Path(mapped_updates[15]["value"])
        self.assertEqual(json.loads(spec_path.read_text(encoding="utf-8")),
                         asdict(reconstructed))
        self.assertRegex(spec_path.name, r"chart_spec_[0-9a-f]{32}\.json")

        spec_blocked_result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 3,
            mapped_frame, None, None, mapped, None, reconstructed,
            Completeness(CANNOT_BUILD, blockers=["Ambiguous structure."]), None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=spec_blocked_result):
            spec_blocked_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertIsInstance(spec_blocked_updates[8], ReviewModel)
        self.assertTrue(spec_blocked_updates[9]["visible"])
        self.assertFalse(spec_blocked_updates[13]["interactive"])
        self.assertEqual(spec_blocked_updates[11]["value"],
                         [[0, "2024", 0, "(unnamed)", 50.0]])

        blocked_ocr = OCRFrame("input/chart.png", (
            RecognizedText("label", "North", 0.9),
        ))
        blocked_result = AnalysisResult(
            "summary", np.zeros((20, 20, 3), dtype=np.uint8), 3,
            mapped_frame, blocked_ocr, None, mapped, None, None,
            Completeness(
                CANNOT_BUILD,
                blockers=["No x tick labels were found. Try a clearer image."],
            ),
            None,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=blocked_result):
            blocked_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertIn("Analysis complete", blocked_updates[0]["value"])
        for index in (2, 3, 4, 5, 6, 7):
            self.assertTrue(blocked_updates[index]["visible"])
        self.assertIn("Reconstruction: Cannot build", blocked_updates[7]["value"])
        self.assertIn("No x tick labels", blocked_updates[7]["value"])
        self.assertIsNone(blocked_updates[8])
        self.assertFalse(blocked_updates[9]["visible"])
        self.assertFalse(blocked_updates[14]["visible"])
        self.assertFalse(blocked_updates[15]["visible"])
        self.assertTrue(all(not blocked_updates[index]["visible"]
                            for index in (20, 21, 22, 23)))

        warning = AnalysisResult(
            "summary", np.zeros((2, 2, 3), dtype=np.uint8), 1,
            display_frame, None, OCR_WARNING,
            reconstruction_warning=RECONSTRUCTION_WARNING,
        )
        with patch("src.gradio_ui.callbacks.analyze_image", return_value=warning):
            warning_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertTrue(warning_updates[2]["visible"])
        self.assertTrue(warning_updates[3]["visible"])
        self.assertFalse(warning_updates[4]["visible"])
        self.assertTrue(warning_updates[20]["visible"])
        self.assertFalse(warning_updates[21]["visible"])
        self.assertTrue(warning_updates[22]["visible"])
        self.assertIn('role="alert"', warning_updates[22]["value"])
        self.assertIn("reconstruction was unavailable", warning_updates[22]["value"])
        self.assertFalse(warning_updates[23]["visible"])
        with patch(
            "src.gradio_ui.callbacks.analyze_image",
            side_effect=UploadValidationError("Upload a supported image."),
        ):
            failure_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertFalse(failure_updates[2]["visible"])
        self.assertFalse(failure_updates[3]["visible"])
        self.assertFalse(failure_updates[4]["visible"])
        self.assertTrue(failure_updates[23]["visible"])
        self.assertIn('role="alert"', failure_updates[23]["value"])

        private_path = "/private/tmp/gradio/secret-chart.png"
        with self.assertLogs("src.gradio_app", level="ERROR"), patch(
            "src.gradio_ui.callbacks.analyze_image",
            side_effect=RuntimeError(private_path),
        ):
            unexpected_updates = analyze("chart.png", 0.25, 0.70, "eng")
        self.assertTrue(unexpected_updates[23]["visible"])
        self.assertNotIn(private_path, unexpected_updates[23]["value"])
        self.assertNotIn("RuntimeError", unexpected_updates[23]["value"])

        change_events = [dependency for dependency in config["dependencies"]
                         if dependency["targets"] and dependency["targets"][0][1] == "change"]
        self.assertEqual(len(change_events), 1)
        self.assertTrue(all(not event["queue"] for event in change_events))
        self.assertEqual(change_events[0]["api_visibility"], "private")
        self.assertEqual(change_events[0]["outputs"],
                         [output._id for output in inference_fn.outputs])
        review_events = [dependency for dependency in config["dependencies"]
                         if dependency.get("api_name") == "update_review"]
        self.assertEqual(len(review_events), 1)
        self.assertEqual(review_events[0]["api_visibility"], "private")
        self.assertFalse(review_events[0]["queue"])
        review_fn = next(fn.fn for fn in demo.fns.values()
                         if fn.api_name == "update_review")
        review_model = build_review_model(make_spec(), Completeness(READY))
        invalid_review = review_fn(review_model, review_text_rows(review_model)[:-1])
        self.assertIs(invalid_review[0], review_model)
        self.assertTrue(invalid_review[2]["visible"])
        changed_rows = review_text_rows(review_model)
        changed_rows[0] = ["category[0]", "Updated title"]
        valid_review = review_fn(review_model, changed_rows)
        self.assertEqual(valid_review[0].draft_values[0], "Updated title")
        self.assertEqual(valid_review[1]["value"][0][0], "Chart title")
        self.assertFalse(valid_review[2]["visible"])

        generation_events = [
            dependency for dependency in config["dependencies"]
            if dependency.get("api_name") == "generate_tactile"
        ]
        self.assertEqual(len(generation_events), 1)
        self.assertEqual(generation_events[0]["api_visibility"], "private")
        generate_fn = next(
            fn for fn in demo.fns.values() if fn.api_name == "generate_tactile"
        )
        self.assertEqual(generate_fn.concurrency_limit, 1)
        self.assertEqual(generate_fn.concurrency_id, "tactilegen-generation")
        generated_rows = review_text_rows(review_model)
        generated_rows[0][1] = "Updated sales"
        generated = generate_fn.fn(review_model, generated_rows)
        self.assertEqual(generated[1].spec.title, "Updated sales")
        self.assertEqual(generated[2]["value"][0], ["Chart title", "Updated sales"])
        self.assertTrue(generated[5]["visible"])
        self.assertTrue(generated[7]["visible"])
        self.assertTrue(generated[8]["visible"])
        self.assertTrue(generated[9]["visible"])
        self.assertFalse(generated[10]["visible"])
        demo.close(verbose=False)


if __name__ == "__main__":
    unittest.main()
