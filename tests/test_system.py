import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "main"))

import derma_agent as project
import run as launcher


class MetricsTests(unittest.TestCase):
    def test_perfect_metrics(self):
        labels = np.arange(7)
        probabilities = np.eye(7) * 0.94 + (1 - np.eye(7)) * 0.01
        result = project.classification_metrics(labels, probabilities)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["confusion_matrix"], np.eye(7, dtype=int).tolist())

    def test_auc_handles_ties(self):
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(project.binary_auc(labels, np.ones(4)), 0.5)
        self.assertAlmostEqual(project.binary_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)

    def test_uncertainty_threshold_is_observed_and_bounded(self):
        labels = np.array([0, 1, 2, 3])
        probabilities = np.array([
            [0.8, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05],
            [0.4, 0.1, 0.2, 0.1, 0.1, 0.05, 0.05],
            [0.2, 0.1, 0.1, 0.25, 0.15, 0.1, 0.1],
        ])
        result = project.choose_uncertainty_threshold(labels, probabilities)
        self.assertGreaterEqual(result["threshold"], 0)
        self.assertLessEqual(result["threshold"], 1)
        self.assertGreaterEqual(result["validation_error_detection_f1"], 0)


class ModelAndDataTests(unittest.TestCase):
    def test_all_approved_models_return_seven_logits(self):
        for name in project.load_config()["training"]["models"]:
            model = project.make_model(name)
            output = model(torch.zeros(2, 3, 28, 28))
            self.assertEqual(tuple(output.shape), (2, 7), name)
            self.assertGreater(project.count_parameters(model), 0)

    def test_stratified_subset_is_repeatable_and_contains_every_class(self):
        labels = np.repeat(np.arange(7), [50, 40, 30, 20, 10, 8, 5])
        first = project.stratified_indices(labels, 70, 42)
        second = project.stratified_indices(labels, 70, 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 70)
        self.assertEqual(set(labels[first]), set(range(7)))

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            project.make_model("unapproved_model")

    def test_invalid_image_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.png"
            path.write_text("not an image", encoding="utf-8")
            with self.assertRaises(ValueError):
                project.validate_input_image(path)


class AgentBoundaryTests(unittest.TestCase):
    def test_agent_names_are_agent1_to_agent3(self):
        self.assertEqual(set(project.AGENT_SPECS), {"Agent1", "Agent2", "Agent3"})
        self.assertEqual(project.TrainingAgent.name, "Agent1")
        self.assertEqual(project.EvaluationAgent.name, "Agent2")
        self.assertEqual(project.ReviewAgent.name, "Agent3")

    def test_agent_rejects_unapproved_tool(self):
        with patch.object(project, "append_event"):
            with self.assertRaises(PermissionError):
                project.TrainingAgent().call("open_test_labels", lambda: None)

    def test_explanation_preserves_probabilities_and_trace(self):
        prediction = {
            "trace_id": "trace-test", "sample_id": "test-00000", "predicted_class": "melanoma",
            "confidence": 0.7, "uncertain": False, "probabilities": {str(i): value for i, value in enumerate([0.05, 0.05, 0.05, 0.05, 0.7, 0.05, 0.05])},
        }
        xai = {"trace_id": "trace-test", "overlay_file": "attribution.png"}
        enrichment = {"trace_id": "trace-test", "fact": "Cited fact.", "citations": [{"title": "Source", "url": "https://example.org"}]}
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "case.json"
            with patch.object(project, "deliverable_path", return_value=target), patch.object(project, "append_event"):
                report = project.ExplanationAgent().run(prediction, xai, enrichment)
            self.assertEqual(report["model_evidence"]["probabilities"], prediction["probabilities"])
            self.assertEqual(json.loads(target.read_text())["trace_id"], "trace-test")

    def test_explanation_blocks_trace_mismatch(self):
        with self.assertRaises(ValueError):
            project.ExplanationAgent().run({"trace_id": "a"}, {"trace_id": "b"}, {"trace_id": "a"})

    def test_training_limits_accept_interactive_values(self):
        config = project.load_config()
        with patch("builtins.input", side_effect=["0.65", "4", "yes"]):
            self.assertEqual(project.resolve_training_limits(config, None, None, False), (0.65, 4))

    def test_training_limits_reject_invalid_accuracy(self):
        with self.assertRaises(ValueError):
            project.resolve_training_limits(project.load_config(), 1.1, 3, True)


class LauncherTests(unittest.TestCase):
    def test_current_trained_model_is_ready(self):
        self.assertIsNone(launcher.trained_model_issue())
        manifest = json.loads(launcher.MANIFEST_FILE.read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_file"], "model/model.pt")

    def test_missing_model_returns_clear_instruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(launcher, "MANIFEST_FILE", Path(temporary) / "missing.json"):
                self.assertIn("Choose 1", launcher.trained_model_issue())

    def test_image_choice_retries_invalid_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "image.png"
            image.touch()
            with patch("builtins.input", side_effect=["missing.png", str(image)]), patch("builtins.print"):
                self.assertEqual(launcher.choose_image(), image.resolve())

    def test_platform_has_two_processing_choices(self):
        page = (PROJECT_ROOT / "platform" / "app.py").read_text(encoding="utf-8")
        self.assertEqual(page.count('data-mode="'), 2)
        self.assertIn("Test trained model", page)
        self.assertIn("Run full process", page)


if __name__ == "__main__":
    unittest.main()
