import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "main"))

import derma_agent as project
import agentic_derma as agentic
import knowledge
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

    def test_adaptive_learning_rate_reduces_on_plateau(self):
        config = project.load_config()
        history = [
            {"train_loss": 1.0, "validation": {"macro_f1": 0.30}},
            {"train_loss": 0.9, "validation": {"macro_f1": 0.29}},
        ]
        updated, decision = project.adapt_learning_rate(history, 0.003, config)
        self.assertLess(updated, 0.003)
        self.assertIn("reduce", decision)

    def test_automatic_parameters_stay_inside_limits(self):
        config = project.load_config()
        selected = project.choose_training_parameters("resnet18_28", config)
        self.assertGreaterEqual(selected["learning_rate"], config["training"]["adaptive"]["minimum_learning_rate"])
        self.assertLessEqual(selected["learning_rate"], config["training"]["adaptive"]["maximum_learning_rate"])

    def test_manual_hyperparameters_are_validated_against_the_same_bounds(self):
        config = project.load_config()
        adaptive = config["training"]["adaptive"]
        accepted = project.validate_manual_hyperparameters({"learning_rate": adaptive["maximum_learning_rate"]}, config)
        self.assertEqual(accepted["selection_source"], "user-specified")
        self.assertEqual(accepted["batch_size"], config["training"]["batch_size"])
        with self.assertRaises(ValueError):
            project.validate_manual_hyperparameters({"learning_rate": adaptive["maximum_learning_rate"] * 10}, config)

    def test_tuning_agent_switches_between_automatic_and_manual(self):
        config = project.load_config()
        tuner = project.TuningAgent()
        automatic = tuner.plan("resnet18_28", config, None)
        self.assertNotEqual(automatic["selection_source"], "user-specified")
        manual = tuner.plan("resnet18_28", config, {"learning_rate": config["training"]["learning_rate"]})
        self.assertEqual(manual["selection_source"], "user-specified")


class AgentBoundaryTests(unittest.TestCase):
    def test_agent_names_follow_process_order(self):
        self.assertEqual(set(project.AGENT_SPECS), {"Agent1", "Agent2", "Agent3", "Agent4"})
        self.assertEqual(project.TuningAgent.name, "Agent1")
        self.assertEqual(project.TrainingAgent.name, "Agent2")
        self.assertEqual(project.EvaluationAgent.name, "Agent3")
        self.assertEqual(project.ReviewAgent.name, "Agent4")

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

    def test_resolve_hyperparameters_from_flags_skips_prompt(self):
        result = project.resolve_hyperparameters(project.load_config(), 0.002, None, 64, True)
        self.assertEqual(result, {"learning_rate": 0.002, "batch_size": 64})

    def test_resolve_hyperparameters_interactive_manual_choice(self):
        with patch("builtins.input", side_effect=["m", "0.0025"]):
            result = project.resolve_hyperparameters(project.load_config(), None, None, None, False)
        self.assertEqual(result, {"learning_rate": 0.0025})

    def test_resolve_hyperparameters_interactive_automatic_choice(self):
        with patch("builtins.input", side_effect=["a"]):
            result = project.resolve_hyperparameters(project.load_config(), None, None, None, False)
        self.assertIsNone(result)

    def test_resolve_hyperparameters_skipped_when_confirmed_without_flags(self):
        result = project.resolve_hyperparameters(project.load_config(), None, None, None, True)
        self.assertIsNone(result)


class MemoryAndCoordinationTests(unittest.TestCase):
    def test_memory_keeps_facts_and_samples_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(agentic, "MEMORY_DIR", root), patch.object(agentic, "STATE_PATH", root / "state.json"), patch.object(agentic, "EVENTS_PATH", root / "events.jsonl"):
                memory = agentic.AgentMemory()
                memory.save_state({"model_ready": True})
                memory.record("chat", "User", "check the trained model", "session-a")
                context = memory.context("session-a", "trained model", 2)
                self.assertTrue(context["deterministic"]["model_ready"])
                self.assertEqual(context["stochastic"][0]["session"], "session-a")
                self.assertIsInstance(context["selection_seed"], int)

    def test_agentic_derma_routes_training_and_images(self):
        coordinator = agentic.AgenticDerma()
        self.assertEqual(coordinator.route("train automatically", False), "train")
        self.assertEqual(coordinator.route("train automatically", True), "train")
        self.assertEqual(coordinator.route("Is there a trained model?", False), "status")
        with patch.object(coordinator, "status", return_value={"model": True}):
            self.assertEqual(coordinator.route("analyze", True), "predict")

    def test_prediction_summary_preserves_ranked_probabilities(self):
        coordinator = agentic.AgenticDerma()
        coordinator.memory = MagicMock()
        coordinator.memory.state.return_value = {}
        result = {
            "prediction": "class one", "confidence": 0.6, "uncertain": True, "trace_id": "trace-1",
            "probabilities": {"class one": 0.6, "class two": 0.3, "class three": 0.1},
        }
        with patch.object(coordinator, "status", return_value={"metrics": {"accuracy": 0.55}}):
            message = coordinator.finish_prediction("session", result)
        self.assertIn("class one (60.0%)", message)
        self.assertIn("class two (30.0%) and class three (10.0%)", message)
        self.assertIn("55.0% accuracy", message)

    def test_grounding_rejects_missing_or_unknown_citations(self):
        sources = [{"label": "S1", "title": "Source", "url": "https://example.org"}]
        self.assertFalse(agentic.AgenticDerma._grounded_answer_is_valid("A clinical statement.", sources))
        self.assertFalse(agentic.AgenticDerma._grounded_answer_is_valid("A claim [S2].", sources))
        self.assertTrue(agentic.AgenticDerma._grounded_answer_is_valid("A supported statement [S1].", sources))

    def test_grounding_rejects_unsafe_model_language(self):
        sources = [{"label": "S1", "title": "Source", "url": "https://example.org"}]
        answer = "The model performs well and confirms melanoma [S1]."
        self.assertFalse(agentic.AgenticDerma._grounded_answer_is_valid(answer, sources))

    def test_chat_history_excludes_the_in_flight_message(self):
        coordinator = agentic.AgenticDerma()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(agentic, "MEMORY_DIR", root), patch.object(agentic, "STATE_PATH", root / "state.json"), patch.object(agentic, "EVENTS_PATH", root / "events.jsonl"):
                coordinator.memory = agentic.AgentMemory()
                coordinator.memory.record("chat", "User", "earlier question", "session-x")
                coordinator.memory.record("chat", "AgenticDerma", "earlier answer", "session-x", {"action": "status", "sources": []})
                with patch.object(coordinator, "status", return_value={"model": False}), patch.object(coordinator, "_grounded_response", return_value=("answer", [])) as grounded:
                    coordinator.chat("session-x", "new question about dermoscopy")
                history = grounded.call_args[0][4]
                self.assertEqual(history, [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}])

    def test_handoff_declutter_removes_filler_framing(self):
        text = "Has completed the task of initiating the attribution process, which now proceeds to Agent4 for further analysis."
        cleaned = agentic.AgenticDerma._declutter(text)
        self.assertNotIn("for further analysis", cleaned.lower())
        self.assertNotIn("has completed the task of", cleaned.lower())
        self.assertTrue(cleaned[0].isupper())

    def test_online_endpoint_requires_https(self):
        client = agentic.LLMClient(project.load_config())
        with self.assertRaises(ValueError):
            client.configure("compatible", "http://example.org/v1/chat/completions", "model", "key")

    def test_free_keyed_providers_require_a_key(self):
        client = agentic.LLMClient(project.load_config())
        for provider in ("groq", "gemini", "openrouter"):
            with self.assertRaises(ValueError):
                client.configure(provider)

    def test_free_keyed_provider_accepts_a_key(self):
        client = agentic.LLMClient(project.load_config())
        status = client.configure("groq", api_key="test-key")
        self.assertEqual(status["provider"], "groq")
        self.assertEqual(status["provider_label"], "Groq")
        self.assertTrue(status["key_loaded"])
        self.assertTrue(status["key_required"])


class KnowledgeTests(unittest.TestCase):
    def test_dermoscopy_retrieval_returns_traceable_sources(self):
        result = knowledge.KnowledgeBase().retrieve("dermoscopy melanoma structures", 3)
        self.assertTrue(result["context"])
        self.assertTrue(result["sources"])
        self.assertEqual(result["sources"][0]["label"], "S1")
        self.assertTrue(result["sources"][0]["url"].startswith("https://"))

    def test_retrieval_fallback_keeps_source_and_model_boundary(self):
        retrieval = knowledge.KnowledgeBase().retrieve("dermoscopy lesion image", 2)
        answer, sources = agentic.AgenticDerma._retrieval_fallback(
            "What can dermoscopy show?", retrieval,
            {"prediction": "melanoma", "confidence": 0.62}, "fallback",
        )
        self.assertIn("[S1]", answer)
        self.assertIn("seven classes", answer)
        self.assertEqual(sources, retrieval["sources"])

    def test_retrieval_fallback_quotes_the_top_ranked_passage(self):
        retrieval = knowledge.KnowledgeBase().retrieve("cherry hemangioma dermoscopy lacunae", 3)
        top_text = retrieval["passages"][0]["text"].lower()
        answer, _ = agentic.AgenticDerma._retrieval_fallback(
            "What does dermoscopy show for a cherry hemangioma?", retrieval, {}, "fallback",
        )
        self.assertIn(answer.split(" [")[0].strip().lower()[:40], top_text)

    def test_knowledge_covers_all_seven_dermamnist_classes(self):
        base = knowledge.KnowledgeBase()
        queries = [
            "actinic keratosis intraepithelial carcinoma", "basal cell carcinoma arborizing vessels",
            "seborrheic keratosis milia-like cysts", "dermatofibroma central white patch",
            "melanoma asymmetric pigment network", "melanocytic nevus dermoscopic pattern",
            "vascular lesion red lacunae",
        ]
        for query in queries:
            result = base.retrieve(query, 3)
            self.assertTrue(result["sources"], query)


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

    def test_platform_has_chat_and_live_agent_graph(self):
        page = (PROJECT_ROOT / "platform" / "app.py").read_text(encoding="utf-8")
        self.assertIn("Ask AgenticDerma", page)
        self.assertIn("Process map", page)
        self.assertIn("backdrop-filter:blur", page)
        self.assertIn("@media(max-width:680px)", page)
        self.assertIn("Agent1", page)
        self.assertIn("Agent2", page)
        self.assertIn("Agent3", page)
        self.assertIn("Agent4", page)
        self.assertIn("New chat", page)


if __name__ == "__main__":
    unittest.main()
