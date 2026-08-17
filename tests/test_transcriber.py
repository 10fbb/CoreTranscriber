from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.transcriber import LocalWhisper, recommended_cpu_threads


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _RecordingModel:
    def __init__(self) -> None:
        self.options = {}

    def transcribe(self, samples, **options):
        self.options = options
        return [_Segment(" REST API "), _Segment(" работает ")], object()


class TranscriberTests(unittest.TestCase):
    def test_cpu_profile_matches_core_ultra_14_logical_threads(self) -> None:
        self.assertEqual(recommended_cpu_threads(14), 8)
        self.assertEqual(recommended_cpu_threads(4), 2)
        self.assertEqual(recommended_cpu_threads(1), 1)

    def test_core_ultra_profile_loads_base_with_int8_and_eight_threads(self) -> None:
        created: dict[str, object] = {}

        class FakeWhisperModel:
            def __init__(self, model_name: str, **options) -> None:
                created["model_name"] = model_name
                created.update(options)

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_faster_whisper = types.ModuleType("faster_whisper")
        fake_faster_whisper.WhisperModel = FakeWhisperModel

        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules,
            {"torch": fake_torch, "faster_whisper": fake_faster_whisper},
        ), patch("techtranscriber.transcriber.os.cpu_count", return_value=14):
            transcriber = LocalWhisper("base", Path(temp))
            transcriber.load()

        self.assertEqual(created["model_name"], "base")
        self.assertEqual(created["device"], "cpu")
        self.assertEqual(created["compute_type"], "int8")
        self.assertEqual(created["cpu_threads"], 8)
        self.assertEqual(created["num_workers"], 1)

    def test_live_options_use_fast_search_and_skip_second_vad(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcriber = LocalWhisper(
                "base",
                Path(temp),
                glossary=["REST API", "PostgreSQL", "Dion"],
            )
            model = _RecordingModel()
            transcriber._model = model

            text = transcriber.transcribe(np.ones(16_000, dtype=np.float32))

            self.assertEqual(text, "REST API работает")
            self.assertEqual(model.options["language"], "ru")
            self.assertEqual(model.options["beam_size"], 1)
            self.assertEqual(model.options["best_of"], 1)
            self.assertTrue(model.options["without_timestamps"])
            self.assertFalse(model.options["vad_filter"])
            self.assertIn("техничес", model.options["initial_prompt"].casefold())
            self.assertEqual(
                model.options["hotwords"], "REST API, PostgreSQL, Dion"
            )


if __name__ == "__main__":
    unittest.main()
