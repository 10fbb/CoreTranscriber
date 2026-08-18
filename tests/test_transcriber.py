from __future__ import annotations

import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.transcriber import (
    LocalWhisper,
    TranscriptionCancelled,
    recommended_cpu_threads,
)


class _Segment:
    def __init__(self, text: str, start: float = 0.0, end: float = 1.0) -> None:
        self.text = text
        self.start = start
        self.end = end


class _RecordingModel:
    def __init__(self, segments: list[_Segment] | None = None) -> None:
        self.options = {}
        self.segments = segments or [_Segment(" REST API "), _Segment(" работает ")]

    def transcribe(self, samples, **options):
        self.options = options
        return self.segments, object()


class _RecordingBatchedPipeline:
    def __init__(self, model) -> None:
        self.model = model

    def transcribe(self, samples, **options):
        return self.model.transcribe(samples, **options)


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

    def test_saved_audio_refinement_uses_fast_batched_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcriber = LocalWhisper("turbo", Path(temp), glossary=["REST API"])
            model = _RecordingModel([_Segment(" REST API работает ", 2.5, 4.0)])
            transcriber._model = model

            with patch(
                "faster_whisper.BatchedInferencePipeline",
                _RecordingBatchedPipeline,
            ):
                segments = transcriber.transcribe_file(
                    Path(temp) / "system_audio.wav", batch_size=4, beam_size=1
                )

            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].text, "REST API работает")
            self.assertEqual(segments[0].start_seconds, 2.5)
            self.assertEqual(segments[0].duration_seconds, 1.5)
            self.assertEqual(model.options["beam_size"], 1)
            self.assertEqual(model.options["best_of"], 1)
            self.assertEqual(model.options["batch_size"], 4)
            self.assertFalse(model.options["without_timestamps"])
            self.assertTrue(model.options["vad_filter"])

    def test_hotwords_fit_complete_terms_into_token_budget(self) -> None:
        class FakeTokenizer:
            def encode(self, value: str):
                return types.SimpleNamespace(ids=list(value))

        with tempfile.TemporaryDirectory() as temp:
            first = "A" * 100
            too_large_together = "B" * 100
            last = "C"
            transcriber = LocalWhisper(
                "small", Path(temp), glossary=[first, too_large_together, last]
            )
            transcriber._model = types.SimpleNamespace(hf_tokenizer=FakeTokenizer())

            self.assertEqual(transcriber._hotwords(), f"{first}, {last}")

    def test_refinement_can_be_cancelled_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            transcriber = LocalWhisper("small", Path(temp))
            cancel = threading.Event()
            cancel.set()

            with self.assertRaises(TranscriptionCancelled):
                transcriber.transcribe_file(
                    Path(temp) / "system_audio.wav", cancel_event=cancel
                )


if __name__ == "__main__":
    unittest.main()
