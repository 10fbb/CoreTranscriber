from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.models import AppSettings, TranscriptEntry, Utterance
from techtranscriber.pipeline import MeetingPipeline, _write_masked_wave
from techtranscriber.transcriber import TimedText


class _Transcriber:
    def load(self) -> None:
        return None

    def transcribe(self, samples) -> str:
        return "тест"


class _Refiner:
    calls = []

    def __init__(self, *args, **kwargs) -> None:
        return None

    def transcribe_file(self, path: Path, **options) -> list[TimedText]:
        self.calls.append((path.name, options))
        return [
            TimedText("технический вопрос", 1.0, 1.5),
            TimedText("мой ответ", 3.0, 1.0),
        ]


class _Clusterer:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def identify_many(self, samples, sample_rate: int) -> list[str]:
        return ["remote-2"] * len(samples)


class _Writer:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.replaced = []
        self.entries = [
            TranscriptEntry(
                "system", "Собеседник 2", "вопрос", 1.0, 1.5, "remote-2"
            ),
            TranscriptEntry("microphone", "Вы", "ответ", 3.0, 1.0, "self"),
        ]

    def replace_entries(self, entries) -> None:
        self.replaced = list(entries)
        self.entries = list(entries)


class PipelineTests(unittest.TestCase):
    def test_manual_text_interval_is_silenced_for_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "meeting_audio.wav"
            masked = Path(temp) / "masked.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(
                    np.full(16_000, 12_000, dtype="<i2").tobytes()
                )

            _write_masked_wave(source, masked, [(0.25, 0.25)])

            with wave.open(str(masked), "rb") as audio:
                samples = np.frombuffer(audio.readframes(16_000), dtype="<i2")
            self.assertTrue(np.all(samples[4_000:8_000] == 0))
            self.assertTrue(np.all(samples[:4_000] == 12_000))
            self.assertTrue(np.all(samples[8_000:] == 12_000))

    def test_backlog_keeps_all_segments_and_reports_status(self) -> None:
        statuses: list[str] = []
        errors: list[str] = []
        with tempfile.TemporaryDirectory() as temp, patch(
            "techtranscriber.pipeline.model_cache_dir", return_value=Path(temp)
        ), patch("techtranscriber.pipeline.time.monotonic", return_value=100.0):
            pipeline = MeetingPipeline(
                AppSettings(output_root=Path(temp)),
                "Нагрузочный тест",
                lambda _: None,
                statuses.append,
                errors.append,
                transcriber=_Transcriber(),
            )
            utterance = Utterance(
                "system",
                np.ones(16_000, dtype=np.float32),
                16_000,
                0.0,
                1.0,
            )

            for _ in range(250):
                pipeline._enqueue(utterance)

            self.assertEqual(pipeline._queue.qsize(), 250)
            self.assertEqual(errors, [])
            self.assertEqual(len(statuses), 1)
            self.assertIn("аудио сохраняется", statuses[0])

    def test_offline_small_refinement_replaces_live_entries_after_success(self) -> None:
        statuses: list[str] = []
        displayed = []
        resets: list[bool] = []
        with tempfile.TemporaryDirectory() as temp, patch(
            "techtranscriber.pipeline.model_cache_dir", return_value=Path(temp)
        ), patch("techtranscriber.pipeline.LocalWhisper", _Refiner), patch(
            "techtranscriber.pipeline.OnlineSpeakerClusterer", _Clusterer
        ):
            pipeline = MeetingPipeline(
                AppSettings(output_root=Path(temp)),
                "Уточнение",
                displayed.append,
                statuses.append,
                lambda _: None,
                transcriber=_Transcriber(),
                on_reset=lambda: resets.append(True),
            )
            writer = _Writer(Path(temp))
            pipeline._writer = writer
            pipeline._wall_started_at = datetime(2026, 8, 18, 10, 0, 0)

            pipeline._refine_saved_transcript()

            self.assertEqual(resets, [True])
            self.assertEqual(writer.replaced, displayed)
            self.assertEqual(
                [entry.text for entry in writer.replaced],
                ["технический вопрос", "мой ответ"],
            )
            self.assertEqual(writer.replaced[0].role, "Собеседник 2")
            self.assertEqual(writer.replaced[1].role, "Вы")
            self.assertIn("уточнена", statuses[-1])
            self.assertIn("small", statuses[-1])
            self.assertEqual(_Refiner.calls[-1][1]["beam_size"], 1)
            self.assertEqual(_Refiner.calls[-1][1]["batch_size"], 8)
            self.assertEqual(_Refiner.calls[-1][0], "meeting_audio.wav")


if __name__ == "__main__":
    unittest.main()
