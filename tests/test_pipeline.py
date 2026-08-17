from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.models import AppSettings, Utterance
from techtranscriber.pipeline import MeetingPipeline


class _Transcriber:
    def load(self) -> None:
        return None

    def transcribe(self, samples) -> str:
        return "тест"


class PipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
