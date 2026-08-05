from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.models import AudioPacket, TranscriptEntry
from techtranscriber.storage import SessionWriter


class StorageTests(unittest.TestCase):
    def test_session_exports_and_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            writer = SessionWriter(Path(temp), "Тест")
            writer.write_audio(
                AudioPacket("microphone", np.zeros(4_800, dtype=np.float32), 48_000, 0)
            )
            first = TranscriptEntry(
                "system",
                "Собеседник 1",
                "Проверка REST API",
                1.0,
                2.0,
                "remote-1",
                created_at=datetime(2026, 8, 5, 14, 30, 15),
            )
            second = TranscriptEntry(
                "system",
                "Собеседник 1",
                "Вторая реплика",
                4.0,
                1.0,
                "remote-1",
                created_at=datetime(2026, 8, 5, 14, 30, 18),
            )
            writer.add_entry(first)
            writer.add_entry(second)
            writer.rename_entry(first.entry_id, "Разработчик")
            writer.edit_entry_text(first.entry_id, "Исправленный REST API")
            writer.close()
            text = (writer.directory / "transcript.txt").read_text(encoding="utf-8")
            self.assertIn("[05.08.2026 14:30:15] Разработчик: Исправленный REST API", text)
            self.assertIn("Собеседник 1: Вторая реплика", text)
            self.assertTrue((writer.directory / "microphone.wav").exists())
            self.assertTrue((writer.directory / "system_audio.wav").exists())
            self.assertTrue((writer.directory / "transcript.srt").exists())


if __name__ == "__main__":
    unittest.main()

