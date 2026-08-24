from __future__ import annotations

import sys
import tempfile
import unittest
import wave
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
            writer = SessionWriter(Path(temp), "Тест", session_started_at=0.0)
            writer.write_audio(
                AudioPacket(
                    "microphone", np.full(4_800, 0.25, dtype=np.float32), 48_000, 0.1
                )
            )
            writer.write_audio(
                AudioPacket(
                    "system", np.full(4_800, 0.25, dtype=np.float32), 48_000, 0.1
                )
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
            writer.rename_speaker("remote-1", "Разработчик")
            writer.edit_entry_text(first.entry_id, "Исправленный REST API")
            writer.close()
            text = (writer.directory / "transcript.txt").read_text(encoding="utf-8")
            self.assertIn("[05.08.2026 14:30:15] Разработчик: Исправленный REST API", text)
            self.assertIn("Разработчик: Вторая реплика", text)
            self.assertTrue((writer.directory / "meeting_audio.wav").exists())
            self.assertFalse((writer.directory / "microphone.wav").exists())
            self.assertFalse((writer.directory / "system_audio.wav").exists())
            self.assertTrue((writer.directory / "transcript.srt").exists())
            with wave.open(str(writer.directory / "meeting_audio.wav"), "rb") as audio:
                samples = np.frombuffer(audio.readframes(4_800), dtype="<i2")
            self.assertGreater(float(samples.mean()), 15_000)

            refined_first = TranscriptEntry(
                "system",
                "Собеседник 2",
                "Уточнённый технический текст",
                1.0,
                2.0,
                "remote-1",
                entry_id=first.entry_id,
                created_at=first.created_at,
            )
            refined_second = TranscriptEntry(
                "system",
                "Собеседник 2",
                "Уточнённая вторая реплика",
                4.0,
                1.0,
                "remote-1",
                entry_id=second.entry_id,
                created_at=second.created_at,
            )
            writer.replace_entries([refined_first, refined_second])
            refined_text = (writer.directory / "transcript.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("Разработчик: Исправленный REST API", refined_text)
            self.assertIn("Разработчик: Уточнённая вторая реплика", refined_text)
            self.assertNotIn("Уточнённый технический текст", refined_text)


if __name__ == "__main__":
    unittest.main()
