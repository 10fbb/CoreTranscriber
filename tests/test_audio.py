from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.audio import SpeechSegmenter, resample_mono
from techtranscriber.models import AudioPacket


class AudioTests(unittest.TestCase):
    def test_resample_length(self) -> None:
        source = np.zeros(48_000, dtype=np.float32)
        target = resample_mono(source, 48_000, 16_000)
        self.assertEqual(target.shape, (16_000,))

    def test_segmenter_emits_speech(self) -> None:
        utterances = []
        started = time.monotonic()
        segmenter = SpeechSegmenter(
            "microphone",
            started,
            utterances.append,
            energy_threshold=0.01,
            end_silence_seconds=0.2,
            min_duration_seconds=0.2,
        )
        captured = started
        for _ in range(8):
            segmenter.accept(
                AudioPacket(
                    "microphone",
                    np.full(2_400, 0.1, dtype=np.float32),
                    48_000,
                    captured,
                )
            )
            captured += 0.05
        for _ in range(5):
            segmenter.accept(
                AudioPacket(
                    "microphone",
                    np.zeros(2_400, dtype=np.float32),
                    48_000,
                    captured,
                )
            )
            captured += 0.05
        self.assertEqual(len(utterances), 1)
        self.assertEqual(utterances[0].sample_rate, 16_000)


if __name__ == "__main__":
    unittest.main()

