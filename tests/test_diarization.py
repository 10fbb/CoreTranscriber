from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.diarization import OnlineSpeakerClusterer


class FakeClusterer(OnlineSpeakerClusterer):
    def __init__(self, cache_dir: Path, embeddings: list[np.ndarray]) -> None:
        super().__init__(cache_dir, similarity_threshold=0.8)
        self.embeddings = iter(embeddings)

    def _embedding(self, samples: np.ndarray):
        vector = next(self.embeddings)
        return vector / np.linalg.norm(vector)


class DiarizationTests(unittest.TestCase):
    def test_clusters_same_and_different_voices(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            clusterer = FakeClusterer(
                Path(temp),
                [
                    np.array([1.0, 0.0]),
                    np.array([0.99, 0.01]),
                    np.array([0.0, 1.0]),
                ],
            )
            audio = np.ones(16_000, dtype=np.float32)
            self.assertEqual(clusterer.identify(audio), "remote-1")
            self.assertEqual(clusterer.identify(audio), "remote-1")
            self.assertEqual(clusterer.identify(audio), "remote-2")

    def test_tracks_five_distinct_remote_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            embeddings = [np.eye(5, dtype=np.float32)[index] for index in range(5)]
            clusterer = FakeClusterer(Path(temp), embeddings)
            audio = np.ones(16_000, dtype=np.float32)

            speakers = [clusterer.identify(audio) for _ in range(5)]

            self.assertEqual(
                speakers,
                ["remote-1", "remote-2", "remote-3", "remote-4", "remote-5"],
            )


if __name__ == "__main__":
    unittest.main()
