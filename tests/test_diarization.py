from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def _embeddings(self, samples: list[np.ndarray]):
        result = []
        for _ in samples:
            vector = next(self.embeddings)
            result.append(vector / np.linalg.norm(vector))
        return result


class RecoveringClusterer(OnlineSpeakerClusterer):
    def __init__(self, cache_dir: Path, statuses: list[str]) -> None:
        super().__init__(
            cache_dir,
            on_status=statuses.append,
            retry_base_seconds=10.0,
            retry_max_seconds=30.0,
        )
        self.attempts = 0

    def _load(self) -> None:
        self._encoder = object()

    def _encode_samples(self, samples: np.ndarray) -> np.ndarray:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("временная ошибка тестовой модели")
        return np.array([1.0, 0.0], dtype=np.float32)


class DiarizationTests(unittest.TestCase):
    def test_retries_after_temporary_failure_and_reports_recovery(self) -> None:
        statuses: list[str] = []
        with tempfile.TemporaryDirectory() as temp:
            clusterer = RecoveringClusterer(Path(temp), statuses)
            samples = np.ones(16_000, dtype=np.float32)
            with patch(
                "techtranscriber.diarization.time.monotonic",
                side_effect=[100.0, 105.0, 111.0],
            ):
                self.assertIsNone(clusterer._embedding(samples))
                self.assertIsNone(clusterer._embedding(samples))
                recovered = clusterer._embedding(samples)

            self.assertEqual(clusterer.attempts, 2)
            self.assertTrue(np.array_equal(recovered, np.array([1.0, 0.0])))
            self.assertIn("повтор через 10 с", statuses[0])
            self.assertIn("временная ошибка тестовой модели", statuses[0])
            self.assertEqual(
                statuses[-1], "Разделение удалённых голосов восстановлено"
            )

    def test_prepare_validates_encoder_before_recording(self) -> None:
        statuses: list[str] = []
        with tempfile.TemporaryDirectory() as temp:
            clusterer = RecoveringClusterer(Path(temp), statuses)
            clusterer.attempts = 1

            clusterer.prepare()

            self.assertEqual(clusterer.attempts, 2)
            self.assertEqual(statuses, ["Модель различения голосов готова"])

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

    def test_batches_embeddings_and_keeps_temporal_clustering_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            clusterer = FakeClusterer(
                Path(temp),
                [
                    np.array([1.0, 0.0]),
                    np.array([0.99, 0.01]),
                    np.array([0.0, 1.0]),
                ],
            )
            audio = [np.ones(16_000, dtype=np.float32) for _ in range(3)]

            self.assertEqual(
                clusterer.identify_many(audio),
                ["remote-1", "remote-1", "remote-2"],
            )


if __name__ == "__main__":
    unittest.main()
