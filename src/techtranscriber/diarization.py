from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np


class OnlineSpeakerClusterer:
    """Assigns stable session-local speaker IDs using ECAPA voice embeddings."""

    def __init__(
        self,
        cache_dir: Path,
        similarity_threshold: float = 0.67,
        max_speakers: int = 12,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.on_status = on_status or (lambda _: None)
        self._encoder = None
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._lock = threading.Lock()
        self._disabled_reason: str | None = None

    def identify(self, samples: np.ndarray, sample_rate: int = 16_000) -> str:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size < int(sample_rate * 0.45):
            return self._fallback_id()
        embedding = self._embedding(samples)
        if embedding is None:
            return self._fallback_id()

        with self._lock:
            if not self._centroids:
                return self._add(embedding)
            similarities = [float(np.dot(embedding, center)) for center in self._centroids]
            best = int(np.argmax(similarities))
            if similarities[best] < self.similarity_threshold and len(self._centroids) < self.max_speakers:
                return self._add(embedding)
            count = self._counts[best]
            updated = (self._centroids[best] * count + embedding) / (count + 1)
            self._centroids[best] = _normalize(updated)
            self._counts[best] = count + 1
            return f"remote-{best + 1}"

    def _add(self, embedding: np.ndarray) -> str:
        self._centroids.append(embedding)
        self._counts.append(1)
        return f"remote-{len(self._centroids)}"

    def _fallback_id(self) -> str:
        return "remote-1"

    def _embedding(self, samples: np.ndarray) -> np.ndarray | None:
        try:
            self._load()
            if self._encoder is None:
                return None
            import torch

            waveform = torch.from_numpy(samples).unsqueeze(0)
            with torch.inference_mode():
                encoded = self._encoder.encode_batch(waveform)
            vector = encoded.squeeze().detach().cpu().numpy().astype(np.float32)
            return _normalize(vector)
        except Exception as exc:
            if self._disabled_reason is None:
                self._disabled_reason = str(exc)
                self.on_status(
                    "Разделение удалённых голосов временно недоступно; "
                    "текст продолжит записываться как «Собеседник 1»"
                )
            return None

    def _load(self) -> None:
        if self._encoder is not None or self._disabled_reason is not None:
            return
        with self._lock:
            if self._encoder is not None or self._disabled_reason is not None:
                return
            self.on_status("Загрузка локальной модели различения голосов…")
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            kwargs = {
                "source": "speechbrain/spkrec-ecapa-voxceleb",
                "savedir": str(self.cache_dir / "speaker_ecapa"),
                "run_opts": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            }
            try:
                from speechbrain.utils.fetching import LocalStrategy

                kwargs["local_strategy"] = LocalStrategy.COPY
            except (ImportError, AttributeError):
                pass
            self._encoder = EncoderClassifier.from_hparams(**kwargs)
            self.on_status("Модель различения голосов готова")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm

