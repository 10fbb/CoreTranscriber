from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime
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
        retry_base_seconds: float = 15.0,
        retry_max_seconds: float = 60.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.on_status = on_status or (lambda _: None)
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._encoder = None
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._next_retry_at = 0.0
        self._ready_reported = False

    def prepare(self) -> None:
        """Download and validate the speaker model before a meeting starts."""
        self._load()
        positions = np.arange(16_000, dtype=np.float32)
        test_signal = np.sin(2.0 * np.pi * 220.0 * positions / 16_000.0).astype(
            np.float32
        )
        self._encode_samples(test_signal)
        self._mark_success()

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
        now = time.monotonic()
        if now < self._next_retry_at:
            return None
        try:
            self._load()
            vector = self._encode_samples(samples)
            self._mark_success()
            return vector
        except Exception as exc:
            self._mark_failure(exc, now)
            return None

    def _load(self) -> None:
        if self._encoder is not None:
            return
        with self._lock:
            if self._encoder is not None:
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

    def _encode_samples(self, samples: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("голосовая модель не загружена")
        import torch

        clean_samples = np.nan_to_num(
            np.asarray(samples, dtype=np.float32).reshape(-1), copy=True
        )
        waveform = torch.from_numpy(np.ascontiguousarray(clean_samples)).unsqueeze(0)
        with torch.inference_mode():
            encoded = self._encoder.encode_batch(waveform)
        vector = encoded.squeeze().detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("голосовая модель вернула некорректный отпечаток")
        normalized = _normalize(vector)
        if not np.any(normalized):
            raise RuntimeError("не удалось построить голосовой отпечаток")
        return normalized

    def _mark_success(self) -> None:
        recovered = self._consecutive_failures > 0
        self._consecutive_failures = 0
        self._next_retry_at = 0.0
        if recovered:
            self.on_status("Разделение удалённых голосов восстановлено")
        elif not self._ready_reported:
            self.on_status("Модель различения голосов готова")
        self._ready_reported = True

    def _mark_failure(self, exc: Exception, now: float) -> None:
        self._consecutive_failures += 1
        multiplier = 2 ** min(self._consecutive_failures - 1, 2)
        delay = min(self.retry_max_seconds, self.retry_base_seconds * multiplier)
        self._next_retry_at = now + delay
        reason = _short_error(exc)
        self.on_status(
            "Разделение удалённых голосов временно недоступно; "
            f"повтор через {round(delay)} с. Причина: {reason}"
        )
        self._write_failure_log(exc)

    def _write_failure_log(self, exc: Exception) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat(timespec="seconds")
            log_path = self.cache_dir / "speaker_diarization.log"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{timestamp} {type(exc).__name__}: {exc}\n"
                )
        except OSError:
            pass


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _short_error(exc: Exception, limit: int = 180) -> str:
    message = " ".join(str(exc).split()) or type(exc).__name__
    if len(message) <= limit:
        return message
    return message[: limit - 1] + "…"
