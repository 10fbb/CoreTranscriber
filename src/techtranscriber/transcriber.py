from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_HOTWORDS = 200
MAX_HOTWORD_TOKENS = 180
DEFAULT_REFINEMENT_BATCH_SIZE = 4


@dataclass(frozen=True, slots=True)
class TimedText:
    text: str
    start_seconds: float
    duration_seconds: float


class TranscriptionCancelled(RuntimeError):
    pass


def recommended_cpu_threads(logical_cpu_count: int | None = None) -> int:
    """Leave headroom for audio capture and the UI on CPU-only computers."""
    available = logical_cpu_count or os.cpu_count() or 4
    if available <= 2:
        return max(1, available)
    return max(2, min(8, available - 2))


class LocalWhisper:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        language: str = "ru",
        glossary: list[str] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.language = language
        self.glossary = glossary or []
        self.on_status = on_status or (lambda _: None)
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self.on_status(f"Загрузка модели Whisper {self.model_name}…")
            try:
                import torch
                from faster_whisper import WhisperModel

                use_cuda = bool(torch.cuda.is_available())
                device = "cuda" if use_cuda else "cpu"
                compute_type = "float16" if use_cuda else "int8"
                model_options = {}
                if not use_cuda:
                    model_options["cpu_threads"] = recommended_cpu_threads()
                    model_options["num_workers"] = 1
                self._model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(self.cache_dir / "whisper"),
                    **model_options,
                )
                target = "видеокарта" if use_cuda else "процессор"
                detail = (
                    ""
                    if use_cuda
                    else f", {model_options['cpu_threads']} потоков, int8"
                )
                self.on_status(
                    f"Whisper готов: {self.model_name}, {target}{detail}"
                )
            except Exception:
                self._model = None
                raise

    def transcribe(self, samples: np.ndarray) -> str:
        self.load()
        prompt = self._prompt()
        segments, _ = self._model.transcribe(
            np.asarray(samples, dtype=np.float32),
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            hotwords=self._hotwords() or None,
            without_timestamps=True,
            vad_filter=False,
        )
        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(parts).strip()

    def transcribe_file(
        self,
        path: Path,
        *,
        beam_size: int = 1,
        batch_size: int = DEFAULT_REFINEMENT_BATCH_SIZE,
        cancel_event: threading.Event | None = None,
    ) -> list[TimedText]:
        """Produce a fast, timestamped transcript from saved audio in batches."""
        if cancel_event and cancel_event.is_set():
            raise TranscriptionCancelled("уточнение отменено")
        self.load()
        if cancel_event and cancel_event.is_set():
            raise TranscriptionCancelled("уточнение отменено")

        from faster_whisper import BatchedInferencePipeline

        batched_model = BatchedInferencePipeline(self._model)
        segments, _ = batched_model.transcribe(
            str(path),
            language=self.language,
            beam_size=max(1, beam_size),
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self._prompt(),
            hotwords=self._hotwords() or None,
            without_timestamps=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
            batch_size=max(1, batch_size),
        )
        result: list[TimedText] = []
        for segment in segments:
            if cancel_event and cancel_event.is_set():
                raise TranscriptionCancelled("уточнение отменено")
            text = segment.text.strip()
            if not text:
                continue
            start = max(0.0, float(segment.start))
            end = max(start, float(segment.end))
            result.append(TimedText(text, start, end - start))
        return result

    def _prompt(self) -> str:
        return (
            "Техническое совещание на русском языке. "
            "Участвуют несколько специалистов. Используются точные названия "
            "продуктов, систем, протоколов, аббревиатур и англоязычных технологий."
        )

    def _hotwords(self) -> str:
        terms = [term.strip() for term in self.glossary if term.strip()][
            :MAX_HOTWORDS
        ]
        tokenizer = getattr(self._model, "hf_tokenizer", None)
        if tokenizer is None:
            return ", ".join(terms)

        # faster-whisper truncates hotwords at half of Whisper's context window.
        # Fit complete terms ourselves so the final item is never cut mid-word and
        # enough context remains for the actual transcription.
        selected: list[str] = []
        for term in terms:
            candidate = ", ".join([*selected, term])
            token_count = len(tokenizer.encode(" " + candidate).ids)
            if token_count <= MAX_HOTWORD_TOKENS:
                selected.append(term)
        return ", ".join(selected)


class StubWhisper:
    """Small test double used by the unit tests."""

    def __init__(self, text: str = "тестовая фраза") -> None:
        self.text = text

    def load(self) -> None:
        return None

    def transcribe(self, samples: np.ndarray) -> str:
        return self.text if np.asarray(samples).size else ""
