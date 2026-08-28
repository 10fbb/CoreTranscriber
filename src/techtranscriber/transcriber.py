from __future__ import annotations

import os
import re
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_HOTWORDS = 350
MAX_HOTWORD_TOKENS = 220
MAX_LIVE_HOTWORDS = 350
MAX_LIVE_HOTWORD_TOKENS = 220
DEFAULT_REFINEMENT_BATCH_SIZE = 4
LIVE_VAD_PARAMETERS = {
    "threshold": 0.6,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 450,
    "speech_pad_ms": 180,
}
LIVE_NO_SPEECH_THRESHOLD = 0.65
LIVE_LOG_PROB_THRESHOLD = -1.15
LIVE_COMPRESSION_RATIO_THRESHOLD = 2.2


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
        realtime: bool = False,
    ) -> None:
        self.model_name = model_name
        self.effective_model_name = model_name
        self.cache_dir = cache_dir
        self.language = language
        self.glossary = glossary or []
        self.on_status = on_status or (lambda _: None)
        self.realtime = realtime
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from faster_whisper import WhisperModel

                use_cuda = bool(torch.cuda.is_available())
                self.effective_model_name = _effective_model_name(
                    self.model_name, use_cuda, self.realtime
                )
                if self.effective_model_name != self.model_name:
                    self.on_status(
                        "Для живой записи без отставания на CPU используется "
                        f"модель {self.effective_model_name} вместо {self.model_name}"
                    )
                self.on_status(
                    f"Загрузка модели Whisper {self.effective_model_name}…"
                )
                device = "cuda" if use_cuda else "cpu"
                compute_type = "float16" if use_cuda else "int8"
                model_options = {}
                if not use_cuda:
                    model_options["cpu_threads"] = recommended_cpu_threads()
                    model_options["num_workers"] = 1
                self._model = WhisperModel(
                    self.effective_model_name,
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
                    f"Whisper готов: {self.effective_model_name}, {target}{detail}"
                )
            except Exception:
                self._model = None
                raise

    def transcribe(self, samples: np.ndarray) -> str:
        self.load()
        segments, _ = self._model.transcribe(
            np.asarray(samples, dtype=np.float32),
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=None,
            hotwords=self._hotwords(
                max_terms=MAX_LIVE_HOTWORDS,
                max_tokens=MAX_LIVE_HOTWORD_TOKENS,
            )
            or None,
            without_timestamps=True,
            vad_filter=True,
            vad_parameters=LIVE_VAD_PARAMETERS,
            no_speech_threshold=LIVE_NO_SPEECH_THRESHOLD,
            log_prob_threshold=LIVE_LOG_PROB_THRESHOLD,
            compression_ratio_threshold=LIVE_COMPRESSION_RATIO_THRESHOLD,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
        )
        parts = [
            segment.text.strip()
            for segment in segments
            if segment.text.strip() and _is_reliable_segment(segment)
        ]
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
            if not text or not _is_reliable_segment(segment):
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

    def _hotwords(
        self,
        *,
        max_terms: int = MAX_HOTWORDS,
        max_tokens: int = MAX_HOTWORD_TOKENS,
    ) -> str:
        terms = [term.strip() for term in self.glossary if term.strip()][
            :max_terms
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
            if token_count <= max_tokens:
                selected.append(term)
        return ", ".join(selected)


def _effective_model_name(
    requested: str, use_cuda: bool, realtime: bool
) -> str:
    if realtime and not use_cuda and requested not in {"tiny", "base"}:
        return "base"
    return requested


def _is_reliable_segment(segment) -> bool:
    text = str(getattr(segment, "text", "")).strip()
    if not text or is_repetitive_text(text):
        return False

    no_speech = _optional_float(getattr(segment, "no_speech_prob", None))
    avg_logprob = _optional_float(getattr(segment, "avg_logprob", None))
    compression = _optional_float(getattr(segment, "compression_ratio", None))
    if (
        no_speech is not None
        and no_speech >= LIVE_NO_SPEECH_THRESHOLD
        and (avg_logprob is None or avg_logprob < -0.6)
    ):
        return False
    if avg_logprob is not None and avg_logprob < LIVE_LOG_PROB_THRESHOLD:
        return False
    if compression is not None and compression > LIVE_COMPRESSION_RATIO_THRESHOLD:
        return False
    return True


def is_repetitive_text(text: str) -> bool:
    """Reject long loops such as «агент, агент, агент…» without harming speech."""
    words = re.findall(r"[0-9a-zа-яё]+", text.casefold())
    if len(words) < 6:
        return False
    counts = Counter(words)
    if max(counts.values()) / len(words) >= 0.6:
        return True
    if len(words) >= 9 and len(counts) / len(words) <= 0.3:
        return True
    return False


def _optional_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class StubWhisper:
    """Small test double used by the unit tests."""

    def __init__(self, text: str = "тестовая фраза") -> None:
        self.text = text

    def load(self) -> None:
        return None

    def transcribe(self, samples: np.ndarray) -> str:
        return self.text if np.asarray(samples).size else ""
