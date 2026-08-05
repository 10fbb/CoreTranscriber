from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np


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
                self._model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(self.cache_dir / "whisper"),
                )
                target = "видеокарта" if use_cuda else "процессор"
                self.on_status(f"Whisper готов: {self.model_name}, {target}")
            except Exception:
                self._model = None
                raise

    def transcribe(self, samples: np.ndarray) -> str:
        self.load()
        prompt = self._prompt()
        segments, _ = self._model.transcribe(
            np.asarray(samples, dtype=np.float32),
            language=self.language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            hotwords=self._hotwords() or None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
        )
        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(parts).strip()

    def _prompt(self) -> str:
        return "Техническое совещание на русском языке."

    def _hotwords(self) -> str:
        terms = [term.strip() for term in self.glossary if term.strip()]
        return ", ".join(terms)


class StubWhisper:
    """Small test double used by the unit tests."""

    def __init__(self, text: str = "тестовая фраза") -> None:
        self.text = text

    def load(self) -> None:
        return None

    def transcribe(self, samples: np.ndarray) -> str:
        return self.text if np.asarray(samples).size else ""
