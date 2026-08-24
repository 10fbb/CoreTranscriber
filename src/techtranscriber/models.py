from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

AudioSource = Literal["microphone", "system"]


@dataclass(slots=True)
class DeviceInfo:
    identifier: str
    name: str
    is_default: bool = False


@dataclass(slots=True)
class AudioPacket:
    source: AudioSource
    samples: object
    sample_rate: int
    captured_at: float


@dataclass(slots=True)
class Utterance:
    source: AudioSource
    samples: object
    sample_rate: int
    start_seconds: float
    duration_seconds: float


@dataclass(slots=True)
class TranscriptEntry:
    source: AudioSource
    role: str
    text: str
    start_seconds: float
    duration_seconds: float
    speaker_id: str | None = None
    entry_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=datetime.now)
    role_edited: bool = False
    text_edited: bool = False


@dataclass(slots=True)
class AppSettings:
    microphone_id: str = ""
    speaker_id: str = ""
    whisper_model: str = "base"
    language: str = "ru"
    glossary: list[str] = field(default_factory=list)
    active_dictionaries: list[str] = field(
        default_factory=lambda: ["01_Общие_IT_и_платформы.txt", "08_Аудио_и_AI.txt"]
    )
    energy_threshold: float = 0.008
    speaker_threshold: float = 0.67
    output_root: Path | None = None
    settings_revision: int = 4
    refine_after_recording: bool = True
    refinement_model: str = "small"
    ui_theme: str = "modern_dark"
