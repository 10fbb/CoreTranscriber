from __future__ import annotations

import json
import re
import threading
import wave
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import AudioPacket, AudioSource, TranscriptEntry


class SessionWriter:
    def __init__(self, output_root: Path, title: str, sample_rate: int = 48_000) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_title = _safe_name(title) or "Встреча"
        self.directory = output_root / f"{timestamp}_{safe_title}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sample_rate = sample_rate
        self._entries: list[TranscriptEntry] = []
        self._entry_lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._wave_locks = {
            "microphone": threading.Lock(),
            "system": threading.Lock(),
        }
        self._waves = {
            "microphone": self._open_wave("microphone.wav"),
            "system": self._open_wave("system_audio.wav"),
        }
        self._closed = False

    @property
    def entries(self) -> list[TranscriptEntry]:
        with self._entry_lock:
            return list(self._entries)

    def _open_wave(self, name: str):
        handle = wave.open(str(self.directory / name), "wb")
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(self.sample_rate)
        return handle

    def write_audio(self, packet: AudioPacket) -> None:
        if self._closed:
            return
        values = np.asarray(packet.samples, dtype=np.float32).reshape(-1)
        pcm = (np.clip(values, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        lock = self._wave_locks[packet.source]
        with lock:
            self._waves[packet.source].writeframesraw(pcm)

    def add_entry(self, entry: TranscriptEntry) -> None:
        with self._entry_lock:
            self._entries.append(entry)
        self.export_text_files()

    def rename_speaker(self, speaker_id: str, role: str) -> None:
        with self._entry_lock:
            for entry in self._entries:
                if entry.speaker_id == speaker_id:
                    entry.role = role
        self.export_text_files()

    def rename_entry(self, entry_id: str, role: str) -> None:
        with self._entry_lock:
            for entry in self._entries:
                if entry.entry_id == entry_id:
                    entry.role = role
                    break
        self.export_text_files()
        if self._closed:
            self.export_docx()

    def edit_entry_text(self, entry_id: str, text: str) -> None:
        with self._entry_lock:
            for entry in self._entries:
                if entry.entry_id == entry_id:
                    entry.text = text
                    break
        self.export_text_files()
        if self._closed:
            self.export_docx()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for source, handle in self._waves.items():
            with self._wave_locks[source]:
                handle.close()
        self.export_text_files()
        self.export_docx()

    def export_text_files(self) -> None:
        with self._export_lock:
            entries = self.entries
            txt = "\n".join(
                f"[{_real_time(item.created_at)}] {item.role}: {item.text}" for item in entries
            )
            (self.directory / "transcript.txt").write_text(txt, encoding="utf-8")

            md = "# Стенограмма\n\n" + "\n\n".join(
                f"**{_real_time(item.created_at)} · {item.role}**  \n{item.text}"
                for item in entries
            )
            (self.directory / "transcript.md").write_text(md, encoding="utf-8")

            payload = []
            for item in entries:
                row = asdict(item)
                row["created_at"] = item.created_at.isoformat()
                payload.append(row)
            (self.directory / "transcript.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            srt_parts: list[str] = []
            for index, item in enumerate(entries, 1):
                end = item.start_seconds + max(item.duration_seconds, 1.0)
                srt_parts.append(
                    f"{index}\n{_srt_time(item.start_seconds)} --> {_srt_time(end)}\n"
                    f"{item.role}: {item.text}"
                )
            (self.directory / "transcript.srt").write_text(
                "\n\n".join(srt_parts), encoding="utf-8"
            )

    def export_docx(self) -> None:
        try:
            from docx import Document
            from docx.shared import Pt
        except ImportError:
            return
        document = Document()
        document.add_heading("Стенограмма встречи", level=0)
        for item in self.entries:
            paragraph = document.add_paragraph()
            label = paragraph.add_run(f"{_real_time(item.created_at)} · {item.role}\n")
            label.bold = True
            label.font.size = Pt(10)
            paragraph.add_run(item.text)
        document.save(self.directory / "transcript.docx")


def _safe_name(value: str) -> str:
    value = re.sub(r"[<>:\"/\\|?*]+", "_", value.strip())
    return re.sub(r"\s+", " ", value)[:80]


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _real_time(value: datetime) -> str:
    return value.strftime("%d.%m.%Y %H:%M:%S")


def _srt_time(seconds: float) -> str:
    millis = max(0, int(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
