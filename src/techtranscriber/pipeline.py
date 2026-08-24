from __future__ import annotations

import queue
import threading
import time
import wave
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .audio import CaptureWorker, DeviceService, SpeechSegmenter
from .config import model_cache_dir
from .diarization import OnlineSpeakerClusterer
from .models import AppSettings, AudioPacket, AudioSource, TranscriptEntry, Utterance
from .storage import SessionWriter
from .transcriber import LocalWhisper, TranscriptionCancelled


REFINEMENT_OPTIONS = {
    "small": {"beam_size": 1, "batch_size": 8},
    "turbo": {"beam_size": 1, "batch_size": 4},
    "medium": {"beam_size": 2, "batch_size": 4},
}
class MeetingPipeline:
    BACKLOG_WARNING_UTTERANCES = 20
    BACKLOG_STATUS_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        settings: AppSettings,
        title: str,
        on_entry: Callable[[TranscriptEntry], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        transcriber=None,
        refiner=None,
        clusterer=None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        if settings.output_root is None:
            raise ValueError("Не выбрана папка для сохранения")
        self.settings = settings
        self.title = title
        self.on_entry = on_entry
        self.on_status = on_status
        self.on_error = on_error
        self.on_reset = on_reset or (lambda: None)
        self._transcriber = transcriber or LocalWhisper(
            settings.whisper_model,
            model_cache_dir(),
            settings.language,
            settings.glossary,
            on_status,
        )
        self._refiner = refiner
        self._clusterer = clusterer or OnlineSpeakerClusterer(
            model_cache_dir(), settings.speaker_threshold, on_status=on_status
        )
        self._speaker_names: dict[str, str] = {}
        # Audio is always written to disk first. The in-memory queue is deliberately
        # unbounded so a temporary CPU slowdown never discards transcript segments.
        self._queue: queue.Queue[Utterance | None] = queue.Queue()
        self._captures: list[CaptureWorker] = []
        self._segmenters: dict[AudioSource, SpeechSegmenter] = {}
        self._processor: threading.Thread | None = None
        self._writer: SessionWriter | None = None
        self._running = False
        self._started_at = 0.0
        self._wall_started_at = datetime.now()
        self._last_backlog_status_at = 0.0
        self._refinement_cancel = threading.Event()

    @property
    def session_directory(self) -> Path | None:
        return self._writer.directory if self._writer else None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._started_at = time.monotonic()
        self._wall_started_at = datetime.now()
        self._writer = SessionWriter(
            self.settings.output_root,
            self.title,
            session_started_at=self._started_at,
        )
        self._segmenters = {
            source: SpeechSegmenter(
                source,
                self._started_at,
                self._enqueue,
                energy_threshold=self.settings.energy_threshold,
            )
            for source in ("microphone", "system")
        }
        try:
            microphone = DeviceService.microphone(self.settings.microphone_id)
            loopback = DeviceService.loopback(self.settings.speaker_id)
            self._captures = [
                CaptureWorker(
                    "microphone", microphone, self._on_packet, self._on_capture_error
                ),
                CaptureWorker("system", loopback, self._on_packet, self._on_capture_error),
            ]
            self._running = True
            self._processor = threading.Thread(
                target=self._process_loop, name="transcription", daemon=True
            )
            self._processor.start()
            for worker in self._captures:
                worker.start()
            self.on_status("Запись началась: микрофон и звук приложений подключены")
        except Exception:
            self._running = False
            if self._writer:
                self._writer.close()
            raise

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.on_status("Завершаю запись и сохраняю файлы…")
        for worker in self._captures:
            worker.stop()
        for worker in self._captures:
            worker.join(timeout=3.0)
        for segmenter in self._segmenters.values():
            segmenter.flush()
        self._queue.put(None)
        if self._processor:
            self._processor.join()
        if self._writer:
            self._writer.close()
            if self.settings.refine_after_recording:
                try:
                    self._refine_saved_transcript()
                except TranscriptionCancelled:
                    self.on_status(
                        "Уточнение пропущено; сохранена живая стенограмма"
                    )
                except Exception as exc:
                    model_name = self.settings.refinement_model
                    self.on_status(
                        f"Не удалось уточнить стенограмму моделью {model_name}; "
                        f"сохранён живой текст: {exc}"
                    )
        self.on_status("Стенограмма и аудио сохранены")

    def cancel_refinement(self) -> None:
        self._refinement_cancel.set()
        self.on_status(
            "Останавливаю уточнение; текущая живая стенограмма сохранена…"
        )

    def rename_speaker(self, speaker_id: str, role: str) -> None:
        role = role.strip()
        if not speaker_id or not role:
            return
        self._speaker_names[speaker_id] = role
        if self._writer:
            self._writer.rename_speaker(speaker_id, role)

    def rename_entry(self, entry_id: str, role: str) -> None:
        if self._writer:
            self._writer.rename_entry(entry_id, role.strip())

    def edit_entry_text(self, entry_id: str, text: str) -> None:
        if self._writer:
            self._writer.edit_entry_text(entry_id, text.strip())

    def _on_packet(self, packet: AudioPacket) -> None:
        if not self._running:
            return
        if self._writer:
            self._writer.write_audio(packet)
        self._segmenters[packet.source].accept(packet)

    def _enqueue(self, utterance: Utterance) -> None:
        self._queue.put_nowait(utterance)
        pending = self._queue.qsize()
        now = time.monotonic()
        if (
            pending >= self.BACKLOG_WARNING_UTTERANCES
            and now - self._last_backlog_status_at
            >= self.BACKLOG_STATUS_INTERVAL_SECONDS
        ):
            self._last_backlog_status_at = now
            self.on_status(
                "Распознавание немного отстаёт; аудио сохраняется, "
                f"в очереди {pending} фрагментов"
            )

    def _process_loop(self) -> None:
        try:
            self._transcriber.load()
            while True:
                utterance = self._queue.get()
                if utterance is None:
                    break
                self._process_utterance(utterance)
        except Exception as exc:
            self.on_error(f"Ошибка локального распознавания: {exc}")

    def _process_utterance(self, utterance: Utterance) -> None:
        text = self._transcriber.transcribe(utterance.samples)
        if not text:
            return
        if utterance.source == "microphone":
            speaker_id = "self"
            role = self._speaker_names.get(speaker_id, "Вы")
        else:
            speaker_id = self._clusterer.identify(
                utterance.samples, utterance.sample_rate
            )
            role = self._speaker_names.get(speaker_id)
            if role is None:
                number = speaker_id.rsplit("-", 1)[-1]
                role = f"Собеседник {number}"
        entry = TranscriptEntry(
            source=utterance.source,
            role=role,
            text=text,
            start_seconds=utterance.start_seconds,
            duration_seconds=utterance.duration_seconds,
            speaker_id=speaker_id,
            created_at=self._wall_started_at
            + timedelta(seconds=utterance.start_seconds),
            role_edited=speaker_id in self._speaker_names,
        )
        if self._writer:
            self._writer.add_entry(entry)
        self.on_entry(entry)

    def _refine_saved_transcript(self) -> None:
        if self._writer is None:
            return
        if self._refinement_cancel.is_set():
            raise TranscriptionCancelled("уточнение отменено")
        requested_model = self.settings.refinement_model
        model_name = (
            requested_model if requested_model in REFINEMENT_OPTIONS else "small"
        )
        options = REFINEMENT_OPTIONS[model_name]
        self.on_status(
            f"Быстро уточняю итог моделью {model_name} пакетами…"
        )
        directory = self._writer.directory
        refiner = self._refiner or LocalWhisper(
            model_name, model_cache_dir(), self.settings.language,
            self.settings.glossary, self.on_status,
        )
        refiner.glossary = self.settings.glossary
        live_entries = self._writer.entries
        source_path = directory / "meeting_audio.wav"
        refinement_path = source_path
        temporary_path: Path | None = None
        protected_ranges = [
            (entry.start_seconds, entry.duration_seconds)
            for entry in live_entries
            if entry.text_edited
        ]
        if protected_ranges:
            temporary_path = directory / ".meeting_audio_for_refinement.wav"
            _write_masked_wave(source_path, temporary_path, protected_ranges)
            refinement_path = temporary_path
            self.on_status(
                "Уточняю общую аудиодорожку; вручную изменённые реплики пропущены…"
            )
        else:
            self.on_status("Уточняю общую аудиодорожку…")
        try:
            refined = refiner.transcribe_file(
                refinement_path,
                cancel_event=self._refinement_cancel,
                **options,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if self._refinement_cancel.is_set():
            raise TranscriptionCancelled("уточнение отменено")
        entries = _merge_refined_with_live(refined, live_entries, self._wall_started_at)
        if not entries:
            self.on_status(
                f"Модель {model_name} не нашла речь; сохранён живой текст"
            )
            return
        entries.sort(
            key=lambda item: (item.start_seconds, item.source != "microphone")
        )
        self._writer.replace_entries(entries)
        final_entries = self._writer.entries
        self.on_reset()
        for entry in final_entries:
            self.on_entry(entry)
        self.on_status(f"Итоговая стенограмма уточнена моделью {model_name}")

    def _timed_entry(
        self,
        source: AudioSource,
        role: str,
        speaker_id: str,
        text: str,
        start_seconds: float,
        duration_seconds: float,
    ) -> TranscriptEntry:
        return TranscriptEntry(
            source=source,
            role=role,
            text=text,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            speaker_id=speaker_id,
            created_at=self._wall_started_at + timedelta(seconds=start_seconds),
        )

    def _on_capture_error(self, source: AudioSource, exc: Exception) -> None:
        label = "микрофона" if source == "microphone" else "звука приложений"
        self.on_error(f"Ошибка захвата {label}: {exc}")


def _merge_refined_with_live(
    segments,
    live_entries: list[TranscriptEntry],
    wall_started_at: datetime,
) -> list[TranscriptEntry]:
    entries: list[TranscriptEntry] = []
    unused = {entry.entry_id for entry in live_entries}
    for segment in segments:
        match = _best_live_match(segment, live_entries, unused)
        if match is not None:
            unused.discard(match.entry_id)
            source = match.source
            role = match.role
            speaker_id = match.speaker_id
            entry_id = match.entry_id
            role_edited = match.role_edited
        else:
            nearest = _nearest_live_entry(segment, live_entries)
            source = nearest.source if nearest is not None else "system"
            role = nearest.role if nearest is not None else "Собеседник"
            speaker_id = nearest.speaker_id if nearest is not None else None
            entry_id = None
            role_edited = bool(nearest and nearest.role_edited)
        kwargs = {}
        if entry_id is not None:
            kwargs["entry_id"] = entry_id
        entries.append(
            TranscriptEntry(
                source=source,
                role=role,
                text=segment.text,
                start_seconds=segment.start_seconds,
                duration_seconds=segment.duration_seconds,
                speaker_id=speaker_id,
                created_at=wall_started_at
                + timedelta(seconds=segment.start_seconds),
                role_edited=role_edited,
                **kwargs,
            )
        )
    return entries


def _best_live_match(segment, entries, unused_ids: set[str]) -> TranscriptEntry | None:
    candidates = [entry for entry in entries if entry.entry_id in unused_ids]
    scored = [(_temporal_score(segment, entry), entry) for entry in candidates]
    plausible = [item for item in scored if item[0] > -2.5]
    if not plausible:
        return None
    return max(plausible, key=lambda item: item[0])[1]


def _nearest_live_entry(segment, entries) -> TranscriptEntry | None:
    if not entries:
        return None
    return max(entries, key=lambda entry: _temporal_score(segment, entry))


def _temporal_score(segment, entry: TranscriptEntry) -> float:
    segment_end = segment.start_seconds + max(segment.duration_seconds, 0.01)
    entry_end = entry.start_seconds + max(entry.duration_seconds, 0.01)
    overlap = max(
        0.0,
        min(segment_end, entry_end) - max(segment.start_seconds, entry.start_seconds),
    )
    segment_center = (segment.start_seconds + segment_end) / 2
    entry_center = (entry.start_seconds + entry_end) / 2
    return overlap * 10.0 - abs(segment_center - entry_center)


def _write_masked_wave(
    source: Path,
    destination: Path,
    ranges: list[tuple[float, float]],
) -> None:
    with wave.open(str(source), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise RuntimeError("Ожидалась монофоническая WAV-дорожка PCM 16 бит")
        sample_rate = reader.getframerate()
        intervals = [
            (
                max(0, int(start * sample_rate)),
                max(0, int((start + max(duration, 0.01)) * sample_rate)),
            )
            for start, duration in ranges
        ]
        with wave.open(str(destination), "wb") as writer:
            writer.setparams(reader.getparams())
            frame_position = 0
            while True:
                raw = reader.readframes(sample_rate)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype="<i2").copy()
                chunk_end = frame_position + samples.size
                for start, end in intervals:
                    local_start = max(0, start - frame_position)
                    local_end = min(samples.size, end - frame_position)
                    if local_start < local_end:
                        samples[local_start:local_end] = 0
                writer.writeframes(samples.tobytes())
                frame_position = chunk_end
