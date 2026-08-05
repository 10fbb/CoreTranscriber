from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from .models import AudioPacket, AudioSource, DeviceInfo, Utterance


def _soundcard():
    try:
        import soundcard as sc
    except ImportError as exc:
        raise RuntimeError("Компонент SoundCard не установлен") from exc
    return sc


class DeviceService:
    @staticmethod
    def microphones() -> list[DeviceInfo]:
        sc = _soundcard()
        default = sc.default_microphone()
        default_name = getattr(default, "name", "") if default else ""
        devices: list[DeviceInfo] = []
        for mic in sc.all_microphones(include_loopback=False):
            name = str(mic.name)
            devices.append(DeviceInfo(name, name, name == default_name))
        return devices

    @staticmethod
    def speakers() -> list[DeviceInfo]:
        sc = _soundcard()
        default = sc.default_speaker()
        default_name = getattr(default, "name", "") if default else ""
        devices: list[DeviceInfo] = []
        for speaker in sc.all_speakers():
            name = str(speaker.name)
            devices.append(DeviceInfo(name, name, name == default_name))
        return devices

    @staticmethod
    def microphone(identifier: str):
        sc = _soundcard()
        devices = sc.all_microphones(include_loopback=False)
        return _exact_or_default(devices, identifier, sc.default_microphone())

    @staticmethod
    def loopback(identifier: str):
        sc = _soundcard()
        speakers = sc.all_speakers()
        speaker = _exact_or_default(speakers, identifier, sc.default_speaker())
        return sc.get_microphone(str(speaker.name), include_loopback=True)


def _exact_or_default(devices, identifier: str, default):
    for device in devices:
        if str(device.name) == identifier:
            return device
    if default is None:
        raise RuntimeError("Аудиоустройство не найдено")
    return default


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        source: AudioSource,
        recorder_device,
        on_packet: Callable[[AudioPacket], None],
        on_error: Callable[[AudioSource, Exception], None],
        sample_rate: int = 48_000,
        block_frames: int = 2_400,
    ) -> None:
        super().__init__(name=f"capture-{source}", daemon=True)
        self.source = source
        self.recorder_device = recorder_device
        self.on_packet = on_packet
        self.on_error = on_error
        self.sample_rate = sample_rate
        self.block_frames = block_frames
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            with self.recorder_device.recorder(
                samplerate=self.sample_rate,
                blocksize=self.block_frames,
            ) as recorder:
                while not self._stop_event.is_set():
                    data = np.asarray(
                        recorder.record(numframes=self.block_frames), dtype=np.float32
                    )
                    if data.ndim == 2:
                        data = data.mean(axis=1)
                    data = np.nan_to_num(data.reshape(-1), copy=False)
                    self.on_packet(
                        AudioPacket(
                            source=self.source,
                            samples=data,
                            sample_rate=self.sample_rate,
                            captured_at=time.monotonic(),
                        )
                    )
        except Exception as exc:  # device errors are platform-specific
            if not self._stop_event.is_set():
                self.on_error(self.source, exc)


class SpeechSegmenter:
    """Energy-based utterance detector with pre-roll and silence closing."""

    def __init__(
        self,
        source: AudioSource,
        session_started_at: float,
        on_utterance: Callable[[Utterance], None],
        energy_threshold: float = 0.008,
        target_rate: int = 16_000,
        pre_roll_seconds: float = 0.35,
        end_silence_seconds: float = 0.65,
        min_duration_seconds: float = 0.55,
        max_duration_seconds: float = 14.0,
    ) -> None:
        self.source = source
        self.session_started_at = session_started_at
        self.on_utterance = on_utterance
        self.energy_threshold = energy_threshold
        self.target_rate = target_rate
        self.pre_roll_seconds = pre_roll_seconds
        self.end_silence_seconds = end_silence_seconds
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self._pre_roll: deque[tuple[np.ndarray, int, float]] = deque()
        self._active: list[np.ndarray] = []
        self._active_rate = 48_000
        self._active_start = 0.0
        self._silence_frames = 0
        self._active_frames = 0

    def accept(self, packet: AudioPacket) -> None:
        samples = np.asarray(packet.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))

        if not self._active:
            self._pre_roll.append((samples.copy(), packet.sample_rate, packet.captured_at))
            self._trim_pre_roll(packet.sample_rate)
            if rms >= self.energy_threshold:
                self._active_rate = packet.sample_rate
                self._active_start = self._pre_roll[0][2]
                self._active = [part for part, _, _ in self._pre_roll]
                self._active_frames = sum(part.size for part in self._active)
                self._pre_roll.clear()
            return

        self._active.append(samples.copy())
        self._active_frames += samples.size
        if rms < self.energy_threshold:
            self._silence_frames += samples.size
        else:
            self._silence_frames = 0

        duration = self._active_frames / self._active_rate
        silence = self._silence_frames / self._active_rate
        if duration >= self.max_duration_seconds or (
            duration >= self.min_duration_seconds and silence >= self.end_silence_seconds
        ):
            self._emit()

    def flush(self) -> None:
        if self._active_frames / max(self._active_rate, 1) >= self.min_duration_seconds:
            self._emit()
        self._pre_roll.clear()

    def _trim_pre_roll(self, sample_rate: int) -> None:
        max_frames = int(self.pre_roll_seconds * sample_rate)
        current = sum(part.size for part, _, _ in self._pre_roll)
        while self._pre_roll and current > max_frames:
            part, _, _ = self._pre_roll.popleft()
            current -= part.size

    def _emit(self) -> None:
        if not self._active:
            return
        raw = np.concatenate(self._active)
        duration = raw.size / self._active_rate
        converted = resample_mono(raw, self._active_rate, self.target_rate)
        self.on_utterance(
            Utterance(
                source=self.source,
                samples=converted,
                sample_rate=self.target_rate,
                start_seconds=max(0.0, self._active_start - self.session_started_at),
                duration_seconds=duration,
            )
        )
        self._active = []
        self._active_frames = 0
        self._silence_frames = 0


def resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or samples.size < 2:
        return samples
    target_size = max(1, round(samples.size * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, target_size, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)

