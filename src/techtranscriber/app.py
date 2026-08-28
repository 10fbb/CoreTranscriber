from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from . import __version__
from .audio import DeviceService
from .config import default_output_root, load_settings, model_cache_dir, save_settings
from .diarization import OnlineSpeakerClusterer
from .dictionaries import MAX_PROMPT_TERMS, DictionaryManager
from .dictionary_ui import DictionaryEditorDialog
from .models import AppSettings, DeviceInfo, TranscriptEntry
from .pipeline import MeetingPipeline
from .transcriber import LocalWhisper
from .transcript_editor import TranscriptEditDialog
from .themes import THEME_OPTIONS, theme_colors, theme_stylesheet

ENTRY_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1
SOURCE_ROLE = ENTRY_ID_ROLE + 1


class UiBridge(QObject):
    entry = Signal(object)
    reset_entries = Signal()
    status = Signal(str)
    error = Signal(str)
    stopped = Signal()
    model_ready = Signal(object)
    model_failed = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.dictionary_manager = DictionaryManager()
        self.preloaded_live_models: dict[str, LocalWhisper] = {}
        self.preloaded_refinement_models: dict[str, LocalWhisper] = {}
        self.preloaded_speaker_clusterer: OnlineSpeakerClusterer | None = None
        self.pipeline: MeetingPipeline | None = None
        self.last_session: Path | None = None
        self.elapsed_seconds = 0
        self.bridge = UiBridge()
        self.bridge.entry.connect(self._append_entry)
        self.bridge.reset_entries.connect(self._reset_entries)
        self.bridge.status.connect(self._set_status)
        self.bridge.error.connect(self._show_error)
        self.bridge.stopped.connect(self._after_stop)
        self.bridge.model_ready.connect(self._model_ready)
        self.bridge.model_failed.connect(self._model_failed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._build_ui()
        self._apply_style()
        self._refresh_devices()
        self._refresh_dictionaries()

    def _build_ui(self) -> None:
        self.setWindowTitle("CoreTranscriber — локальная транскрибация")
        self.setMinimumSize(1120, 720)
        self.resize(1400, 860)
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(232)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 24, 18, 20)
        sidebar_layout.setSpacing(8)

        brand = QLabel("CoreTranscriber")
        brand.setObjectName("brand")
        sidebar_layout.addWidget(brand)
        brand_caption = QLabel("Локальная расшифровка")
        brand_caption.setObjectName("sidebarCaption")
        sidebar_layout.addWidget(brand_caption)
        sidebar_layout.addSpacing(26)

        recording_nav = _nav_button("●   Запись", active=True)
        sidebar_layout.addWidget(recording_nav)
        self.dictionaries_nav = _nav_button("▤   Словари")
        self.dictionaries_nav.setToolTip("Открыть редактор технических словарей")
        self.dictionaries_nav.clicked.connect(self._manage_dictionaries)
        sidebar_layout.addWidget(self.dictionaries_nav)
        self.models_nav = _nav_button("↓   Модели")
        self.models_nav.setToolTip("Загрузить выбранные модели заранее")
        self.models_nav.clicked.connect(self._prepare_model)
        sidebar_layout.addWidget(self.models_nav)
        about_nav = _nav_button("ⓘ   О программе")
        about_nav.clicked.connect(self._show_about)
        sidebar_layout.addWidget(about_nav)
        sidebar_layout.addStretch()

        theme_label = QLabel("ОФОРМЛЕНИЕ")
        theme_label.setObjectName("themeLabel")
        sidebar_layout.addWidget(theme_label)
        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("themeSelector")
        for theme_id, label in THEME_OPTIONS:
            self.theme_combo.addItem(label, theme_id)
        selected_theme = self.theme_combo.findData(self.settings.ui_theme)
        self.theme_combo.setCurrentIndex(max(0, selected_theme))
        self.theme_combo.currentIndexChanged.connect(self._theme_changed)
        sidebar_layout.addWidget(self.theme_combo)
        sidebar_layout.addSpacing(6)

        privacy = QLabel("Всё аудио остаётся\nна этом компьютере")
        privacy.setObjectName("privacy")
        privacy.setWordWrap(True)
        sidebar_layout.addWidget(privacy)
        version = QLabel(f"Версия {__version__}")
        version.setObjectName("version")
        sidebar_layout.addWidget(version)
        shell.addWidget(sidebar)

        workspace = QWidget()
        workspace.setObjectName("workspace")
        outer = QVBoxLayout(workspace)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.setSpacing(16)
        shell.addWidget(workspace, 1)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        eyebrow = QLabel("РАБОЧЕЕ ПРОСТРАНСТВО")
        eyebrow.setObjectName("eyebrow")
        title_box.addWidget(eyebrow)
        title = QLabel("Запись встречи")
        title.setObjectName("appTitle")
        subtitle = QLabel("Распознавание русской технической речи в реальном времени")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        timer_caption = QLabel("ДЛИТЕЛЬНОСТЬ")
        timer_caption.setObjectName("timerCaption")
        header.addWidget(timer_caption)
        self.timer_label = QLabel("00:00")
        self.timer_label.setObjectName("timer")
        header.addWidget(self.timer_label)
        outer.addLayout(header)

        content = QWidget()
        content.setObjectName("content")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        outer.addWidget(content, 1)

        settings_card = QFrame()
        settings_card.setObjectName("card")
        settings_card.setMinimumWidth(360)
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(20, 20, 20, 22)
        settings_layout.setSpacing(13)
        settings_layout.addWidget(_section("Параметры встречи"))
        settings_intro = QLabel("Настройте источники звука и качество распознавания")
        settings_intro.setObjectName("hint")
        settings_intro.setWordWrap(True)
        settings_layout.addWidget(settings_intro)

        form = QFormLayout()
        form.setSpacing(10)
        self.meeting_title = QLineEdit("Техническая встреча")
        self.meeting_title.setPlaceholderText("Название встречи")
        form.addRow("Название", self.meeting_title)
        self.mic_combo = QComboBox()
        form.addRow("Микрофон", self.mic_combo)
        self.speaker_combo = QComboBox()
        form.addRow("Звук ПК", self.speaker_combo)
        self.model_combo = QComboBox()
        model_options = (
            ("tiny", "tiny — максимальная скорость"),
            ("base", "base — рекомендуется для CPU"),
            ("small", "small — точнее, но медленнее"),
            ("medium", "medium — не для живой записи на CPU"),
            ("large-v3", "large-v3 — рекомендуется NVIDIA GPU"),
        )
        for model, label in model_options:
            self.model_combo.addItem(label, model)
        self.model_combo.setToolTip(
            "Без NVIDIA CUDA тяжёлые модели живой записи автоматически "
            "заменяются на base; итоговая модель не изменяется"
        )
        selected_model = self.model_combo.findData(self.settings.whisper_model)
        self.model_combo.setCurrentIndex(max(0, selected_model))
        self.model_combo.currentIndexChanged.connect(
            lambda index: self._model_selection_changed(
                str(self.model_combo.itemData(index))
            )
        )
        form.addRow("Модель", self.model_combo)
        settings_layout.addLayout(form)

        self.download_model_button = QPushButton("↓  Подготовить локальные модели")
        self.download_model_button.setObjectName("secondary")
        self.download_model_button.clicked.connect(self._prepare_model)
        settings_layout.addWidget(self.download_model_button)

        self.refine_after_recording = QCheckBox(
            "После остановки уточнить итоговую стенограмму"
        )
        self.refine_after_recording.setChecked(
            self.settings.refine_after_recording
        )
        self.refine_after_recording.setToolTip(
            "Живая стенограмма сохранится; уточнение можно прервать в любой момент"
        )
        self.refine_after_recording.toggled.connect(
            lambda checked: self._refinement_toggled(checked)
        )
        settings_layout.addWidget(self.refine_after_recording)
        self.refinement_combo = QComboBox()
        refinement_options = (
            ("small", "small — быстро, рекомендуется для CPU"),
            ("turbo", "turbo — точнее, но может занять больше 10 минут"),
            ("medium", "medium — медленнее, режим совместимости"),
        )
        for model, label in refinement_options:
            self.refinement_combo.addItem(label, model)
        refinement_index = self.refinement_combo.findData(
            self.settings.refinement_model
        )
        self.refinement_combo.setCurrentIndex(max(0, refinement_index))
        self.refinement_combo.setEnabled(self.refine_after_recording.isChecked())
        self.refinement_combo.currentIndexChanged.connect(
            lambda _: self._model_selection_changed(
                str(self.model_combo.currentData())
            )
        )
        settings_layout.addWidget(self.refinement_combo)

        refresh = QPushButton("↻  Обновить список устройств")
        refresh.setObjectName("secondary")
        refresh.clicked.connect(self._refresh_devices)
        settings_layout.addWidget(refresh)

        settings_layout.addWidget(_divider())
        settings_layout.addWidget(_section("Технические словари"))
        hint = QLabel("Отметьте только темы текущей встречи")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        settings_layout.addWidget(hint)
        self.dictionary_list = QListWidget()
        self.dictionary_list.setMinimumHeight(140)
        self.dictionary_list.itemChanged.connect(self._update_dictionary_count)
        settings_layout.addWidget(self.dictionary_list)
        dictionary_row = QHBoxLayout()
        select_all = QPushButton("Выбрать все")
        select_all.setObjectName("secondary")
        select_all.clicked.connect(lambda: self._set_all_dictionaries(True))
        dictionary_row.addWidget(select_all)
        select_none = QPushButton("Снять все")
        select_none.setObjectName("secondary")
        select_none.clicked.connect(lambda: self._set_all_dictionaries(False))
        dictionary_row.addWidget(select_none)
        manage_dictionaries = QPushButton("Изменить")
        manage_dictionaries.setObjectName("secondary")
        manage_dictionaries.clicked.connect(self._manage_dictionaries)
        dictionary_row.addWidget(manage_dictionaries)
        self.dictionary_count = QLabel("0 терминов")
        self.dictionary_count.setObjectName("hint")
        self.dictionary_count.setToolTip(
            "У Whisper ограничен размер контекстной подсказки; "
            "сначала используются дополнительные и приоритетные термины"
        )
        dictionary_row.addWidget(self.dictionary_count)
        settings_layout.addLayout(dictionary_row)

        settings_layout.addWidget(_divider())
        settings_layout.addWidget(_section("Дополнительные термины"))
        extra_hint = QLabel("Имена, проекты и термины — по одному в строке")
        extra_hint.setObjectName("hint")
        settings_layout.addWidget(extra_hint)
        self.glossary = QPlainTextEdit()
        self.glossary.setPlaceholderText("Kubernetes\nPostgreSQL\nREST API\nDion")
        self.glossary.setPlainText("\n".join(self.settings.glossary))
        self.glossary.textChanged.connect(self._update_dictionary_count)
        self.glossary.setMinimumHeight(85)
        settings_layout.addWidget(self.glossary)

        settings_layout.addWidget(_divider())
        settings_layout.addWidget(_section("Папка записей"))
        path_row = QHBoxLayout()
        self.output_path = QLineEdit(
            str(self.settings.output_root or default_output_root())
        )
        self.output_path.setReadOnly(True)
        choose = QPushButton("…")
        choose.setFixedWidth(42)
        choose.clicked.connect(self._choose_output)
        path_row.addWidget(self.output_path)
        path_row.addWidget(choose)
        settings_layout.addLayout(path_row)
        settings_layout.addStretch()
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setObjectName("settingsScroll")
        settings_scroll.setMinimumWidth(390)
        settings_scroll.setMaximumWidth(470)
        settings_scroll.setWidget(settings_card)
        content_layout.addWidget(settings_scroll)

        transcript_card = QFrame()
        transcript_card.setObjectName("card")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setContentsMargins(20, 20, 20, 18)
        transcript_layout.setSpacing(14)
        transcript_head = QHBoxLayout()
        transcript_title = QVBoxLayout()
        transcript_title.addWidget(_section("Стенограмма"))
        transcript_subtitle = QLabel("Текст появляется здесь во время разговора")
        transcript_subtitle.setObjectName("hint")
        transcript_title.addWidget(transcript_subtitle)
        transcript_head.addLayout(transcript_title)
        transcript_head.addStretch()
        self.transcript_count_label = QLabel("0 реплик")
        self.transcript_count_label.setObjectName("chip")
        transcript_head.addWidget(self.transcript_count_label)
        transcript_layout.addLayout(transcript_head)

        edit_hint = QLabel("Дважды щёлкните по имени участника или тексту, чтобы исправить реплику")
        edit_hint.setObjectName("editorHint")
        edit_hint.setWordWrap(True)
        transcript_layout.addWidget(edit_hint)

        self.table = QTableWidget(0, 3)
        self.table.setObjectName("transcriptTable")
        self.table.setHorizontalHeaderLabels(["ВРЕМЯ", "УЧАСТНИК", "ТЕКСТ"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setWordWrap(True)
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 148)
        self.table.setColumnWidth(1, 150)
        self.table.verticalHeader().setDefaultSectionSize(58)
        self.table.cellDoubleClicked.connect(self._edit_transcript_cell)
        transcript_layout.addWidget(self.table, 1)
        content_layout.addWidget(transcript_card, 1)

        command_bar = QFrame()
        command_bar.setObjectName("commandBar")
        bottom = QHBoxLayout(command_bar)
        bottom.setContentsMargins(16, 12, 12, 12)
        bottom.setSpacing(10)
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        bottom.addWidget(self.status_dot)
        self.status_label = QLabel("Готово к записи")
        self.status_label.setObjectName("status")
        bottom.addWidget(self.status_label, 1)
        self.open_button = QPushButton("Открыть папку")
        self.open_button.setObjectName("secondary")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_session)
        bottom.addWidget(self.open_button)
        self.cancel_refinement_button = QPushButton("Пропустить уточнение")
        self.cancel_refinement_button.setObjectName("secondary")
        self.cancel_refinement_button.setVisible(False)
        self.cancel_refinement_button.clicked.connect(self._cancel_refinement)
        bottom.addWidget(self.cancel_refinement_button)
        self.start_button = QPushButton("●  Начать запись")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumWidth(176)
        self.start_button.clicked.connect(self._toggle_recording)
        bottom.addWidget(self.start_button)
        outer.addWidget(command_bar)

    def _refresh_devices(self) -> None:
        self.mic_combo.clear()
        self.speaker_combo.clear()
        try:
            microphones = DeviceService.microphones()
            speakers = DeviceService.speakers()
            self._fill_devices(self.mic_combo, microphones, self.settings.microphone_id)
            self._fill_devices(self.speaker_combo, speakers, self.settings.speaker_id)
            self._set_status("Аудиоустройства найдены")
        except Exception as exc:
            self._show_error(f"Не удалось получить аудиоустройства: {exc}")

    def _refresh_dictionaries(self) -> None:
        active = set(self.settings.active_dictionaries)
        if self.dictionary_list.count():
            active = set(self._active_dictionary_names())
        self.dictionary_list.blockSignals(True)
        self.dictionary_list.clear()
        for info in self.dictionary_manager.list():
            item = QListWidgetItem(f"{info.title}  ·  {info.term_count}")
            item.setData(Qt.ItemDataRole.UserRole, info.filename)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            state = (
                Qt.CheckState.Checked
                if info.filename in active
                else Qt.CheckState.Unchecked
            )
            item.setCheckState(state)
            self.dictionary_list.addItem(item)
        self.dictionary_list.blockSignals(False)
        self._update_dictionary_count()

    def _active_dictionary_names(self) -> list[str]:
        result: list[str] = []
        for index in range(self.dictionary_list.count()):
            item = self.dictionary_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                result.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return result

    def _update_dictionary_count(self, item=None) -> None:
        names = self._active_dictionary_names()
        extra = [
            line.strip()
            for line in self.glossary.toPlainText().splitlines()
            if line.strip()
        ]
        selected = len(self.dictionary_manager.combine(names, extra))
        available = len(
            self.dictionary_manager.combine(names, extra, max_terms=100_000)
        )
        if available > MAX_PROMPT_TERMS:
            self.dictionary_count.setText(
                f"{selected} приоритетных из {available}"
            )
        else:
            self.dictionary_count.setText(f"{selected} терминов")

    def _manage_dictionaries(self) -> None:
        active = self._active_dictionary_names()
        dialog = DictionaryEditorDialog(self.dictionary_manager, self)
        dialog.exec()
        self.settings.active_dictionaries = active
        self._refresh_dictionaries()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "О CoreTranscriber",
            (
                f"CoreTranscriber {__version__}\n\n"
                "Локальная транскрибация русской технической речи.\n"
                "Записи, словари и модели остаются на вашем компьютере."
            ),
        )

    def _theme_changed(self, index: int) -> None:
        theme_id = str(self.theme_combo.itemData(index))
        if not theme_id:
            return
        self.settings.ui_theme = theme_id
        save_settings(self.settings)
        self._apply_style()
        self._refresh_role_colors()
        self._set_status(f"Оформление: {self.theme_combo.currentText()}")

    def _set_all_dictionaries(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.dictionary_list.blockSignals(True)
        for index in range(self.dictionary_list.count()):
            self.dictionary_list.item(index).setCheckState(state)
        self.dictionary_list.blockSignals(False)
        self._update_dictionary_count()

    def _prepare_model(self) -> None:
        model_name = str(self.model_combo.currentData())
        refinement_model = str(self.refinement_combo.currentData())
        prepare_refinement = self.refine_after_recording.isChecked()
        if self._models_prepared():
            self._set_status("Все выбранные модели уже готовы")
            return
        self.download_model_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.start_button.setEnabled(False)
        self.download_model_button.setText("Подготовка моделей…")
        transcriber = LocalWhisper(
            model_name,
            model_cache_dir(),
            "ru",
            [],
            self.bridge.status.emit,
            realtime=True,
        )
        refiner = (
            LocalWhisper(
                refinement_model,
                model_cache_dir(),
                "ru",
                [],
                self.bridge.status.emit,
            )
            if prepare_refinement
            else None
        )

        def load_job() -> None:
            try:
                transcriber.load()
                if refiner is not None:
                    refiner.load()
                speaker_model = OnlineSpeakerClusterer(
                    model_cache_dir(), on_status=self.bridge.status.emit
                )
                speaker_model.prepare()
                self.bridge.model_ready.emit(
                    (model_name, transcriber, refinement_model, refiner, speaker_model)
                )
            except Exception as exc:
                log_path = _write_model_preparation_error(exc)
                details = f"\nПодробности: {log_path}" if log_path else ""
                self.bridge.model_failed.emit(
                    f"Не удалось подготовить локальные модели: {exc}{details}"
                )

        threading.Thread(target=load_job, name="model-download", daemon=True).start()

    def _model_ready(self, payload: object) -> None:
        model_name, transcriber, refinement_model, refiner, speaker_model = payload
        effective_live_model = transcriber.effective_model_name
        self.preloaded_live_models[effective_live_model] = transcriber
        if refiner is not None:
            self.preloaded_refinement_models[refinement_model] = refiner
        self.preloaded_speaker_clusterer = speaker_model
        if effective_live_model != model_name:
            effective_index = self.model_combo.findData(effective_live_model)
            if effective_index >= 0:
                self.model_combo.setCurrentIndex(effective_index)
        self.model_combo.setEnabled(True)
        self.download_model_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self._model_selection_changed(str(self.model_combo.currentData()))
        if effective_live_model != model_name:
            self._set_status(
                f"Для живой записи выбрана {effective_live_model}; "
                f"{refinement_model} сохранена для итогового уточнения"
            )
        else:
            self._set_status(
                "Модели распознавания и разделения голосов готовы"
            )

    def _model_failed(self, message: str) -> None:
        self.model_combo.setEnabled(True)
        self.download_model_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.download_model_button.setText("↻  Повторить подготовку моделей")
        self._show_error(message)

    def _model_selection_changed(self, model_name: str) -> None:
        if self._models_prepared():
            self.download_model_button.setText("✓  Выбранные модели готовы")
        else:
            self.download_model_button.setText("↓  Подготовить локальные модели")

    def _models_prepared(self) -> bool:
        live_ready = (
            str(self.model_combo.currentData()) in self.preloaded_live_models
        )
        refinement_ready = (
            not self.refine_after_recording.isChecked()
            or str(self.refinement_combo.currentData())
            in self.preloaded_refinement_models
        )
        return (
            live_ready
            and refinement_ready
            and self.preloaded_speaker_clusterer is not None
        )

    def _refinement_toggled(self, checked: bool) -> None:
        self.refinement_combo.setEnabled(checked)
        self._model_selection_changed(str(self.model_combo.currentData()))

    @staticmethod
    def _fill_devices(combo: QComboBox, devices: list[DeviceInfo], selected: str) -> None:
        target = -1
        default = -1
        for index, device in enumerate(devices):
            suffix = " · по умолчанию" if device.is_default else ""
            combo.addItem(device.name + suffix, device.identifier)
            if device.identifier == selected:
                target = index
            if device.is_default:
                default = index
        combo.setCurrentIndex(target if target >= 0 else max(default, 0))

    def _toggle_recording(self) -> None:
        if self.pipeline and self.pipeline.running:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self.mic_combo.currentIndex() < 0 or self.speaker_combo.currentIndex() < 0:
            self._show_error("Выберите микрофон и устройство звука приложений")
            return
        extra_terms = [
            line.strip()
            for line in self.glossary.toPlainText().splitlines()
            if line.strip()
        ]
        active_dictionaries = self._active_dictionary_names()
        self.settings = AppSettings(
            microphone_id=str(self.mic_combo.currentData()),
            speaker_id=str(self.speaker_combo.currentData()),
            whisper_model=str(self.model_combo.currentData()),
            language="ru",
            glossary=extra_terms,
            active_dictionaries=active_dictionaries,
            energy_threshold=self.settings.energy_threshold,
            speaker_threshold=self.settings.speaker_threshold,
            output_root=Path(self.output_path.text()),
            refine_after_recording=self.refine_after_recording.isChecked(),
            refinement_model=str(self.refinement_combo.currentData()),
            ui_theme=self.settings.ui_theme,
        )
        save_settings(self.settings)
        combined_terms = self.dictionary_manager.combine(
            active_dictionaries, extra_terms
        )
        pipeline_settings = replace(self.settings, glossary=combined_terms)
        preloaded = self.preloaded_live_models.get(self.settings.whisper_model)
        if preloaded:
            preloaded.glossary = combined_terms
        refiner = self.preloaded_refinement_models.get(
            self.settings.refinement_model
        )
        if refiner:
            refiner.glossary = combined_terms
        if self.preloaded_speaker_clusterer:
            self.preloaded_speaker_clusterer.reset_session()
        self.pipeline = MeetingPipeline(
            pipeline_settings,
            self.meeting_title.text(),
            self.bridge.entry.emit,
            self.bridge.status.emit,
            self.bridge.error.emit,
            transcriber=preloaded,
            refiner=refiner,
            clusterer=self.preloaded_speaker_clusterer,
            on_reset=self.bridge.reset_entries.emit,
        )
        try:
            self.pipeline.start()
        except Exception as exc:
            self._show_error(f"Не удалось начать запись: {exc}")
            self.pipeline = None
            return
        self.table.setRowCount(0)
        self.elapsed_seconds = 0
        self.timer.start(1000)
        self.start_button.setText("■  Остановить")
        self.start_button.setObjectName("danger")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self._set_controls_enabled(False)

    def _stop_recording(self) -> None:
        if not self.pipeline:
            return
        self.start_button.setEnabled(False)
        self.timer.stop()
        if self.settings.refine_after_recording:
            self.cancel_refinement_button.setEnabled(True)
            self.cancel_refinement_button.setText("Пропустить уточнение")
            self.cancel_refinement_button.setVisible(True)

        def stop_job() -> None:
            try:
                self.pipeline.stop()
                self.last_session = self.pipeline.session_directory
            except Exception as exc:
                self.bridge.error.emit(f"Ошибка при сохранении: {exc}")
            finally:
                self.bridge.stopped.emit()

        threading.Thread(target=stop_job, name="stop-session", daemon=True).start()

    def _after_stop(self) -> None:
        self.cancel_refinement_button.setVisible(False)
        self.start_button.setEnabled(True)
        self.start_button.setText("●  Начать запись")
        self.start_button.setObjectName("primary")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.open_button.setEnabled(bool(self.last_session))
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.meeting_title,
            self.mic_combo,
            self.speaker_combo,
            self.model_combo,
            self.download_model_button,
            self.dictionary_list,
            self.glossary,
            self.refine_after_recording,
            self.refinement_combo,
            self.dictionaries_nav,
            self.models_nav,
        ):
            widget.setEnabled(enabled)
        if enabled:
            self.refinement_combo.setEnabled(
                self.refine_after_recording.isChecked()
            )

    def _cancel_refinement(self) -> None:
        if not self.pipeline:
            return
        self.cancel_refinement_button.setEnabled(False)
        self.cancel_refinement_button.setText("Останавливаю…")
        self.pipeline.cancel_refinement()

    def _append_entry(self, entry: TranscriptEntry) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        timestamp = entry.created_at.strftime("%d.%m.%Y %H:%M:%S")
        time_item = QTableWidgetItem(timestamp)
        time_item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
        role_item = QTableWidgetItem(entry.role)
        role_item.setData(Qt.ItemDataRole.UserRole, entry.speaker_id)
        role_item.setData(ENTRY_ID_ROLE, entry.entry_id)
        role_item.setData(SOURCE_ROLE, entry.source)
        role_item.setForeground(QColor(self._role_color(entry.source)))
        font = QFont()
        font.setBold(True)
        role_item.setFont(font)
        text_item = QTableWidgetItem(entry.text)
        text_item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
        text_item.setToolTip(entry.text)
        self.table.setItem(row, 0, time_item)
        self.table.setItem(row, 1, role_item)
        self.table.setItem(row, 2, text_item)
        self.table.resizeRowToContents(row)
        self.table.scrollToBottom()
        self.transcript_count_label.setText(_plural_replicas(row + 1))

    def _reset_entries(self) -> None:
        self.table.setRowCount(0)
        self.transcript_count_label.setText("0 реплик")

    def _role_color(self, source: str) -> str:
        colors = theme_colors(self.settings.ui_theme)
        return colors["mic" if source == "microphone" else "system"]

    def _refresh_role_colors(self) -> None:
        for row in range(self.table.rowCount()):
            role_item = self.table.item(row, 1)
            if role_item:
                role_item.setForeground(
                    QColor(self._role_color(str(role_item.data(SOURCE_ROLE))))
                )

    def _edit_transcript_cell(self, row: int, column: int) -> None:
        if column == 1:
            self._rename_role(row)
        elif column == 2:
            self._edit_text(row)

    def _rename_role(self, row: int) -> None:
        role_item = self.table.item(row, 1)
        if not role_item:
            return
        speaker_id = role_item.data(Qt.ItemDataRole.UserRole)
        entry_id = role_item.data(ENTRY_ID_ROLE)
        role, accepted = QInputDialog.getText(
            self, "Имя участника", "Введите имя или роль:", text=role_item.text()
        )
        role = role.strip()
        if not accepted or not role:
            return
        if speaker_id:
            for current_row in range(self.table.rowCount()):
                current_item = self.table.item(current_row, 1)
                if (
                    current_item
                    and current_item.data(Qt.ItemDataRole.UserRole) == speaker_id
                ):
                    current_item.setText(role)
            if self.pipeline:
                self.pipeline.rename_speaker(str(speaker_id), role)
        elif self.pipeline and entry_id:
            role_item.setText(role)
            self.pipeline.rename_entry(str(entry_id), role)
        else:
            role_item.setText(role)

    def _edit_text(self, row: int) -> None:
        text_item = self.table.item(row, 2)
        if not text_item:
            return
        entry_id = text_item.data(Qt.ItemDataRole.UserRole)
        dialog = TranscriptEditDialog(text_item.text(), self)
        if not dialog.exec():
            return
        text = dialog.edited_text().strip()
        if not text:
            return
        text_item.setText(text)
        text_item.setToolTip(text)
        self.table.resizeRowToContents(row)
        if self.pipeline and entry_id:
            self.pipeline.edit_entry_text(str(entry_id), text)

    def _choose_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Папка для записей", self.output_path.text()
        )
        if chosen:
            self.output_path.setText(chosen)

    def _open_session(self) -> None:
        if not self.last_session:
            return
        path = str(self.last_session)
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _tick(self) -> None:
        self.elapsed_seconds += 1
        self.timer_label.setText(
            f"{self.elapsed_seconds // 60:02d}:{self.elapsed_seconds % 60:02d}"
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_dot.setStyleSheet(
            f"color: {theme_colors(self.settings.ui_theme)['success']};"
        )

    def _show_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_dot.setStyleSheet(
            f"color: {theme_colors(self.settings.ui_theme)['error']};"
        )
        QMessageBox.warning(self, "CoreTranscriber", message)

    def closeEvent(self, event) -> None:
        if self.pipeline and self.pipeline.running:
            answer = QMessageBox.question(
                self,
                "Остановить запись?",
                "Запись ещё идёт. Остановить её и сохранить стенограмму?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.pipeline.cancel_refinement()
            self.pipeline.stop()
        event.accept()

    def _apply_style(self) -> None:
        self.setStyleSheet(theme_stylesheet(self.settings.ui_theme))
        if hasattr(self, "status_dot"):
            self.status_dot.setStyleSheet(
                f"color: {theme_colors(self.settings.ui_theme)['success']};"
            )


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    return label


def _divider() -> QFrame:
    divider = QFrame()
    divider.setObjectName("divider")
    divider.setFrameShape(QFrame.Shape.HLine)
    return divider


def _nav_button(text: str, active: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("navActive" if active else "nav")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def _plural_replicas(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        noun = "реплика"
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        noun = "реплики"
    else:
        noun = "реплик"
    return f"{count} {noun}"


def _write_model_preparation_error(exc: Exception) -> Path | None:
    try:
        path = model_cache_dir() / "model_preparation.log"
        timestamp = datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{timestamp}] {type(exc).__name__}: {exc}\n")
            handle.write("".join(traceback.format_exception(exc)))
        return path
    except OSError:
        return None


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CoreTranscriber")
    app.setOrganizationName("CoreTranscriber")
    window = MainWindow()
    window.show()
    return app.exec()
