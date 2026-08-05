from __future__ import annotations

import os
import subprocess
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .dictionaries import DictionaryManager


class DictionaryEditorDialog(QDialog):
    def __init__(self, manager: DictionaryManager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.current_filename: str | None = None
        self.setWindowTitle("Управление словарями")
        self.resize(900, 620)
        self._build_ui()
        self._reload()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Локальные словари")
        title.setObjectName("section")
        layout.addWidget(title)
        hint = QLabel(
            "Каждая строка — отдельный термин. Файлы находятся на вашем компьютере "
            "и не отправляются в облако."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._select_item)
        left_layout.addWidget(self.list_widget)
        create_button = QPushButton("Создать словарь")
        create_button.setObjectName("primary")
        create_button.clicked.connect(self._create)
        left_layout.addWidget(create_button)
        delete_button = QPushButton("Удалить")
        delete_button.setObjectName("secondary")
        delete_button.clicked.connect(self._delete)
        left_layout.addWidget(delete_button)
        folder_button = QPushButton("Открыть папку словарей")
        folder_button.setObjectName("secondary")
        folder_button.clicked.connect(self._open_folder)
        left_layout.addWidget(folder_button)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        self.file_label = QLabel("Выберите словарь")
        self.file_label.setObjectName("section")
        right_layout.addWidget(self.file_label)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Введите термины — по одному в строке")
        self.editor.textChanged.connect(self._update_count)
        right_layout.addWidget(self.editor, 1)
        row = QHBoxLayout()
        self.count_label = QLabel("0 терминов")
        self.count_label.setObjectName("hint")
        row.addWidget(self.count_label)
        row.addStretch()
        save_button = QPushButton("Сохранить изменения")
        save_button.setObjectName("primary")
        save_button.clicked.connect(self._save)
        row.addWidget(save_button)
        right_layout.addLayout(row)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_button = QPushButton("Готово")
        close_button.setObjectName("secondary")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

    def _reload(self, select_filename: str | None = None) -> None:
        self.list_widget.clear()
        selected_item = None
        for info in self.manager.list():
            item = QListWidgetItem(f"{info.title}  ·  {info.term_count}")
            item.setData(Qt.ItemDataRole.UserRole, info.filename)
            self.list_widget.addItem(item)
            if info.filename == select_filename:
                selected_item = item
        if selected_item:
            self.list_widget.setCurrentItem(selected_item)
        elif self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def _select_item(self, current, previous) -> None:
        if not current:
            self.current_filename = None
            self.editor.clear()
            return
        self.current_filename = str(current.data(Qt.ItemDataRole.UserRole))
        self.file_label.setText(self.manager.display_name(self.current_filename))
        self.editor.setPlainText(self.manager.read_raw(self.current_filename))

    def _create(self) -> None:
        name, accepted = QInputDialog.getText(
            self, "Новый словарь", "Название словаря:"
        )
        if not accepted or not name.strip():
            return
        filename = self.manager.create(name)
        self._reload(filename)
        self.editor.setFocus()

    def _save(self) -> None:
        if not self.current_filename:
            return
        self.manager.save(self.current_filename, self.editor.toPlainText())
        filename = self.current_filename
        self._reload(filename)

    def _delete(self) -> None:
        if not self.current_filename:
            return
        title = self.manager.display_name(self.current_filename)
        answer = QMessageBox.question(
            self,
            "Удалить словарь?",
            f"Словарь «{title}» будет удалён с компьютера.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.manager.delete(self.current_filename)
        self.current_filename = None
        self._reload()

    def _update_count(self) -> None:
        count = len(
            [line for line in self.editor.toPlainText().splitlines() if line.strip()]
        )
        self.count_label.setText(f"{count} терминов")

    def _open_folder(self) -> None:
        path = str(self.manager.directory)
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

