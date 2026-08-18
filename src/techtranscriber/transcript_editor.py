from __future__ import annotations

from PySide6.QtGui import QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TranscriptEditDialog(QDialog):
    """Resizable transcript editor with visual wrapping and text search."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Редактирование стенограммы")
        self.setMinimumSize(680, 420)
        self.resize(860, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        instruction = QLabel(
            "Исправьте текст выбранной реплики. Текст автоматически переносится "
            "по ширине окна."
        )
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Найти слово или фразу в реплике")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self.find_next)
        search_row.addWidget(self.search_input, 1)
        self.find_button = QPushButton("Найти далее")
        self.find_button.clicked.connect(self.find_next)
        search_row.addWidget(self.find_button)
        layout.addLayout(search_row)

        self.search_status = QLabel("")
        self.search_status.setObjectName("hint")
        layout.addWidget(self.search_status)

        self.editor = QPlainTextEdit(text)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.editor.setWordWrapMode(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
        )
        self.editor.setTabChangesFocus(True)
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.editor.setTextCursor(cursor)
        layout.addWidget(self.editor, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.editor.setFocus()

    def edited_text(self) -> str:
        return self.editor.toPlainText()

    def find_next(self) -> None:
        query = self.search_input.text()
        if not query:
            self.search_status.setText("")
            self.search_input.setFocus()
            return
        if self.editor.find(query):
            self.search_status.setText("")
            self.editor.setFocus()
            return
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.editor.setTextCursor(cursor)
        if self.editor.find(query):
            self.search_status.setText("Поиск продолжен с начала реплики")
            self.editor.setFocus()
            return
        self.search_status.setText("Совпадений не найдено")
        self.search_input.setFocus()
