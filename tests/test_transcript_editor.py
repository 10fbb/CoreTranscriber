from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtGui import QTextCursor, QTextOption
from PySide6.QtWidgets import QApplication, QPlainTextEdit

from techtranscriber.transcript_editor import TranscriptEditDialog


class TranscriptEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_editor_wraps_long_text_to_window_width(self) -> None:
        dialog = TranscriptEditDialog("Длинная техническая реплика " * 20)
        try:
            self.assertEqual(
                dialog.editor.lineWrapMode(),
                QPlainTextEdit.LineWrapMode.WidgetWidth,
            )
            self.assertEqual(
                dialog.editor.wordWrapMode(),
                QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere,
            )
            self.assertGreaterEqual(dialog.minimumWidth(), 680)
            self.assertGreaterEqual(dialog.minimumHeight(), 420)
        finally:
            dialog.close()

    def test_search_finds_phrase_and_wraps_to_start(self) -> None:
        dialog = TranscriptEditDialog(
            "Первый технический термин. Второй технический термин."
        )
        try:
            dialog.search_input.setText("Первый")
            cursor = dialog.editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            dialog.editor.setTextCursor(cursor)

            dialog.find_next()

            self.assertEqual(dialog.editor.textCursor().selectedText(), "Первый")
            self.assertIn("с начала", dialog.search_status.text())
        finally:
            dialog.close()

    def test_edited_text_preserves_manual_paragraphs(self) -> None:
        dialog = TranscriptEditDialog("Первая строка\nВторая строка")
        try:
            self.assertEqual(
                dialog.edited_text(), "Первая строка\nВторая строка"
            )
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
