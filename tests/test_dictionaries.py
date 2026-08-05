from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.dictionaries import DictionaryManager


class DictionaryTests(unittest.TestCase):
    def test_create_edit_combine_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = DictionaryManager(Path(temp), install_defaults=False)
            first = manager.create("Первый словарь")
            second = manager.create("Банковский")
            manager.save(first, "API\nREST API\nAPI\n")
            manager.save(second, "эквайринг\nPAN\n")
            terms = manager.combine([first, second], ["Dion", "API"])
            self.assertEqual(terms, ["Dion", "API", "REST API", "эквайринг", "PAN"])
            manager.delete(first)
            self.assertFalse((Path(temp) / first).exists())

    def test_default_acquiring_dictionary_is_packaged(self) -> None:
        path = (
            ROOT
            / "src"
            / "techtranscriber"
            / "default_dictionaries"
            / "11_Эквайринг_и_банковский_процессинг.txt"
        )
        content = path.read_text(encoding="utf-8")
        for term in ("эквайринг", "ISO 8583", "3DS Server", "PCI DSS", "СБП"):
            self.assertIn(term, content)

    def test_defaults_are_installed_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("techtranscriber.dictionaries.app_data_dir", return_value=root):
                manager = DictionaryManager(root / "Словари")
                self.assertEqual(len(manager.list()), 11)
                acquiring = "11_Эквайринг_и_банковский_процессинг.txt"
                manager.delete(acquiring)
                DictionaryManager(root / "Словари")
                self.assertFalse((root / "Словари" / acquiring).exists())


if __name__ == "__main__":
    unittest.main()
