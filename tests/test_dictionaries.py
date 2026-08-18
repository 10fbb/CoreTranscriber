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
            self.assertEqual(terms, ["Dion", "API", "эквайринг", "REST API", "PAN"])
            manager.delete(first)
            self.assertFalse((Path(temp) / first).exists())

    def test_acquiring_terms_are_split_into_focused_dictionaries(self) -> None:
        root = ROOT / "src" / "techtranscriber" / "default_dictionaries"
        expected = {
            "11_Эквайринг_и_банковский_процессинг.txt": "эквайринг",
            "12_EMV_терминалы_и_авторизация.txt": "ISO 8583",
            "14_PCI_токенизация_и_3DS.txt": "3DS Server",
            "15_Регулярные_кошельки_и_СБП.txt": "СБП",
        }
        for filename, term in expected.items():
            self.assertIn(term, (root / filename).read_text(encoding="utf-8"))

    def test_packaged_dictionaries_have_no_case_insensitive_duplicates(self) -> None:
        root = ROOT / "src" / "techtranscriber" / "default_dictionaries"
        for path in root.glob("*.txt"):
            terms = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(
                len(terms), len({term.casefold() for term in terms}), path.name
            )

    def test_defaults_are_installed_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("techtranscriber.dictionaries.app_data_dir", return_value=root):
                manager = DictionaryManager(root / "Словари")
                self.assertEqual(len(manager.list()), 16)
                acquiring = "11_Эквайринг_и_банковский_процессинг.txt"
                manager.delete(acquiring)
                DictionaryManager(root / "Словари")
                self.assertFalse((root / "Словари" / acquiring).exists())

    def test_dictionary_update_preserves_custom_terms(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dictionary_dir = root / "Словари"
            dictionary_dir.mkdir()
            filename = "01_Общие_IT_и_платформы.txt"
            (dictionary_dir / filename).write_text(
                "Мой внутренний сервис\nDion\n", encoding="utf-8"
            )
            with patch("techtranscriber.dictionaries.app_data_dir", return_value=root):
                manager = DictionaryManager(dictionary_dir)

            terms = manager.load(filename)
            self.assertEqual(terms[0], "Мой внутренний сервис")
            self.assertIn("platform engineering", terms)

    def test_prompt_terms_prioritize_extras_and_balance_dictionaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = DictionaryManager(Path(temp), install_defaults=False)
            first = manager.create("Первый")
            second = manager.create("Второй")
            manager.save(first, "API\nPostgreSQL\nKubernetes\n")
            manager.save(second, "эквайринг\nпроцессинг\nСБП\n")

            terms = manager.combine(
                [first, second], ["Dion"], max_terms=5
            )

            self.assertEqual(
                terms,
                ["Dion", "API", "эквайринг", "PostgreSQL", "процессинг"],
            )


if __name__ == "__main__":
    unittest.main()
