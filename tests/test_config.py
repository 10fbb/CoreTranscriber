from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_existing_cpu_install_migrates_from_medium_to_base_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "whisper_model": "medium",
                        "output_root": temp,
                    }
                ),
                encoding="utf-8",
            )
            with patch("techtranscriber.config.settings_path", return_value=path):
                settings = load_settings()

            self.assertEqual(settings.whisper_model, "base")
            self.assertEqual(settings.settings_revision, 2)

    def test_explicit_model_choice_is_preserved_after_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "whisper_model": "small",
                        "output_root": temp,
                        "settings_revision": 2,
                    }
                ),
                encoding="utf-8",
            )
            with patch("techtranscriber.config.settings_path", return_value=path):
                settings = load_settings()

            self.assertEqual(settings.whisper_model, "small")


if __name__ == "__main__":
    unittest.main()
