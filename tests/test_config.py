from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber import __version__
from techtranscriber.config import app_data_dir, default_output_root, load_settings


class ConfigTests(unittest.TestCase):
    def test_fresh_install_uses_coretranscriber_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            local_app_data = Path(temp) / "LocalAppData"
            documents_home = Path(temp) / "User"
            with patch.dict(os.environ, {"LOCALAPPDATA": str(local_app_data)}):
                self.assertEqual(
                    app_data_dir(), local_app_data / "CoreTranscriber"
                )
            with patch(
                "techtranscriber.config.Path.home", return_value=documents_home
            ):
                self.assertEqual(
                    default_output_root(),
                    documents_home / "Documents" / "CoreTranscriber",
                )

    def test_legacy_app_data_is_reused_after_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            local_app_data = Path(temp)
            legacy = local_app_data / "TechTranscriber"
            legacy.mkdir()
            with patch.dict(os.environ, {"LOCALAPPDATA": str(local_app_data)}):
                self.assertEqual(app_data_dir(), legacy)

    def test_version_is_0_7_0(self) -> None:
        self.assertEqual(__version__, "0.7.0")

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
            self.assertEqual(settings.settings_revision, 4)
            self.assertTrue(settings.refine_after_recording)
            self.assertEqual(settings.refinement_model, "small")
            self.assertEqual(settings.ui_theme, "modern_dark")

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
            self.assertEqual(settings.refinement_model, "small")

    def test_valid_theme_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "output_root": temp,
                        "settings_revision": 4,
                        "ui_theme": "glass_fluent",
                    }
                ),
                encoding="utf-8",
            )
            with patch("techtranscriber.config.settings_path", return_value=path):
                settings = load_settings()

            self.assertEqual(settings.ui_theme, "glass_fluent")

    def test_unknown_theme_falls_back_to_modern_dark(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "output_root": temp,
                        "settings_revision": 4,
                        "ui_theme": "missing-theme",
                    }
                ),
                encoding="utf-8",
            )
            with patch("techtranscriber.config.settings_path", return_value=path):
                settings = load_settings()

            self.assertEqual(settings.ui_theme, "modern_dark")


if __name__ == "__main__":
    unittest.main()
