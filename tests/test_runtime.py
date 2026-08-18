from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.runtime import configure_windowed_runtime
from techtranscriber.app import _write_model_preparation_error


class RuntimeTests(unittest.TestCase):
    def test_windowed_exe_gets_safe_output_streams(self) -> None:
        with patch.object(sys, "stdout", None), patch.object(
            sys, "stderr", None
        ), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            os.environ.pop("TQDM_DISABLE", None)

            configure_windowed_runtime()

            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            sys.stdout.write("model download progress")
            sys.stderr.write("model warning")
            self.assertEqual(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"], "1")
            self.assertEqual(os.environ["TQDM_DISABLE"], "1")

    def test_model_preparation_error_writes_full_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch(
            "techtranscriber.app.model_cache_dir", return_value=Path(temp)
        ):
            try:
                raise RuntimeError("download failed")
            except RuntimeError as exc:
                path = _write_model_preparation_error(exc)

            self.assertIsNotNone(path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("RuntimeError: download failed", content)
            self.assertIn("Traceback", content)


if __name__ == "__main__":
    unittest.main()
