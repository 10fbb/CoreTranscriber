from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techtranscriber.themes import (
    DEFAULT_THEME,
    THEME_IDS,
    THEME_OPTIONS,
    theme_colors,
    theme_stylesheet,
)


class ThemeTests(unittest.TestCase):
    def test_all_five_designs_are_available(self) -> None:
        self.assertEqual(len(THEME_OPTIONS), 5)
        self.assertEqual(len(THEME_IDS), 5)
        self.assertEqual(DEFAULT_THEME, "modern_dark")

    def test_each_design_has_a_complete_unique_stylesheet(self) -> None:
        styles = [theme_stylesheet(theme_id) for theme_id, _ in THEME_OPTIONS]
        self.assertEqual(len(set(styles)), 5)
        for stylesheet in styles:
            self.assertIn("QFrame#sidebar", stylesheet)
            self.assertIn("QTableWidget#transcriptTable", stylesheet)
            self.assertNotIn("$", stylesheet)

    def test_unknown_design_uses_default_palette(self) -> None:
        self.assertEqual(
            theme_stylesheet("unknown"), theme_stylesheet(DEFAULT_THEME)
        )
        self.assertEqual(theme_colors("unknown"), theme_colors(DEFAULT_THEME))


if __name__ == "__main__":
    unittest.main()
