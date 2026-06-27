"""Basic smoke tests for the reorganized project."""

from __future__ import annotations

import unittest

from analysis.loader import prepare_dataset
from config import AnalysisConfig


class SmokeTests(unittest.TestCase):
    """Sanity checks for the project wiring."""

    def test_config_instantiates(self) -> None:
        self.assertIsInstance(AnalysisConfig(), AnalysisConfig)

    def test_dataset_loads(self) -> None:
        frame = prepare_dataset()
        self.assertFalse(frame.empty)
        self.assertIn("archetype", frame.columns)
        self.assertIn("deck_code", frame.columns)


if __name__ == "__main__":
    unittest.main()
