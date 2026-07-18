import os
import shutil
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.main_window_state import StreamEventRouter
from ui.streaming import StreamEvent
from ui.theme import build_stylesheet
from ui.widgets.composer import ComposerTextEdit
from ui.widgets.foundation import SummaryProgressRing


class UiHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_stream_event_router_dispatches_known_events(self):
        handler = mock.Mock()
        router = StreamEventRouter({"status_changed": handler})

        router.dispatch(StreamEvent("status_changed", {"label": "Working"}))
        router.dispatch(StreamEvent("ignored", {"label": "noop"}))

        handler.assert_called_once_with({"label": "Working"})

    def test_build_stylesheet_is_cached(self):
        first = build_stylesheet()
        second = build_stylesheet()

        self.assertIs(first, second)
        self.assertIn("QStatusBar", first)

    def test_model_image_checkbox_indicator_has_visible_border(self):
        stylesheet = build_stylesheet()

        self.assertIn("QCheckBox#ModelSupportsImagesCheckbox::indicator", stylesheet)
        self.assertIn("QCheckBox#ModelSupportsImagesCheckbox::indicator:checked", stylesheet)
        self.assertIn("border: 1px solid", stylesheet)
        self.assertNotIn("checkbox-check.svg", stylesheet)

    def test_summary_progress_ring_updates_tooltip_from_payload(self):
        ring = SummaryProgressRing()
        self.addCleanup(ring.deleteLater)

        ring.set_summary_progress(
            {
                "estimated_tokens": 6400,
                "threshold": 8000,
                "remaining_tokens": 1600,
                "reserved_tokens": 3000,
                "summary_tokens": 850,
                "provider_input_tokens": 229094,
                "progress": 0.8,
                "will_summarize": False,
            }
        )

        self.assertTrue(ring.isVisible())
        self.assertIn("1,600", ring.toolTip())
        self.assertIn("6,400 / 8,000", ring.toolTip())
        self.assertIn("3,000 reserved", ring.toolTip())
        self.assertIn("850 estimated tokens from compressed memory", ring.toolTip())
        self.assertIn("229,094 provider-reported tokens", ring.toolTip())

    def test_composer_file_index_refreshes_empty_mention_after_directory_change(self):
        composer = ComposerTextEdit()
        self.addCleanup(composer.deleteLater)
        temp_root = Path.cwd() / ".tmp_tests" / f"composer-index-{time.time_ns()}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: temp_root.exists() and shutil.rmtree(temp_root, ignore_errors=True))

        with mock.patch("ui.widgets.composer.Path.cwd", return_value=temp_root):
            (temp_root / "first.py").write_text("print('first')", encoding="utf-8")
            composer._ensure_file_index(force_refresh=True)
            self.assertEqual([row["relative"] for row in composer._filter_mention_candidates("")], ["first.py"])

            (temp_root / "late.py").write_text("print('late')", encoding="utf-8")
            composer._on_file_index_directory_changed(str(temp_root))
            rows = composer._filter_mention_candidates("")

        self.assertIn("first.py", [row["relative"] for row in rows])
        self.assertIn("late.py", [row["relative"] for row in rows])
