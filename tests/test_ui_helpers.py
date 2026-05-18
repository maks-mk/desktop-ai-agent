import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.main_window_state import StreamEventRouter
from ui.streaming import StreamEvent
from ui.theme import build_stylesheet
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

    def test_summary_progress_ring_updates_tooltip_from_payload(self):
        ring = SummaryProgressRing()
        self.addCleanup(ring.deleteLater)

        ring.set_summary_progress(
            {
                "estimated_tokens": 6400,
                "threshold": 8000,
                "remaining_tokens": 1600,
                "progress": 0.8,
                "will_summarize": False,
            }
        )

        self.assertTrue(ring.isVisible())
        self.assertIn("1,600", ring.toolTip())
        self.assertIn("6,400 / 8,000", ring.toolTip())
