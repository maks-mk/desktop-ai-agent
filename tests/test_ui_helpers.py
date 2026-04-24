import unittest
from unittest import mock

from ui.main_window_state import StreamEventRouter
from ui.streaming import StreamEvent
from ui.theme import build_stylesheet


class UiHelperTests(unittest.TestCase):
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
