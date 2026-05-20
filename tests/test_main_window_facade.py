import unittest
from types import SimpleNamespace
from unittest import mock

import main as agent_cli
import ui.main_window as public_main_window
import ui.window_components.main_window as window_components
from ui.window_components.main_window import MainWindow as WindowComponentsMainWindow


class MainWindowFacadeTests(unittest.TestCase):
    def test_public_facade_reexports_window_symbols(self):
        self.assertIs(public_main_window.MainWindow, WindowComponentsMainWindow)
        self.assertIs(agent_cli.MainWindow, WindowComponentsMainWindow)
        self.assertIs(agent_cli.ModelSettingsDialog, public_main_window.ModelSettingsDialog)
        self.assertIs(agent_cli.QFileDialog, public_main_window.QFileDialog)
        self.assertIs(agent_cli.QMessageBox, public_main_window.QMessageBox)
        self.assertIs(
            agent_cli._configure_windows_app_user_model_id,
            public_main_window._configure_windows_app_user_model_id,
        )
        self.assertIs(agent_cli._build_app_icon, public_main_window._build_app_icon)

    def test_windows_app_user_model_id_is_configured_on_windows(self):
        calls = []
        fake_ctypes = SimpleNamespace(
            windll=SimpleNamespace(
                shell32=SimpleNamespace(
                    SetCurrentProcessExplicitAppUserModelID=lambda app_id: calls.append(app_id)
                )
            )
        )

        with (
            mock.patch.object(window_components.sys, "platform", "win32"),
            mock.patch.dict("sys.modules", {"ctypes": fake_ctypes}),
        ):
            window_components._configure_windows_app_user_model_id()

        self.assertEqual(calls, [window_components.WINDOWS_APP_USER_MODEL_ID])

    def test_windows_app_user_model_id_is_skipped_off_windows(self):
        with mock.patch.object(window_components.sys, "platform", "linux"):
            window_components._configure_windows_app_user_model_id()


if __name__ == "__main__":
    unittest.main()
