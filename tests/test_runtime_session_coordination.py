import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from core.session_store import SessionSnapshot, SessionStore
from ui.runtime import AgentRunWorker


class RuntimeSessionCoordinationTests(unittest.IsolatedAsyncioTestCase):
    def _session_snapshot(self, project_path: Path) -> SessionSnapshot:
        return SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(project_path),
        )

    async def test_current_project_path_uses_runtime_facade_path(self):
        worker = AgentRunWorker()
        project_path = Path("D:/tmp/project-a")

        with mock.patch("ui.runtime.Path.cwd", return_value=project_path):
            self.assertEqual(worker._current_project_path(), str(project_path))

    async def test_try_change_workdir_uses_runtime_facade_os(self):
        worker = AgentRunWorker()
        project_path = Path("D:/tmp/project-b")

        with mock.patch("ui.runtime.Path.cwd", return_value=Path("D:/tmp/other")), mock.patch(
            "ui.runtime.os.chdir"
        ) as chdir_mock:
            changed, error = worker._try_change_workdir(str(project_path))

        self.assertTrue(changed)
        self.assertEqual(error, "")
        chdir_mock.assert_called_once_with(str(project_path))

    async def test_emit_session_payload_keeps_public_shape(self):
        tmp_dir = Path.cwd() / ".tmp_tests" / uuid4().hex
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp_dir, ignore_errors=True))

        worker = AgentRunWorker()
        session_path = tmp_dir / "session.json"
        worker.store = SessionStore(session_path)
        project_path = tmp_dir
        worker.current_session = worker.store.new_session(
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            project_path=str(project_path),
        )
        worker.config = mock.Mock(
            provider="openai",
            openai_model="gpt-4o",
            checkpoint_backend="sqlite",
            enable_approvals=True,
            debug=False,
        )
        worker.model_profiles = {"active_profile": None, "profiles": []}
        worker.model_capabilities = {"image_input_supported": False}
        worker.tool_registry = mock.Mock(
            tools=[],
            tool_metadata={},
            mcp_server_status=[],
            checkpoint_info={"resolved_backend": "sqlite"},
            get_runtime_status_lines=mock.Mock(return_value=[]),
        )
        worker.agent_app = None

        payload = await worker._emit_session_payload(include_transcript=False)

        self.assertIn("snapshot", payload)
        self.assertIn("sessions", payload)
        self.assertEqual(payload["active_session_id"], worker.current_session.session_id)
