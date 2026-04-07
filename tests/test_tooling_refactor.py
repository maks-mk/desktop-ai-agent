import importlib
import re
import shutil
import sys
import unittest
from unittest import mock
from pathlib import Path
from uuid import uuid4
from types import SimpleNamespace

from core.config import AgentConfig
from core.multimodal import DEFAULT_MODEL_CAPABILITIES
from core.validation import validate_tool_result
from tools import process_tools
from tools.tool_registry import ToolRegistry


class ToolingRefactorTests(unittest.IsolatedAsyncioTestCase):
    def _make_config(self, **overrides):
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MCP_CONFIG_PATH": Path(__file__).resolve().parents[1] / "tests" / "missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_SYSTEM_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    async def test_tool_registry_preserves_filesystem_delete_tools(self):
        registry = ToolRegistry(self._make_config())
        await registry.load_all()
        names = {tool.name for tool in registry.tools}
        self.assertIn("safe_delete_file", names)
        self.assertIn("safe_delete_directory", names)

    async def test_tool_registry_does_not_keep_selector_catalog_state(self):
        registry = ToolRegistry(self._make_config())
        await registry.load_all()
        self.assertFalse(hasattr(registry, "selector_catalog"))

    async def test_tool_registry_fallback_delete_tools_without_filesystem(self):
        registry = ToolRegistry(self._make_config(ENABLE_FILESYSTEM_TOOLS=False))
        await registry.load_all()
        names = [tool.name for tool in registry.tools]
        self.assertEqual(names, ["safe_delete_file", "safe_delete_directory", "request_user_input"])

    def test_tool_registry_initializes_model_capabilities_slot(self):
        registry = ToolRegistry(self._make_config())
        self.assertEqual(registry.model_capabilities, DEFAULT_MODEL_CAPABILITIES)

    def test_mcp_metadata_keeps_safe_tools_read_only(self):
        metadata = ToolRegistry._infer_mcp_metadata("context7:resolve-library-id")

        self.assertTrue(metadata.read_only)
        self.assertFalse(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertFalse(metadata.requires_approval)
        self.assertTrue(metadata.networked)

    def test_mcp_metadata_flags_write_like_tools_without_default_approval(self):
        metadata = ToolRegistry._infer_mcp_metadata("filesystem:write_file")

        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertTrue(metadata.requires_approval)
        self.assertTrue(metadata.networked)

    def test_mcp_metadata_flags_execution_like_tools_without_default_approval(self):
        metadata = ToolRegistry._infer_mcp_metadata("terminal:run_command")

        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertTrue(metadata.requires_approval)
        self.assertTrue(metadata.networked)

    def test_mcp_metadata_respects_safe_readonly_hint(self):
        tool = SimpleNamespace(
            name="acme_search_docs",
            description="Search documentation pages",
            metadata={"readOnlyHint": True, "openWorldHint": True},
        )

        metadata = ToolRegistry._infer_mcp_metadata(tool)

        self.assertTrue(metadata.read_only)
        self.assertFalse(metadata.mutating)
        self.assertFalse(metadata.destructive)
        self.assertFalse(metadata.requires_approval)

    def test_mcp_metadata_uses_destructive_hint_for_approval(self):
        tool = SimpleNamespace(
            name="acme_workspace",
            description="Manage workspace entries",
            metadata={"destructiveHint": True},
        )

        metadata = ToolRegistry._infer_mcp_metadata(tool)

        self.assertFalse(metadata.read_only)
        self.assertTrue(metadata.mutating)
        self.assertTrue(metadata.destructive)
        self.assertTrue(metadata.requires_approval)

    def test_validation_supports_delete_argument_aliases(self):
        tmp = self._workspace_tempdir()
        file_path = tmp / "data.txt"
        file_path.write_text("demo", encoding="utf-8")
        error = validate_tool_result("safe_delete_file", {"file_path": str(file_path)}, "Success")
        self.assertIsNotNone(error)
        self.assertIn("still exists", error)

        dir_path = tmp / "folder"
        dir_path.mkdir()
        error = validate_tool_result("safe_delete_directory", {"dir_path": str(dir_path)}, "Success")
        self.assertIsNotNone(error)
        self.assertIn("still exists", error)

    def test_max_file_size_numeric_value_is_bytes(self):
        config = self._make_config(MAX_FILE_SIZE="4096")
        self.assertEqual(config.max_file_size, 4096)

    def test_max_file_size_supports_explicit_units(self):
        self.assertEqual(self._make_config(MAX_FILE_SIZE="4MB").max_file_size, 4_000_000)
        self.assertEqual(self._make_config(MAX_FILE_SIZE="300MiB").max_file_size, 300 * 1024 * 1024)

    def test_max_file_size_rejects_invalid_strings(self):
        with self.assertRaises(Exception):
            self._make_config(MAX_FILE_SIZE="300MBps")

    def test_logging_env_keys_are_loaded_via_agent_config(self):
        config = self._make_config(LOG_LEVEL="debug", LOG_FILE="logs/custom-agent.log")
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.log_file.name, "custom-agent.log")

    def test_text_utils_import_does_not_require_prompt_toolkit(self):
        import core.text_utils as text_utils

        with mock.patch.dict(sys.modules, {"prompt_toolkit": None, "prompt_toolkit.key_binding": None}):
            reloaded = importlib.reload(text_utils)
            self.assertTrue(callable(reloaded.prepare_markdown_for_render))

    def test_run_background_process_rejects_shell_operators(self):
        result = process_tools.run_background_process.invoke({"command": "python -c \"print(1)\" && whoami"})
        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("Shell operators are not allowed", result)

    def test_run_background_process_rejects_cwd_outside_workspace(self):
        tmp = self._workspace_tempdir()
        process_tools.set_working_directory(str(tmp))
        result = process_tools.run_background_process.invoke(
            {"command": [sys.executable, "-c", "print('ok')"], "cwd": ".."}
        )
        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("ACCESS DENIED", result)

    def test_run_background_process_accepts_argument_list(self):
        tmp = self._workspace_tempdir()
        process_tools.set_working_directory(str(tmp))
        result = process_tools.run_background_process.invoke(
            {"command": [sys.executable, "-c", "import time; time.sleep(30)"], "cwd": "."}
        )
        self.assertIn("Success: Process started with PID", result)
        match = re.search(r"PID (\d+)", result)
        self.assertIsNotNone(match)
        stop_result = process_tools.stop_background_process.invoke({"pid": int(match.group(1))})
        self.assertIn("Success:", stop_result)

    def test_find_process_by_port_uses_net_connections(self):
        class FakePsutil:
            class AccessDenied(Exception):
                pass

            class NoSuchProcess(Exception):
                pass

            class ZombieProcess(Exception):
                pass

            @staticmethod
            def net_connections(kind="inet"):
                return [SimpleNamespace(laddr=SimpleNamespace(port=8000), pid=4321)]

            @staticmethod
            def Process(pid):
                return SimpleNamespace(name=lambda: "python")

        with mock.patch.object(process_tools, "psutil", FakePsutil):
            result = process_tools.find_process_by_port.invoke({"port": 8000})
        self.assertIn("Found process 'python' (PID: 4321) on port 8000.", result)

    def test_find_process_by_port_returns_non_error_when_port_is_free(self):
        class FakePsutil:
            class AccessDenied(Exception):
                pass

            class NoSuchProcess(Exception):
                pass

            class ZombieProcess(Exception):
                pass

            @staticmethod
            def net_connections(kind="inet"):
                return []

        with mock.patch.object(process_tools, "psutil", FakePsutil):
            result = process_tools.find_process_by_port.invoke({"port": 8000})
        self.assertEqual(result, "No process found listening on port 8000.")
        self.assertNotIn("ERROR[", result)


if __name__ == "__main__":
    unittest.main()
