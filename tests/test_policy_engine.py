import unittest

from core.policy_engine import PolicyEngine, classify_shell_command


class PolicyEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PolicyEngine()

    def test_classify_shell_command_detects_inspect_only(self):
        profile = classify_shell_command("Get-Process | Select-Object Id,ProcessName")
        self.assertTrue(profile["inspect_only"])
        self.assertFalse(profile["mutating"])
        self.assertFalse(profile["long_running_service"])

    def test_classify_shell_command_detects_network_probes_as_inspect_only(self):
        for command in (
            "ping google.com",
            "Test-NetConnection google.com -Port 443",
            "curl -I https://ya.ru",
        ):
            profile = classify_shell_command(command)
            self.assertTrue(profile["inspect_only"], command)
            self.assertFalse(profile["mutating"], command)

    def test_classify_shell_command_detects_foreground_service(self):
        profile = classify_shell_command("npm exec -- npx http-server -p 8080")
        self.assertTrue(profile["long_running_service"])
        self.assertTrue(profile["mutating"])

    def test_classify_shell_command_detects_http_write_as_mutating(self):
        profile = classify_shell_command("curl -X POST https://example.com -d 'x=1'")
        self.assertTrue(profile["mutating"])
        self.assertFalse(profile["inspect_only"])

    def test_tool_call_allowed_for_inspect_turn(self):
        decision = self.engine.tool_call_allowed_for_turn(
            {"name": "read_file", "args": {"path": "README.md"}},
            inspect_only=True,
            tool_is_read_only=lambda name: name == "read_file",
        )
        self.assertTrue(decision.allowed)

    def test_cli_exec_network_probe_allowed_in_inspect_turn(self):
        decision = self.engine.tool_call_allowed_for_turn(
            {"name": "cli_exec", "args": {"command": "ping google.com; curl -I https://ya.ru"}},
            inspect_only=True,
            tool_is_read_only=lambda _name: False,
        )
        self.assertTrue(decision.allowed)

    def test_cli_exec_mutating_command_blocked_in_inspect_turn(self):
        decision = self.engine.tool_call_allowed_for_turn(
            {"name": "cli_exec", "args": {"command": "taskkill /IM node.exe /F"}},
            inspect_only=True,
            tool_is_read_only=lambda _name: False,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "cli_exec_non_inspect")

    def test_choose_fallback_tool_names_prefers_read_only(self):
        chosen = self.engine.choose_fallback_tool_names(
            preferred_tool_names=["run_background_process", "find_process_by_port"],
            recent_tool_names=["find_process_by_port", "stop_background_process"],
            all_tool_names=["run_background_process", "find_process_by_port", "stop_background_process"],
            prefer_read_only=True,
            tool_is_read_only=lambda name: name == "find_process_by_port",
        )
        self.assertEqual(chosen, ["find_process_by_port"])

    def test_evaluate_turn_detects_read_only_action_request(self):
        decision = self.engine.evaluate_turn(
            task="Проверь статус процесса на порту 8080",
            messages=[],
            current_turn_id=1,
            is_internal_retry=lambda _msg: False,
        )
        self.assertTrue(decision.inspect_only)
        self.assertTrue(decision.requires_operational_evidence)
        self.assertTrue(decision.should_force_tools)
        self.assertEqual(decision.intent, "inspect")


if __name__ == "__main__":
    unittest.main()
