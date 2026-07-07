import logging
import shutil
import unittest
import warnings
from pathlib import Path

from core.logging_config import SensitiveDataFilter, setup_logging


class SensitiveDataFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = SensitiveDataFilter()

    def test_filter_redacts_mapping_values_by_sensitive_key(self):
        record = logging.LogRecord(
            "agent",
            logging.INFO,
            __file__,
            10,
            "Profile payload: %s",
            ({"api_key": "sk-demo-secret", "nested": {"token": "plain-secret-token"}},),
            None,
        )

        self.assertTrue(self.filter.filter(record))
        rendered = record.getMessage()

        self.assertIn("sk-d...<redacted>", rendered)
        self.assertIn("plai...<redacted>", rendered)
        self.assertNotIn("sk-demo-secret", rendered)
        self.assertNotIn("plain-secret-token", rendered)

    def test_filter_redacts_secrets_in_plain_text_message_and_extra_fields(self):
        record = logging.LogRecord(
            "agent",
            logging.INFO,
            __file__,
            20,
            "Authorization: Bearer supersecrettoken api_key=AIzaSyDemoSecretValue",
            (),
            None,
        )
        record.api_key = "gm-demo-secret"

        self.assertTrue(self.filter.filter(record))

        rendered = record.getMessage()
        self.assertIn("Bearer supe...<redacted>", rendered)
        self.assertIn("api_key=AIza...<redacted>", rendered)
        self.assertEqual(record.api_key, "gm-d...<redacted>")
        self.assertNotIn("supersecrettoken", rendered)

    def test_setup_logging_writes_reasoning_debug_to_separate_file(self):
        log_dir = Path.cwd() / ".tmp_tests" / f"reasoning_log_{id(self)}"
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True)
        try:
            log_file = log_dir / "agent.log"
            setup_logging(level="INFO", log_file=log_file, reasoning_debug_enabled=True)

            logger = logging.getLogger("agent.reasoning_debug")
            logger.debug("reasoning api_key=sk-demo-secret provider=%s", "openrouter")
            for handler in logger.handlers:
                handler.flush()

            reasoning_log = log_file.parent / "reasoning_debug.log"
            self.assertTrue(reasoning_log.exists())
            content = reasoning_log.read_text(encoding="utf-8")
            self.assertIn("provider=openrouter", content)
            self.assertIn("api_key=sk-d...<redacted>", content)
            self.assertNotIn("sk-demo-secret", content)
        finally:
            for handler in list(logging.getLogger("agent.reasoning_debug").handlers):
                handler.close()
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_setup_logging_keeps_reasoning_debug_disabled_by_default(self):
        log_dir = Path.cwd() / ".tmp_tests" / f"reasoning_log_disabled_{id(self)}"
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True)
        try:
            log_file = log_dir / "agent.log"
            setup_logging(level="INFO", log_file=log_file)

            logger = logging.getLogger("agent.reasoning_debug")
            logger.debug("reasoning provider=openrouter")
            for handler in logger.handlers:
                handler.flush()

            self.assertFalse((log_file.parent / "reasoning_debug.log").exists())
        finally:
            for handler in list(logging.getLogger("agent.reasoning_debug").handlers):
                handler.close()
            shutil.rmtree(log_dir, ignore_errors=True)

    def test_setup_logging_suppresses_known_pydantic_content_block_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            setup_logging(level="INFO", log_file=None)

            warnings.warn_explicit(
                (
                    "Pydantic serializer warnings:\n"
                    "  PydanticSerializationUnexpectedValue(Expected `str` - serialized value may not be as expected "
                    "[field_name='content', input_value=[{'type': 'thinking'}, {'type': 'text', 'text': 'П'}], "
                    "input_type=list])"
                ),
                UserWarning,
                "pydantic/main.py",
                475,
                module="pydantic.main",
            )
            warnings.warn_explicit(
                "Pydantic serializer warnings: something else",
                UserWarning,
                "pydantic/main.py",
                475,
                module="pydantic.main",
            )

        self.assertEqual(len(caught), 1)
        self.assertIn("something else", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
