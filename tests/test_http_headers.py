import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.http_headers import load_openai_headers


class OpenAIHeadersTests(unittest.TestCase):
    def test_missing_file_returns_empty_for_standard_sdk_headers(self):
        with TemporaryDirectory() as temp_dir:
            result = load_openai_headers(Path(temp_dir) / "missing.json")

        self.assertEqual(result, {})

    def test_json_values_override_and_add_headers(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "headers.json"
            path.write_text(
                json.dumps({"User-Agent": "CustomAgent/1.0", "x-custom": "enabled", "ignored": 42}),
                encoding="utf-8",
            )

            result = load_openai_headers(path)

        self.assertEqual(result, {"User-Agent": "CustomAgent/1.0", "x-custom": "enabled"})

    def test_invalid_json_returns_empty(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "headers.json"
            path.write_text("{invalid", encoding="utf-8")

            result = load_openai_headers(path)

        self.assertEqual(result, {})

    def test_non_object_json_returns_empty(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "headers.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

            result = load_openai_headers(path)

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
