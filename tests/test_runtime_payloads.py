import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ui.runtime_payloads import (
    append_project_label,
    build_transcript_payload,
    build_user_choice_payload,
    generate_chat_title,
)


class RuntimePayloadTests(unittest.TestCase):
    def test_generate_chat_title_strips_common_prefixes_and_limits_length(self):
        self.assertEqual(
            generate_chat_title("Помоги скачать и настроить Apache на Windows"),
            "Скачать и настроить Apache на Windows",
        )
        self.assertEqual(generate_chat_title("   \n   "), "New Chat")
        self.assertTrue(generate_chat_title("сделай " + ("очень длинный запрос " * 10)).endswith("…"))

    def test_append_project_label_uses_last_two_segments(self):
        project_path = Path("D:/work/client/demo-app")
        self.assertEqual(append_project_label("New Chat", project_path), "New Chat [client/demo-app]")

    def test_build_user_choice_payload_marks_recommended_entry(self):
        payload = build_user_choice_payload(
            {
                "question": "Continue?",
                "recommended": "option_2",
                "options": [
                    {"label": "Skip", "submit_text": "skip"},
                    {"label": "Retry", "submit_text": "retry"},
                ],
            }
        )

        self.assertEqual(payload["question"], "Continue?")
        self.assertEqual(payload["recommended_key"], "Retry")
        self.assertEqual([item["recommended"] for item in payload["options"]], [False, True])

    def test_build_transcript_payload_restores_turns_attachments_and_tool_args(self):
        payload = build_transcript_payload(
            {
                "summary": "compressed",
                "messages": [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Покажи diff"},
                            {
                                "type": "image",
                                "path": "C:/tmp/screenshot.png",
                                "mime_type": "image/png",
                                "file_name": "screenshot.png",
                                "attachment_id": "img-1",
                            },
                        ]
                    ),
                    AIMessage(
                        content="Смотрю изменения",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "name": "edit_file",
                                "args": {"path": "app.py", "oldText": "a", "newText": "b"},
                            }
                        ],
                    ),
                    ToolMessage(
                        content="```diff\n-a\n+b\n```",
                        tool_call_id="call-1",
                        name="edit_file",
                    ),
                ],
            }
        )

        self.assertIn("compressed automatically", payload["summary_notice"])
        self.assertEqual(len(payload["turns"]), 1)
        turn = payload["turns"][0]
        self.assertEqual(turn["user_text"], "Покажи diff")
        self.assertEqual(turn["attachments"][0]["id"], "img-1")
        tool_block = turn["blocks"][-1]["payload"]
        self.assertEqual(tool_block["args"]["path"], "app.py")
        self.assertEqual(tool_block["diff"], "-a\n+b")
