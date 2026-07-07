import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.summarize_policy import estimate_tokens, should_summarize
from ui.runtime_payloads import (
    append_project_label,
    build_pending_plan_review_payload,
    build_summary_progress_payload,
    build_transcript_payload,
    build_ui_payload,
    build_user_choice_payload,
    generate_chat_title,
)


def _pending_plan_state():
    return {
        "plan_status": "pending_approval",
        "current_plan": {
            "id": "plan-1",
            "version": 1,
            "summary": "Внести изменение",
            "steps": [
                {
                    "id": "inspect",
                    "title": "Проверить код",
                    "description": "Найти точку изменения.",
                    "status": "pending",
                }
            ],
            "risks": [],
            "assumptions": [],
            "verification": ["Запустить тесты"],
            "estimated_tools": [],
            "estimated_files": [],
            "complexity": "low",
            "status": "pending_approval",
            "active_step_id": "",
        },
    }


class RuntimePayloadTests(unittest.TestCase):
    def test_build_summary_progress_payload_reports_context_budget(self):
        config = type("Config", (), {"summary_threshold": 100, "summary_keep_last": 4})()

        payload = build_summary_progress_payload(
            config,
            {"messages": [HumanMessage(content="hello world")], "summary": ""},
        )

        self.assertGreater(payload["estimated_tokens"], 0)
        self.assertEqual(payload["threshold"], 100)
        self.assertEqual(payload["remaining_tokens"], max(0, 100 - payload["estimated_tokens"]))
        self.assertGreaterEqual(payload["progress"], 0.0)
        self.assertLessEqual(payload["progress"], 1.0)
        self.assertFalse(payload["will_summarize"])

    def test_build_summary_progress_payload_includes_reserved_context_tokens(self):
        config = type(
            "Config",
            (),
            {
                "summary_threshold": 100,
                "summary_keep_last": 4,
                "summary_reserved_tokens": 25,
            },
        )()
        messages = [HumanMessage(content="hello world")]

        payload = build_summary_progress_payload(config, {"messages": messages, "summary": ""})

        self.assertEqual(payload["reserved_tokens"], 25)
        self.assertEqual(payload["estimated_tokens"], estimate_tokens(messages) + 25)
        self.assertEqual(payload["remaining_tokens"], max(0, 100 - payload["estimated_tokens"]))

    def test_should_summarize_uses_reserved_context_tokens(self):
        messages = [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="new question"),
        ]

        self.assertTrue(
            should_summarize(
                messages,
                threshold=50,
                keep_last=1,
                reserved_tokens=1000,
            )
        )

    def test_build_summary_progress_payload_marks_ready_to_summarize(self):
        config = type("Config", (), {"summary_threshold": 10, "summary_keep_last": 1})()
        messages = [
            HumanMessage(content="token " * 1200),
            AIMessage(content="old answer " * 1200),
            HumanMessage(content="new question"),
        ]

        payload = build_summary_progress_payload(config, {"messages": messages})

        self.assertEqual(payload["threshold"], 10)
        self.assertEqual(payload["remaining_tokens"], 0)
        self.assertEqual(payload["progress"], 1.0)
        self.assertTrue(payload["will_summarize"])

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

    def test_build_pending_plan_review_payload_restores_checkpointed_plan_choice(self):
        payload = build_pending_plan_review_payload(_pending_plan_state())

        self.assertIsNotNone(payload)
        self.assertEqual(payload["choice_type"], "plan_review")
        self.assertEqual(payload["current_plan"]["summary"], "Внести изменение")
        self.assertEqual(payload["current_plan"]["verification"], ["Запустить тесты"])
        self.assertEqual(
            [option["submit_text"] for option in payload["options"]],
            ["implement", "revise", "rebuild", "cancel"],
        )

    def test_build_pending_plan_review_payload_ignores_invalid_checkpoint_plan(self):
        payload = build_pending_plan_review_payload(
            {"plan_status": "pending_approval", "current_plan": {"summary": "broken"}}
        )

        self.assertIsNone(payload)

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

    def test_build_transcript_payload_restores_implementation_plan_as_collapsed_plan_block(self):
        payload = build_transcript_payload(
            {
                "current_plan": {
                    "id": "plan-1",
                    "version": 1,
                    "summary": "Small UI update",
                    "status": "completed",
                    "steps": [
                        {"id": "inspect", "title": "Inspect", "description": "Read UI code", "status": "completed"},
                        {"id": "fix", "title": "Fix", "description": "Keep card", "status": "completed"},
                    ],
                    "risks": [],
                    "assumptions": [],
                    "estimated_tools": [],
                    "estimated_files": [],
                    "complexity": "low",
                    "active_step_id": "",
                },
                "messages": [
                    HumanMessage(content="implement plan"),
                    AIMessage(content="<implementation_plan>\n# Plan\n- Inspect\n- Fix\n</implementation_plan>"),
                ],
            }
        )

        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["plan"])
        self.assertEqual(blocks[0]["payload"]["plan_markdown"], "# Plan\n- Inspect\n- Fix")
        self.assertEqual(blocks[0]["payload"]["current_plan"]["id"], "plan-1")
        self.assertFalse(blocks[0]["actions_visible"])
        self.assertTrue(blocks[0]["implemented"])

    def test_build_transcript_payload_strips_implementation_plan_tags_from_assistant_text(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="implement plan"),
                    AIMessage(content="Before\n<implementation_plan>\n# Plan\n</implementation_plan>\nAfter"),
                ],
            }
        )

        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["plan", "assistant"])
        self.assertEqual(blocks[0]["payload"]["plan_markdown"], "# Plan")
        self.assertEqual(blocks[1]["markdown"], "Before\n\nAfter")
        self.assertNotIn("implementation_plan", blocks[1]["markdown"])

    def test_build_transcript_payload_hides_plan_step_completion_marker(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="continue"),
                    AIMessage(content="Шаг выполнен.\n<!--plan-step-complete:s1-->"),
                ],
            }
        )

        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["assistant"])
        self.assertEqual(blocks[0]["markdown"], "Шаг выполнен.")
        self.assertNotIn("plan-step-complete", blocks[0]["markdown"])

    def test_build_transcript_payload_does_not_parse_assistant_thought_markdown(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Сделай вывод"),
                    AIMessage(content="<think>Проверяю ограничения.</think>Готово"),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        assistant_block = payload["turns"][0]["blocks"][0]
        self.assertEqual(assistant_block["type"], "assistant")
        self.assertEqual(assistant_block["markdown"], "Готово")
        self.assertNotIn("thought_markdown", assistant_block)

    def test_build_transcript_payload_ignores_structured_assistant_reasoning(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Проверь"),
                    AIMessage(
                        content=[
                            {"type": "reasoning", "text": "Сначала сверю ограничения."},
                            {"type": "text", "text": "Итог готов."},
                        ]
                    ),
                ]
            }
        )

        assistant_block = payload["turns"][0]["blocks"][0]
        self.assertEqual(assistant_block["type"], "assistant")
        self.assertEqual(assistant_block["markdown"], "Итог готов.")
        self.assertNotIn("thought_markdown", assistant_block)

    def test_build_transcript_payload_ignores_reasoning_details_text(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Проверь"),
                    AIMessage(
                        content=[
                            {"type": "reasoning.text", "text": "Скрытое рассуждение."},
                            {"type": "text", "text": "Итог готов."},
                        ]
                    ),
                ]
            }
        )

        assistant_block = payload["turns"][0]["blocks"][0]
        self.assertEqual(assistant_block["markdown"], "Итог готов.")
        self.assertNotIn("Скрытое", assistant_block["markdown"])

    def test_build_transcript_payload_restores_hidden_internal_notice_as_notice_block(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Проверь завершение"),
                    AIMessage(
                        content="internal handoff",
                        additional_kwargs={
                            "agent_internal": {
                                "kind": "tool_issue_handoff",
                                "visible_in_ui": False,
                                "ui_notice": "Нужен новый запрос.",
                            }
                        },
                    ),
                ]
            }
        )

        turn = payload["turns"][0]
        self.assertEqual(turn["user_text"], "Проверь завершение")
        self.assertEqual(
            turn["blocks"],
            [{"type": "notice", "message": "Нужен новый запрос.", "level": "warning"}],
        )

    def test_build_transcript_payload_appends_last_run_stats_to_final_turn(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Подведи итог"),
                    AIMessage(content="Готово."),
                ],
            },
            last_run_stats="3.1s  ↓ 5328  ↑ 106",
        )

        turn = payload["turns"][0]
        self.assertEqual([block["type"] for block in turn["blocks"]], ["assistant", "stats"])
        self.assertEqual(turn["blocks"][-1]["stats"], "3.1s  ↓ 5328  ↑ 106")

    def test_build_transcript_payload_does_not_attach_stats_to_empty_trailing_turn(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Первый запрос"),
                    AIMessage(content="Первый ответ"),
                    HumanMessage(content="Второй запрос"),
                ],
            },
            last_run_stats="2.0s  ↓ 100  ↑ 20",
        )

        self.assertEqual([block["type"] for block in payload["turns"][0]["blocks"]], ["assistant"])
        self.assertEqual(payload["turns"][1]["blocks"], [])

    def test_build_user_choice_payload_includes_default_selection(self):
        payload = build_user_choice_payload(
            {
                "kind": "user_choice",
                "question": "Введите ключ API или выберите другой вариант:",
                "options": [
                    "Ввести ключ API",
                    "Пропустить проверку и вернуть скрипт",
                    "Завершить проверку",
                ],
                "recommended": "Ввести ключ API",
            }
        )

        self.assertEqual(payload["kind"], "user_choice")
        self.assertEqual(payload["recommended_key"], "Ввести ключ API")
        recommended = [option for option in payload["options"] if option["recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["submit_text"], "Ввести ключ API")


class RuntimePayloadAsyncTests(unittest.IsolatedAsyncioTestCase):
    def _config(self):
        return SimpleNamespace(
            provider="openai",
            openai_model="gpt-4o",
            gemini_model="gemini-pro",
            summary_threshold=100,
            summary_keep_last=4,
            summary_reserved_tokens=0,
            checkpoint_backend="memory",
            enable_approvals=True,
            debug=False,
        )

    def _snapshot(self):
        return SimpleNamespace(
            session_id="session-1",
            thread_id="thread-1",
            title="Chat",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
            checkpoint_backend="memory",
            checkpoint_target="memory",
            last_run_stats="",
            approval_mode="prompt",
        )

    async def _build_payload(self, *, tasks):
        class FakeApp:
            async def aget_state(self, _config):
                return SimpleNamespace(values=_pending_plan_state(), tasks=tasks)

        store = SimpleNamespace(list_sessions=lambda: [])
        registry = SimpleNamespace(
            tools=[],
            metadata={},
            model_capabilities={},
            checkpoint_info={},
            get_runtime_status_lines=lambda: [],
        )
        return await build_ui_payload(
            self._config(),
            registry,
            store,
            self._snapshot(),
            agent_app=FakeApp(),
        )

    async def test_build_ui_payload_restores_plan_choice_only_with_pending_interrupt(self):
        interrupt = SimpleNamespace(value={"choice_type": "plan_review"})
        payload = await self._build_payload(tasks=[SimpleNamespace(interrupts=(interrupt,))])

        self.assertIn("pending_user_choice", payload)
        self.assertEqual(payload["pending_user_choice"]["choice_type"], "plan_review")

    async def test_build_ui_payload_skips_stale_pending_plan_without_interrupt(self):
        payload = await self._build_payload(tasks=[])

        self.assertNotIn("pending_user_choice", payload)
        self.assertNotIn("checkpoint_notice", payload)
