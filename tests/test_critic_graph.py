import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import create_agent_workflow
from core.config import AgentConfig
from core.nodes import AgentNodes


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, context):
        self.invocations.append(context)
        if not self.responses:
            return AIMessage(
                content="STATUS: FINISHED\nREASON: fallback\nNEXT_STEP: NONE\nCONTROL: FINISH_TURN"
            )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ProviderSafeFakeLLM(FakeLLM):
    async def ainvoke(self, context):
        last_visible = next(
            (message for message in reversed(context) if not isinstance(message, SystemMessage)),
            None,
        )
        if isinstance(last_visible, AIMessage):
            raise AssertionError("provider-unsafe assistant-last context")
        return await super().ainvoke(context)


class StrictToolOrderFakeLLM(FakeLLM):
    async def ainvoke(self, context):
        last_visible = None
        for message in context:
            if isinstance(message, SystemMessage):
                continue
            if isinstance(last_visible, ToolMessage) and isinstance(message, HumanMessage):
                raise AssertionError("unexpected role 'user' after role 'tool'")
            last_visible = message
        return await super().ainvoke(context)


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.description = f"Fake tool {name}"
        self.result = result
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if callable(self.result):
            return self.result(args)
        return self.result


class CriticGraphTests(unittest.IsolatedAsyncioTestCase):
    def _make_config(
        self,
        *,
        model_supports_tools=True,
        max_loops=6,
        max_retries=3,
        retry_delay=0,
        critic_mode="always",
        critic_max_retries_per_turn=1,
    ):
        return AgentConfig(
            provider="openai",
            openai_api_key="test-key",
            model_supports_tools=model_supports_tools,
            max_loops=max_loops,
            max_retries=max_retries,
            retry_delay=retry_delay,
            critic_mode=critic_mode,
            critic_max_retries_per_turn=critic_max_retries_per_turn,
            prompt_path=Path(__file__).resolve().parents[1] / "prompt.txt",
        )

    def _build_app(
        self,
        *,
        agent_responses,
        critic_responses,
        tools=None,
        model_supports_tools=True,
        max_loops=6,
        max_retries=3,
        retry_delay=0,
        critic_mode="always",
        critic_max_retries_per_turn=1,
        agent_llm_cls=FakeLLM,
    ):
        config = self._make_config(
            model_supports_tools=model_supports_tools,
            max_loops=max_loops,
            max_retries=max_retries,
            retry_delay=retry_delay,
            critic_mode=critic_mode,
            critic_max_retries_per_turn=critic_max_retries_per_turn,
        )
        agent_llm = agent_llm_cls(agent_responses)
        critic_llm = FakeLLM(critic_responses)
        nodes = AgentNodes(
            config=config,
            llm=critic_llm,
            tools=tools or [],
            llm_with_tools=agent_llm,
        )
        workflow = create_agent_workflow(
            nodes,
            config,
            tools_enabled=bool(tools) and model_supports_tools,
        )
        app = workflow.compile()
        return app, agent_llm, critic_llm, nodes

    def _initial_state(self, task="Проверь задачу"):
        return {
            "messages": [HumanMessage(content=task)],
            "steps": 0,
            "token_usage": {},
            "current_task": task,
            "critic_status": "",
            "critic_source": "",
            "critic_feedback": "",
            "turn_outcome": "",
            "retry_instruction": "",
            "turn_id": 1,
            "open_tool_issue": None,
        }

    def _assert_feedback_hidden(self, messages):
        joined = "\n".join(str(message.content) for message in messages)
        self.assertNotIn("Task incomplete.", joined)
        self.assertNotIn("Task appears complete based on the tool results.", joined)
        self.assertNotIn("Task may still be incomplete.", joined)

    def _last_model_visible(self, context):
        return next((message for message in reversed(context) if not isinstance(message, SystemMessage)), None)

    async def test_agent_final_answer_finishes_turn_via_critic_control(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[AIMessage(content="Задача выполнена.")],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: FINISHED\n"
                        "REASON: Goal achieved.\n"
                        "NEXT_STEP: NONE\n"
                        "CONTROL: FINISH_TURN"
                    )
                )
            ],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Скажи готово"), config={"recursion_limit": 24})

        self.assertEqual(result["critic_status"], "FINISHED")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["messages"][-1].content, "Задача выполнена.")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)

    async def test_critic_retry_uses_internal_human_message_and_provider_safe_order(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(content="Черновой результат."),
                AIMessage(content="Итог подтверждён."),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: Explicit verification is missing.\n"
                        "NEXT_STEP: verify the result directly\n"
                        "CONTROL: RETRY_AGENT"
                    )
                ),
                AIMessage(
                    content=(
                        "STATUS: FINISHED\n"
                        "REASON: Verification is now present.\n"
                        "NEXT_STEP: NONE\n"
                        "CONTROL: FINISH_TURN"
                    )
                ),
            ],
            tools=[],
            model_supports_tools=False,
            agent_llm_cls=ProviderSafeFakeLLM,
        )

        result = await app.ainvoke(self._initial_state("Сделай и проверь"), config={"recursion_limit": 36})

        self.assertEqual(result["messages"][-1].content, "Итог подтверждён.")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(len(critic_llm.invocations), 2)

        retry_context = [
            message
            for message in agent_llm.invocations[1]
            if isinstance(message, SystemMessage) and "RETRY CONTEXT:" in str(message.content)
        ]
        self.assertFalse(retry_context)

        last_visible = self._last_model_visible(agent_llm.invocations[1])
        self.assertIsInstance(last_visible, HumanMessage)
        self.assertEqual(last_visible.additional_kwargs["agent_internal"]["kind"], "retry_instruction")
        self.assertIn("Why retry:", str(last_visible.content))

        internal_retry_messages = [
            message for message in result["messages"] if isinstance(message, HumanMessage) and message.additional_kwargs
        ]
        self.assertFalse(internal_retry_messages)
        self._assert_feedback_hidden(result["messages"])

    async def test_tools_flow_reaches_final_answer_before_running_critic(self):
        tool = FakeTool("demo_tool", "Deployment complete and service responds with 200 OK.")
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "demo_tool", "args": {"action": "deploy"}, "id": "tc-2"}],
                ),
                AIMessage(content="Деплой завершён, сервис отвечает 200 OK."),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: FINISHED\n"
                        "REASON: Final answer delivered.\n"
                        "NEXT_STEP: NONE\n"
                        "CONTROL: FINISH_TURN"
                    )
                )
            ],
            tools=[tool],
        )

        result = await app.ainvoke(self._initial_state("Задеплой сервис"), config={"recursion_limit": 36})

        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(len(critic_llm.invocations), 1)
        self.assertEqual(result["messages"][-1].content, "Деплой завершён, сервис отвечает 200 OK.")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self._assert_feedback_hidden(result["messages"])

    async def test_chat_only_mode_still_runs_critic(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[AIMessage(content="Готово без инструментов.")],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: FINISHED\n"
                        "REASON: Answer is sufficient.\n"
                        "NEXT_STEP: NONE\n"
                        "CONTROL: FINISH_TURN"
                    )
                )
            ],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Ответь без tools"), config={"recursion_limit": 24})

        self.assertEqual(result["messages"][-1].content, "Готово без инструментов.")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)

    async def test_hybrid_mode_chat_only_skips_critic(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[AIMessage(content="Готово без инструментов.")],
            critic_responses=[],
            tools=[],
            model_supports_tools=False,
            critic_mode="hybrid",
        )

        result = await app.ainvoke(self._initial_state("Ответь без tools"), config={"recursion_limit": 24})

        self.assertEqual(result["messages"][-1].content, "Готово без инструментов.")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 0)

    async def test_hybrid_mode_open_tool_issue_and_user_question_finishes_turn_without_retry(self):
        failing_tool = FakeTool("demo_tool", "ERROR[NOT_FOUND]: File not found")
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "demo_tool", "args": {"path": "missing.py"}, "id": "tc-hybrid-1"}],
                ),
                AIMessage(content="Файл не найден. Пожалуйста, укажите правильный путь к файлу."),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: Need one more retry.\n"
                        "NEXT_STEP: retry immediately\n"
                        "CONTROL: RETRY_AGENT"
                    )
                )
            ],
            tools=[failing_tool],
            critic_mode="hybrid",
        )

        result = await app.ainvoke(self._initial_state("Обработай файл"), config={"recursion_limit": 36})

        self.assertIn("укажите правильный путь", str(result["messages"][-1].content).lower())
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(len(critic_llm.invocations), 1)

    async def test_hybrid_mode_blocks_tool_retries_after_validation_issue_and_handoffs(self):
        failing_tool = FakeTool(
            "edit_file",
            "ERROR[VALIDATION]: Missing required field: new_string. Accepted aliases: new_text, replace_text, replacement.",
        )
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "edit_file", "args": {"path": "demo.py", "old_string": "a"}, "id": "tc-val-1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "edit_file", "args": {"path": "demo.py", "old_string": "a"}, "id": "tc-val-2"}],
                ),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: Needs one more retry.\n"
                        "NEXT_STEP: retry\n"
                        "CONTROL: RETRY_AGENT"
                    )
                )
            ],
            tools=[failing_tool],
            critic_mode="hybrid",
        )

        result = await app.ainvoke(self._initial_state("Обнови файл"), config={"recursion_limit": 36})

        self.assertIn("укажите путь", str(result["messages"][-1].content).lower())
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(len(critic_llm.invocations), 1)
        self.assertEqual(len(failing_tool.calls), 1)

    async def test_hybrid_mode_suppresses_repeated_retry_on_same_response_fingerprint(self):
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        repeated_answer = "Не удалось завершить задачу из-за ошибки инструмента."
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-hybrid-2"}],
                ),
                AIMessage(content=repeated_answer),
                AIMessage(content=repeated_answer),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: Try again.\n"
                        "NEXT_STEP: retry\n"
                        "CONTROL: RETRY_AGENT"
                    )
                ),
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: Try again once more.\n"
                        "NEXT_STEP: retry\n"
                        "CONTROL: RETRY_AGENT"
                    )
                ),
            ],
            tools=[failing_tool],
            critic_mode="hybrid",
            critic_max_retries_per_turn=1,
        )

        result = await app.ainvoke(self._initial_state("Обработай файл"), config={"recursion_limit": 36})

        self.assertEqual(result["messages"][-1].content, repeated_answer)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(len(critic_llm.invocations), 2)

    async def test_planning_tool_budget_blocks_third_call_even_with_different_args(self):
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[FakeTool("sequentialthinking", "ok")],
            llm_with_tools=FakeLLM([]),
        )
        state = self._initial_state("Сделай задачу")
        recent_calls = [
            {"name": "sequentialthinking", "args": {"thought": "first"}},
            {"name": "sequentialthinking", "args": {"thought": "second"}},
        ]
        tool_call = {"name": "sequentialthinking", "args": {"thought": "third"}, "id": "tc-plan-3"}

        tool_msg, had_error, issue = await nodes._process_tool_call(
            tool_call,
            recent_calls,
            state,
            {},
            1,
        )

        self.assertTrue(had_error)
        self.assertIn("ERROR[LOOP_DETECTED]", str(tool_msg.content))
        self.assertIn("Planning tool budget reached", str(tool_msg.content))
        self.assertEqual(issue["error_type"], "LOOP_DETECTED")
        self.assertEqual(issue["tool_names"], ["sequentialthinking"])

    async def test_process_tool_call_persists_tool_args_in_tool_message_metadata(self):
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[FakeTool("read_file", "contents")],
            llm_with_tools=FakeLLM([]),
        )
        state = self._initial_state("Покажи файл")
        tool_call = {
            "name": "read_file",
            "args": {"path": "demo.txt", "line_start": 1, "line_end": 5},
            "id": "tc-meta-1",
        }

        tool_msg, had_error, issue = await nodes._process_tool_call(
            tool_call,
            [],
            state,
            {},
            1,
        )

        self.assertFalse(had_error)
        self.assertIsNone(issue)
        self.assertEqual(tool_msg.tool_call_id, "tc-meta-1")
        self.assertEqual(
            (tool_msg.additional_kwargs or {}).get("tool_args"),
            {"path": "demo.txt", "line_start": 1, "line_end": 5},
        )

    async def test_malformed_critic_verdict_finishes_turn_conservatively(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[AIMessage(content="Похоже, всё готово.")],
            critic_responses=[AIMessage(content="Looks good to me.")],
            tools=[],
            model_supports_tools=False,
            agent_llm_cls=ProviderSafeFakeLLM,
        )

        result = await app.ainvoke(self._initial_state("Сделай и проверь"), config={"recursion_limit": 24})

        self.assertEqual(result["messages"][-1].content, "Похоже, всё готово.")
        self.assertEqual(result["critic_status"], "FINISHED")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)
        self._assert_feedback_hidden(result["messages"])

    async def test_user_handoff_answer_finishes_turn_via_control_not_phrase_guard(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content=(
                        "Не удалось скачать архив напрямую.\n\n"
                        "Альтернативные варианты:\n"
                        "1 Через winget\n"
                        "2 Через chocolatey\n"
                        "3 Ручная загрузка\n\n"
                        "Как поступить? Укажите предпочтительный способ."
                    )
                )
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: The assistant is waiting for the next decision.\n"
                        "NEXT_STEP: wait for user input\n"
                        "CONTROL: FINISH_TURN"
                    )
                )
            ],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Обнови ffmpeg.exe"), config={"recursion_limit": 24})

        self.assertEqual(result["critic_status"], "INCOMPLETE")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)
        self.assertIn("Как поступить?", str(result["messages"][-1].content))
        self._assert_feedback_hidden(result["messages"])

    async def test_confirmation_request_finishes_turn_via_control_not_phrase_guard(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content=(
                        "Чтобы заменить старую версию, нужно:\n"
                        "1 Удалить C:\\Windows\\ffmpeg.exe.\n"
                        "2 Скопировать ffmpeg-new.exe в C:\\Windows\\.\n"
                        "Внимание: Удаление файлов из C:\\Windows\\ требует прав администратора. "
                        "Подтвердите, если разрешаете выполнить эти действия."
                    )
                )
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: The assistant still needs permission.\n"
                        "NEXT_STEP: wait for approval\n"
                        "CONTROL: FINISH_TURN"
                    )
                )
            ],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Обнови ffmpeg.exe"), config={"recursion_limit": 24})

        self.assertEqual(result["critic_status"], "INCOMPLETE")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)
        self.assertIn("Подтвердите, если разрешаете", str(result["messages"][-1].content))
        self._assert_feedback_hidden(result["messages"])

    async def test_help_offer_finishes_turn_via_control_not_phrase_guard(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content=(
                        "Проверка: Если выполните шаг 1, команда ffmpeg -version покажет 8.1. "
                        "Иначе используйте вариант 2. Уточните, если нужна помощь."
                    )
                )
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: The assistant described follow-up options.\n"
                        "NEXT_STEP: wait for user clarification\n"
                        "CONTROL: FINISH_TURN"
                    )
                )
            ],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Обнови ffmpeg.exe"), config={"recursion_limit": 24})

        self.assertEqual(result["critic_status"], "INCOMPLETE")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)
        self.assertIn("Уточните, если нужна помощь.", str(result["messages"][-1].content))
        self._assert_feedback_hidden(result["messages"])

    async def test_user_choice_prompt_overrides_critic_retry_to_finish_turn(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content=(
                        "Скрипт не смог извлечь данные.\n"
                        "Предлагаю варианты: 1) обновить парсер, 2) Selenium, 3) другой источник.\n"
                        "Какой вариант выбираете? Укажите номер."
                    )
                )
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: Still needs another execution step.\n"
                        "NEXT_STEP: run another attempt immediately\n"
                        "CONTROL: RETRY_AGENT"
                    )
                )
            ],
            tools=[],
            model_supports_tools=False,
        )

        result = await app.ainvoke(self._initial_state("Собери прогноз погоды"), config={"recursion_limit": 24})

        self.assertEqual(result["critic_status"], "INCOMPLETE")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 1)
        self.assertIn("Какой вариант выбираете?", str(result["messages"][-1].content))
        self._assert_feedback_hidden(result["messages"])

    async def test_multiple_critic_rounds_fit_into_recursion_budget(self):
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(content="Черновой результат."),
                AIMessage(content="Ещё одна проверка выполнена."),
                AIMessage(content="Итог подтверждён."),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: Need explicit verification.\n"
                        "NEXT_STEP: verify output\n"
                        "CONTROL: RETRY_AGENT"
                    )
                ),
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: Verification still weak.\n"
                        "NEXT_STEP: confirm final state\n"
                        "CONTROL: RETRY_AGENT"
                    )
                ),
                AIMessage(
                    content=(
                        "STATUS: FINISHED\n"
                        "REASON: Final confirmation present.\n"
                        "NEXT_STEP: NONE\n"
                        "CONTROL: FINISH_TURN"
                    )
                ),
            ],
            tools=[],
            model_supports_tools=False,
            max_loops=5,
            critic_max_retries_per_turn=2,
            agent_llm_cls=ProviderSafeFakeLLM,
        )

        result = await app.ainvoke(self._initial_state("Сделай несколько проверок"), config={"recursion_limit": 30})

        self.assertEqual(result["messages"][-1].content, "Итог подтверждён.")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(len(critic_llm.invocations), 3)

    async def test_fatal_llm_error_stops_graph_without_reentering_critic(self):
        fatal_error = Exception(
            "Error code: 402 - {'error': {'code': '402', 'message': 'Insufficient account balance', 'type': 'insufficient_balance'}}"
        )
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[fatal_error, fatal_error, fatal_error],
            critic_responses=[],
            tools=[],
            model_supports_tools=False,
            max_retries=3,
            retry_delay=0,
            agent_llm_cls=ProviderSafeFakeLLM,
        )

        with self.assertRaises(Exception) as ctx:
            await app.ainvoke(self._initial_state("Собери информацию"), config={"recursion_limit": 24})

        self.assertIn("402", str(ctx.exception))
        self.assertEqual(len(agent_llm.invocations), 1)
        self.assertEqual(len(critic_llm.invocations), 0)

    async def test_invalid_tool_calls_become_protocol_error_not_unknown_tool(self):
        config = self._make_config(model_supports_tools=True)
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[FakeTool("demo_tool", "ok")],
            llm_with_tools=ProviderSafeFakeLLM(
                [
                    AIMessage(
                        content="",
                        invalid_tool_calls=[{"name": "demo_tool", "args": "{", "id": "broken-1", "error": "bad json"}],
                    )
                ]
            ),
        )

        result = await nodes.agent_node(self._initial_state("Сделай вызов инструмента"))
        response = result["messages"][0]

        self.assertIsInstance(response, AIMessage)
        self.assertFalse(response.tool_calls)
        self.assertNotIn("unknown_tool", str(response.content))
        self.assertIn("INTERNAL TOOL PROTOCOL ERROR", str(response.content))
        self.assertEqual(result["critic_source"], "agent")

    async def test_tool_errors_no_longer_store_fake_critic_feedback(self):
        config = self._make_config(model_supports_tools=True)
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=FakeLLM([]),
        )

        result = await nodes.tools_node(
            {
                **self._initial_state("Почини ошибку"),
                "messages": [
                    AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-9"}])
                ],
            }
        )

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].name, "demo_tool")
        self.assertEqual(result["critic_feedback"], "")
        self.assertEqual(result["turn_outcome"], "run_tools")

    async def test_agent_context_includes_unresolved_tool_failure_system_message(self):
        config = self._make_config(model_supports_tools=True)
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        agent_llm = ProviderSafeFakeLLM([AIMessage(content="Не удалось завершить задачу из-за ошибки инструмента.")])
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=agent_llm,
        )

        await nodes.agent_node(
            {
                **self._initial_state("Почини ошибку"),
                "messages": [
                    HumanMessage(content="Почини ошибку"),
                    AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-10"}]),
                    ToolMessage(content="ERROR[EXECUTION]: boom", tool_call_id="tc-10", name="demo_tool"),
                ],
                "open_tool_issue": {
                    "turn_id": 1,
                    "kind": "tool_error",
                    "summary": "boom",
                    "tool_names": ["demo_tool"],
                    "source": "tools",
                },
            }
        )

        unresolved_messages = [
            str(message.content)
            for message in agent_llm.invocations[0]
            if isinstance(message, SystemMessage) and "UNRESOLVED TOOL FAILURE" in str(message.content)
        ]
        self.assertTrue(unresolved_messages)

    async def test_agent_context_inserts_bridge_when_user_follows_tool_message(self):
        config = self._make_config(model_supports_tools=True)
        strict_llm = StrictToolOrderFakeLLM([AIMessage(content="Продолжаю выполнение после tool-результата.")])
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[FakeTool("demo_tool", "ok")],
            llm_with_tools=strict_llm,
        )

        result = await nodes.agent_node(
            {
                **self._initial_state("Почини проблему"),
                "messages": [
                    HumanMessage(content="Почини проблему"),
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-order-1"}],
                    ),
                    ToolMessage(content="ERROR[NOT_FOUND]: boom", tool_call_id="tc-order-1", name="demo_tool"),
                    HumanMessage(content="продолжай"),
                ],
                "open_tool_issue": {
                    "turn_id": 2,
                    "kind": "tool_error",
                    "summary": "boom",
                    "tool_names": ["demo_tool"],
                    "source": "tools",
                },
                "critic_source": "tools",
            }
        )

        self.assertEqual(result["messages"][-1].content, "Продолжаю выполнение после tool-результата.")
        non_system = [m for m in strict_llm.invocations[0] if not isinstance(m, SystemMessage)]
        for prev, curr in zip(non_system, non_system[1:]):
            self.assertFalse(isinstance(prev, ToolMessage) and isinstance(curr, HumanMessage))

    async def test_unresolved_tool_issue_requires_retry_until_critic_finishes(self):
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-11"}],
                ),
                AIMessage(content="Файл успешно обработан и всё готово."),
                AIMessage(content="Не удалось завершить задачу: инструмент завершился с ошибкой."),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: INCOMPLETE\n"
                        "REASON: The unresolved tool blocker was not reported honestly.\n"
                        "NEXT_STEP: report the blocker to the user\n"
                        "CONTROL: RETRY_AGENT"
                    )
                ),
                AIMessage(
                    content=(
                        "STATUS: FINISHED\n"
                        "REASON: Honest blocker report delivered.\n"
                        "NEXT_STEP: NONE\n"
                        "CONTROL: FINISH_TURN"
                    )
                ),
            ],
            tools=[failing_tool],
            agent_llm_cls=ProviderSafeFakeLLM,
        )

        result = await app.ainvoke(self._initial_state("Обработай файл"), config={"recursion_limit": 36})

        self.assertEqual(result["messages"][-1].content, "Не удалось завершить задачу: инструмент завершился с ошибкой.")
        self.assertEqual(result["critic_status"], "FINISHED")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(len(critic_llm.invocations), 2)

        retry_last_visible = self._last_model_visible(agent_llm.invocations[2])
        self.assertIsInstance(retry_last_visible, HumanMessage)
        self.assertEqual(retry_last_visible.additional_kwargs["agent_internal"]["kind"], "retry_instruction")

        final_hint_messages = [
            str(message.content)
            for message in agent_llm.invocations[2]
            if isinstance(message, SystemMessage) and "UNRESOLVED TOOL FAILURE" in str(message.content)
        ]
        self.assertTrue(final_hint_messages)

    async def test_critic_cannot_finish_successfully_while_open_tool_issue_is_unacknowledged(self):
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        app, agent_llm, critic_llm, _ = self._build_app(
            agent_responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-guard"}],
                ),
                AIMessage(content="Всё успешно завершено."),
                AIMessage(content="Не удалось завершить задачу: инструмент завершился с ошибкой."),
            ],
            critic_responses=[
                AIMessage(
                    content=(
                        "STATUS: FINISHED\n"
                        "REASON: The task is complete.\n"
                        "NEXT_STEP: NONE\n"
                        "CONTROL: FINISH_TURN"
                    )
                ),
                AIMessage(
                    content=(
                        "STATUS: FINISHED\n"
                        "REASON: Honest blocker report delivered.\n"
                        "NEXT_STEP: NONE\n"
                        "CONTROL: FINISH_TURN"
                    )
                ),
            ],
            tools=[failing_tool],
            agent_llm_cls=ProviderSafeFakeLLM,
        )

        result = await app.ainvoke(self._initial_state("Обработай файл"), config={"recursion_limit": 36})

        self.assertEqual(result["messages"][-1].content, "Не удалось завершить задачу: инструмент завершился с ошибкой.")
        self.assertEqual(result["critic_status"], "FINISHED")
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(len(agent_llm.invocations), 3)
        self.assertEqual(len(critic_llm.invocations), 2)

    async def test_tool_failure_sets_open_tool_issue_for_current_turn(self):
        config = self._make_config(model_supports_tools=True)
        failing_tool = FakeTool("demo_tool", "ERROR[EXECUTION]: boom")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=FakeLLM([]),
        )

        result = await nodes.tools_node(
            {
                **self._initial_state("Почини ошибку"),
                "messages": [
                    AIMessage(content="", tool_calls=[{"name": "demo_tool", "args": {"action": "x"}, "id": "tc-12"}])
                ],
            }
        )

        self.assertEqual(result["turn_id"], 1)
        self.assertEqual(result["open_tool_issue"]["kind"], "tool_error")
        self.assertEqual(result["open_tool_issue"]["tool_names"], ["demo_tool"])

    def test_parse_critic_response_supports_control_line(self):
        config = self._make_config(model_supports_tools=False)
        nodes = AgentNodes(config=config, llm=FakeLLM([]), tools=[], llm_with_tools=FakeLLM([]))

        parsed = nodes._parse_critic_response(
            "STATUS: INCOMPLETE\nREASON: Missing step.\nNEXT_STEP: verify output\nCONTROL: RETRY_AGENT"
        )

        self.assertEqual(parsed, ("INCOMPLETE", "Missing step.", "verify output", "RETRY_AGENT"))

    def test_parse_critic_response_defaults_control_from_status_when_missing(self):
        config = self._make_config(model_supports_tools=False)
        nodes = AgentNodes(config=config, llm=FakeLLM([]), tools=[], llm_with_tools=FakeLLM([]))

        parsed = nodes._parse_critic_response("STATUS: FINISHED\nREASON: Done.\nNEXT_STEP: NONE")

        self.assertEqual(parsed, ("FINISHED", "Done.", "NONE", "FINISH_TURN"))


if __name__ == "__main__":
    unittest.main()
