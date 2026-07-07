from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import ValidationError

from core.message_utils import compact_text
from core.message_utils import stringify_content
from core.planning import (
    RuntimePlanDraft,
    active_step,
    build_runtime_plan,
    coerce_runtime_plan,
    extract_json_object,
    first_pending_step,
    normalize_plan_action,
    normalize_plan_feedback,
    plan_is_complete,
    set_step_status,
    update_plan_status,
)
from core.state import AgentState
from core.turn_outcomes import TURN_OUTCOME_FINISH_TURN, TURN_OUTCOME_RECOVER_AGENT


def _runtime_plan_markdown(plan: dict[str, Any] | None) -> str:
    if not isinstance(plan, dict):
        return ""
    summary = str(plan.get("summary") or "").strip()
    steps = plan.get("steps") or []
    lines: list[str] = []
    if summary:
        lines.append(summary)
    if isinstance(steps, list) and steps:
        lines.extend(["", "**Подход**"])
        for step in steps:
            if not isinstance(step, dict):
                continue
            title = str(step.get("title") or "").strip()
            description = str(step.get("description") or "").strip()
            if title and description and description != title:
                lines.append(f"- **{title}:** {description}")
            elif title or description:
                lines.append(f"- {title or description}")
    return "\n".join(lines).strip()


_PLAN_STEP_COMPLETE_RE = re.compile(
    r"<!--\s*plan-step-complete\s*:\s*(?P<step_id>[^>\s]+)\s*-->",
    re.IGNORECASE,
)


class PlanningMixin:
    """Plan Mode nodes.

    Planning is a first-class graph branch. The normal agent node may still do
    read-only research in ``turn_mode=plan``; this mixin turns that research
    history into durable structured state and manages human review/resume.
    """

    def _plan_review_options(self) -> list[dict[str, str]]:
        return [
            {"key": "implement", "label": "Реализовать", "submit_text": "implement"},
            {"key": "revise", "label": "Внести изменения в план", "submit_text": "revise"},
            {"key": "rebuild", "label": "Перестроить план", "submit_text": "rebuild"},
            {"key": "cancel", "label": "Отменить", "submit_text": "cancel"},
        ]

    def _plan_review_payload(self, state: AgentState, *, reason: str = "") -> dict[str, Any]:
        plan = coerce_runtime_plan(state.get("current_plan"))
        return {
            "kind": "user_choice",
            "choice_type": "plan_review",
            "question": "Что сделать с этим планом?",
            "options": self._plan_review_options(),
            "recommended": "implement",
            "current_plan": plan,
            "plan_markdown": _runtime_plan_markdown(plan),
            "reason": reason,
        }

    def _build_plan_prompt(self, state: AgentState, *, rebuild: bool) -> str:
        current_plan = coerce_runtime_plan(state.get("current_plan"))
        feedback = str(state.get("plan_feedback") or "").strip()
        schema = RuntimePlanDraft.model_json_schema()
        lines = [
            "STRUCTURED PLAN BUILDER:",
            "Return one concise implementation plan as structured data matching the provided schema.",
            "Do not execute the plan. Do not request approval. Do not include markdown prose outside the object.",
            "Use 3 to 5 practical steps unless the task is truly smaller.",
            "All steps must start with status='pending'.",
            "Each step needs id, title, description, and status.",
            "Keep risks, assumptions, verification, estimated_tools, and estimated_files short and concrete.",
            "List files only when repository context supports them; otherwise leave estimated_files empty or broad.",
            "The plan must be decision-complete for implementation.",
            f"JSON schema: {json.dumps(schema, ensure_ascii=False)}",
        ]
        if feedback:
            lines.append(f"User feedback for this revision: {feedback}")
        if current_plan and not rebuild:
            lines.append(
                "Previous plan to revise, preserving intent unless feedback says otherwise: "
                + json.dumps(current_plan, ensure_ascii=False)
            )
        if rebuild:
            lines.append("Rebuild from scratch. Do not preserve the previous plan id, version, or step structure.")
        return "\n".join(lines)

    async def _invoke_plan_structured_output(self, context: list[BaseMessage], state: AgentState) -> RuntimePlanDraft:
        structured = None
        binder = getattr(self.llm, "with_structured_output", None)
        if callable(binder):
            try:
                structured = binder(RuntimePlanDraft, method="function_calling")
            except TypeError:
                structured = binder(RuntimePlanDraft)
            except Exception as exc:
                self._log_run_event(
                    state,
                    "plan_structured_output_bind_failed",
                    run_id=state.get("run_id", ""),
                    error=compact_text(str(exc), 300),
                )
                structured = None

        if structured is not None:
            result = await structured.ainvoke(self._normalize_system_prefix_for_provider(context))
            if isinstance(result, RuntimePlanDraft):
                return result
            return RuntimePlanDraft.model_validate(result)

        # Strict JSON fallback for models/adapters without structured-output
        # wrappers. This is still schema-validated and never accepts markdown.
        response = await self._invoke_llm_with_retry(
            self.llm,
            [
                *context,
                SystemMessage(
                    content=(
                        "The selected model adapter has no structured-output wrapper. "
                        "Return ONLY one JSON object matching the plan schema."
                    )
                ),
            ],
            state=state,
            node_name="plan_build_json_fallback",
        )
        payload = extract_json_object(str(getattr(response, "content", "") or ""))
        return RuntimePlanDraft.model_validate(payload)

    async def plan_build_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "plan_build",
            has_existing_plan=bool(state.get("current_plan")),
            plan_status=str(state.get("plan_status") or ""),
        )
        messages = state.get("messages", [])
        current_task = self._resolve_current_task(state, messages)
        current_turn_id = self._current_turn_id(state, messages)
        rebuild = str(state.get("plan_status") or "").strip().lower() == "rebuild_requested"
        summary = str(state.get("summary") or "")
        try:
            context = self.context_builder.build(
                messages,
                state,
                summary=summary,
                current_task=current_task,
                tools_available=False,
                active_tool_names=[],
                open_tool_issue=None,
                recovery_state=self._get_recovery_state(state, current_turn_id=current_turn_id),
            )
            context.append(SystemMessage(content=self._build_plan_prompt(state, rebuild=rebuild)))
            draft = await self._invoke_plan_structured_output(context, state)
            plan = build_runtime_plan(
                draft,
                previous_plan=state.get("current_plan") if not rebuild else None,
                rebuild=rebuild,
            )
            self._log_run_event(
                state,
                "plan_built",
                run_id=state.get("run_id", ""),
                plan_id=plan.get("id"),
                version=plan.get("version"),
                step_count=len(plan.get("steps") or []),
            )
            self._log_node_end(
                state,
                "plan_build",
                node_timer,
                outcome="built",
                step_count=len(plan.get("steps") or []),
            )
            return {
                "current_task": current_task,
                "turn_id": current_turn_id,
                "turn_mode": "plan",
                "current_plan": plan,
                "plan_status": "pending_approval",
                "plan_revision": int(plan.get("version") or 1),
                "active_plan_step_id": "",
                "plan_feedback": "",
                "turn_outcome": TURN_OUTCOME_FINISH_TURN,
                "pending_approval": None,
                "open_tool_issue": None,
                "has_protocol_error": False,
                "last_tool_error": "",
                "last_tool_result": "",
            }
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            summary_text = (
                "INTERNAL PLAN PROTOCOL ERROR: the model did not return a valid structured plan. "
                "Retry with a JSON object matching the runtime plan schema."
            )
            issue = self._build_protocol_issue(
                current_turn_id=current_turn_id,
                summary=summary_text,
                reason="structured_plan_validation_error",
                source="plan_build",
                details={"error": compact_text(str(exc), 500)},
            )
            self._log_node_error(state, "plan_build", node_timer, exc, handled=True)
            return {
                "turn_id": current_turn_id,
                "turn_outcome": TURN_OUTCOME_RECOVER_AGENT,
                "open_tool_issue": issue,
                "has_protocol_error": True,
                "last_tool_error": summary_text,
                "last_tool_result": "",
            }

    async def plan_review_node(self, state: AgentState):
        node_timer = self._log_node_start(state, "plan_review", plan_status=state.get("plan_status", ""))
        plan = update_plan_status(state.get("current_plan"), status="pending_approval", active_step_id="")
        decision = interrupt(self._plan_review_payload({**state, "current_plan": plan}))
        action = normalize_plan_action(decision)
        self._log_run_event(
            state,
            "plan_review_resolved",
            run_id=state.get("run_id", ""),
            action=action,
        )
        if action == "implement":
            approved_plan = update_plan_status(plan, status="approved", active_step_id="")
            self._log_node_end(state, "plan_review", node_timer, outcome="approved")
            return {
                "current_plan": approved_plan,
                "plan_status": "approved",
                "turn_mode": "chat",
                "active_plan_step_id": "",
            }
        if action == "revise":
            self._log_node_end(state, "plan_review", node_timer, outcome="needs_changes")
            return {"current_plan": plan, "plan_status": "needs_changes", "turn_mode": "plan"}
        if action == "rebuild":
            self._log_node_end(state, "plan_review", node_timer, outcome="rebuild_requested")
            return {
                "current_plan": update_plan_status(plan, status="rebuild_requested", active_step_id=""),
                "plan_status": "rebuild_requested",
                "plan_feedback": "",
                "turn_mode": "plan",
            }
        self._log_node_end(state, "plan_review", node_timer, outcome="rejected")
        rejected_plan = update_plan_status(plan, status="rejected", active_step_id="")
        return {
            "messages": [AIMessage(content="План отменён. Я не выполнял изменения.")],
            "current_plan": rejected_plan,
            "plan_status": "rejected",
            "turn_mode": "chat",
            "active_plan_step_id": "",
            "turn_outcome": TURN_OUTCOME_FINISH_TURN,
        }

    async def plan_revision_input_node(self, state: AgentState):
        node_timer = self._log_node_start(state, "plan_revision_input")
        payload = {
            "kind": "user_choice",
            "choice_type": "plan_revision",
            "question": "Какие изменения внести в план?",
            "options": [],
            "recommended": "",
            "allow_custom_text": True,
            "custom_label": "Ввести замечания",
            "current_plan": coerce_runtime_plan(state.get("current_plan")),
        }
        feedback = interrupt(payload)
        text = normalize_plan_feedback(feedback)
        while not text:
            feedback = interrupt(
                {
                    **payload,
                    "question": "Опишите, какие изменения внести в план.",
                    "reason": "Пустые замечания не запускают пересборку плана.",
                }
            )
            text = normalize_plan_feedback(feedback)
        self._log_node_end(state, "plan_revision_input", node_timer, has_feedback=True)
        return {
            "plan_feedback": text,
            "plan_status": "needs_changes",
            "turn_mode": "plan",
        }

    async def plan_select_step_node(self, state: AgentState):
        node_timer = self._log_node_start(state, "plan_select_step", plan_status=state.get("plan_status", ""))
        plan = coerce_runtime_plan(state.get("current_plan"))
        if plan is None:
            issue = self._build_protocol_issue(
                current_turn_id=self._current_turn_id(state, state.get("messages", [])),
                summary="INTERNAL PLAN PROTOCOL ERROR: approved execution requested without a valid current_plan.",
                reason="missing_runtime_plan",
                source="plan_select_step",
            )
            self._log_node_end(state, "plan_select_step", node_timer, outcome="missing_plan")
            return {
                "turn_outcome": TURN_OUTCOME_RECOVER_AGENT,
                "open_tool_issue": issue,
                "has_protocol_error": True,
                "last_tool_error": str(issue.get("summary") or ""),
            }
        if plan_is_complete(plan):
            completed = update_plan_status(plan, status="completed", active_step_id="")
            self._log_node_end(state, "plan_select_step", node_timer, outcome="completed")
            return {
                "messages": [AIMessage(content="План выполнен.")],
                "current_plan": completed,
                "plan_status": "completed",
                "active_plan_step_id": "",
                "turn_outcome": TURN_OUTCOME_FINISH_TURN,
            }
        step = first_pending_step(plan)
        if not step:
            blocked = update_plan_status(plan, status="replan_pending", active_step_id="")
            issue = self._build_protocol_issue(
                current_turn_id=self._current_turn_id(state, state.get("messages", [])),
                summary=(
                    "INTERNAL PLAN EXECUTION ERROR: the plan has unfinished steps, "
                    "but no pending step can be selected. Keep the plan visible and request recovery."
                ),
                reason="no_pending_unfinished_plan_step",
                source="plan_select_step",
            )
            self._log_node_end(state, "plan_select_step", node_timer, outcome="blocked_no_pending")
            return {
                "current_plan": blocked or plan,
                "plan_status": "replan_pending",
                "active_plan_step_id": "",
                "turn_outcome": TURN_OUTCOME_RECOVER_AGENT,
                "open_tool_issue": issue,
                "has_protocol_error": True,
                "last_tool_error": str(issue.get("summary") or ""),
            }
        selected = set_step_status(plan, str(step.get("id")), "in_progress")
        self._log_node_end(
            state,
            "plan_select_step",
            node_timer,
            outcome="selected",
            active_step_id=str(step.get("id")),
        )
        return {
            "current_plan": selected,
            "plan_status": "executing",
            "active_plan_step_id": str(step.get("id")),
            "turn_mode": "chat",
            "turn_outcome": TURN_OUTCOME_FINISH_TURN,
        }

    async def plan_complete_step_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "plan_complete_step",
            active_step_id=state.get("active_plan_step_id", ""),
        )
        plan = coerce_runtime_plan(state.get("current_plan"))
        step_id = str(state.get("active_plan_step_id") or (plan or {}).get("active_step_id") or "").strip()
        completed = set_step_status(plan, step_id, "completed") if step_id else plan
        if completed is not None:
            completed["status"] = "completed" if plan_is_complete(completed) else "executing"
        self._log_node_end(state, "plan_complete_step", node_timer, completed_step_id=step_id)
        return {
            "current_plan": completed,
            "plan_status": "completed" if completed and plan_is_complete(completed) else "executing",
            "active_plan_step_id": "",
            "turn_outcome": TURN_OUTCOME_FINISH_TURN,
        }

    def plan_step_completion_requested(self, state: AgentState) -> bool:
        plan = coerce_runtime_plan(state.get("current_plan"))
        step_id = str(state.get("active_plan_step_id") or (plan or {}).get("active_step_id") or "").strip()
        if not step_id:
            return False
        messages = state.get("messages") or []
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            text = stringify_content(message.content)
            for match in _PLAN_STEP_COMPLETE_RE.finditer(text):
                if match.group("step_id").strip() == step_id:
                    return True
            return False
        return False

    async def plan_block_step_node(self, state: AgentState):
        node_timer = self._log_node_start(state, "plan_block_step")
        plan = coerce_runtime_plan(state.get("current_plan"))
        step_id = str(state.get("active_plan_step_id") or (plan or {}).get("active_step_id") or "").strip()
        blocked = set_step_status(plan, step_id, "blocked") if step_id else plan
        if blocked is not None:
            blocked["status"] = "replan_pending"
            blocked["active_step_id"] = ""
        reason = str(state.get("last_tool_error") or state.get("self_correction_last_reason") or "Текущий шаг требует перепланирования.")
        decision = interrupt(
            {
                "kind": "user_choice",
                "choice_type": "plan_replan",
                "question": "Текущий этап заблокирован. Обновить план?",
                "options": [
                    {"key": "replan", "label": "Обновить план", "submit_text": "rebuild"},
                    {"key": "cancel", "label": "Отменить выполнение", "submit_text": "cancel"},
                ],
                "recommended": "replan",
                "reason": reason,
                "current_plan": blocked,
            }
        )
        action = normalize_plan_action(decision)
        self._log_node_end(state, "plan_block_step", node_timer, action=action)
        if action in {"rebuild", "revise", "implement"}:
            return {
                "current_plan": blocked,
                "plan_status": "rebuild_requested",
                "plan_feedback": reason,
                "active_plan_step_id": "",
                "turn_mode": "plan",
            }
        return {
            "current_plan": update_plan_status(blocked, status="rejected", active_step_id=""),
            "plan_status": "rejected",
            "active_plan_step_id": "",
            "turn_mode": "chat",
            "messages": [AIMessage(content="Выполнение плана остановлено.")],
            "turn_outcome": TURN_OUTCOME_FINISH_TURN,
        }

    def _plan_execution_context_message(self, state: AgentState | None) -> SystemMessage | None:
        if not isinstance(state, dict):
            return None
        plan = coerce_runtime_plan(state.get("current_plan"))
        if plan is None:
            return None
        status = str(state.get("plan_status") or plan.get("status") or "").strip().lower()
        if status not in {"approved", "executing"}:
            return None
        step = active_step(plan, str(state.get("active_plan_step_id") or ""))
        if not step:
            return None
        content = (
            "APPROVED PLAN EXECUTION CONTEXT:\n"
            "A user-approved runtime plan is active. Do not rebuild or re-approve the plan. "
            "Execute only the active step below. Do not do work from later steps, and do not say you are moving to another step.\n"
            "Final visible status must be exactly one short line in Russian, no more than 140 characters. "
            "Do not write reports, bullet lists, section headings, file inventories, verification summaries, "
            "or phrases like 'Что вошло', 'Проверка выполнена', or 'Переход к шагу'.\n"
            "Use this shape only: 'Готово: <what changed in this active step>.'\n"
            "Only when the active step is actually complete, append this hidden marker on its own final line: "
            f"<!--plan-step-complete:{step.get('id')}-->\n"
            "If the step is not complete, do not include the marker; use one short line: 'Не завершено: <what remains or blocked it>.'\n\n"
            f"Plan summary: {plan.get('summary')}\n"
            f"Active step id: {step.get('id')}\n"
            f"Active step title: {step.get('title')}\n"
            f"Active step description: {step.get('description')}\n"
        )
        return SystemMessage(content=content)
