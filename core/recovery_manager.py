from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from core import constants
from core.message_utils import compact_text, stringify_content
from core.self_correction_engine import RepairPlan, build_repair_plan
from core.tool_results import parse_tool_execution_result


def _compact_json_payload(payload: Dict[str, Any], *, limit: int = 180) -> str:
    items: List[str] = []
    for key, value in payload.items():
        if value is None:
            continue
        rendered = (
            json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, (dict, list))
            else str(value)
        )
        items.append(f"{key}={rendered}")
    return compact_text(", ".join(items), limit)


def _actionable_recovery_hint(repair_plan: RepairPlan | None) -> str:
    if repair_plan is None:
        return ""
    parts: List[str] = []
    suggested_tool = str(repair_plan.suggested_tool_name or "").strip()
    if suggested_tool:
        parts.append(f"Suggested next tool: `{suggested_tool}`.")
    if isinstance(repair_plan.patched_args, dict) and repair_plan.patched_args:
        parts.append(f"Prepared arguments: {_compact_json_payload(repair_plan.patched_args)}")
    if repair_plan.notes:
        parts.append(f"Hint: {repair_plan.notes}")
    return ("\n" + "\n".join(parts)) if parts else ""


class RecoveryManager:
    def empty_state(self, *, turn_id: int) -> Dict[str, Any]:
        return {
            "turn_id": int(turn_id or 0),
            "active_issue": None,
            "active_strategy": None,
            "strategy_queue": [],
            "attempts_by_strategy": {},
            "progress_markers": [],
            "last_successful_evidence": "",
            "external_blocker": None,
            "llm_replan_attempted_for": [],
        }

    def get_recovery_state(self, raw: Any, *, current_turn_id: int) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return self.empty_state(turn_id=current_turn_id)
        recovery = self.empty_state(turn_id=current_turn_id)
        recovery.update({k: v for k, v in raw.items() if k in recovery})
        if int(recovery.get("turn_id", 0) or 0) != current_turn_id:
            return self.empty_state(turn_id=current_turn_id)
        recovery["turn_id"] = current_turn_id
        if not isinstance(recovery.get("strategy_queue"), list):
            recovery["strategy_queue"] = []
        if not isinstance(recovery.get("attempts_by_strategy"), dict):
            recovery["attempts_by_strategy"] = {}
        if not isinstance(recovery.get("progress_markers"), list):
            recovery["progress_markers"] = []
        if not isinstance(recovery.get("llm_replan_attempted_for"), list):
            recovery["llm_replan_attempted_for"] = []
        return recovery

    def reset_after_success(
        self,
        recovery_state: Any,
        *,
        current_turn_id: int,
        successful_evidence: str = "",
    ) -> Dict[str, Any]:
        next_state = self.empty_state(turn_id=current_turn_id)
        evidence = str(successful_evidence or "").strip()
        if evidence:
            next_state["last_successful_evidence"] = evidence
            return next_state

        current_state = self.get_recovery_state(recovery_state, current_turn_id=current_turn_id)
        previous_evidence = str(current_state.get("last_successful_evidence") or "").strip()
        if previous_evidence:
            next_state["last_successful_evidence"] = previous_evidence
        return next_state

    def repair_plan_strategy_id(self, repair_plan: RepairPlan) -> str:
        payload = {
            "strategy": repair_plan.strategy,
            "suggested_tool": repair_plan.suggested_tool_name,
            "patched_args": repair_plan.patched_args,
            "progress_fingerprint": repair_plan.progress_fingerprint or repair_plan.fingerprint,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def build_recovery_strategy(
        self,
        *,
        repair_plan: RepairPlan,
        open_tool_issue: Dict[str, Any] | None,
        current_task: str,
        strategy_id: str,
    ) -> Dict[str, Any]:
        return {
            "id": strategy_id,
            "strategy": repair_plan.strategy,
            "strategy_kind": self._normalize_strategy_kind(repair_plan.strategy),
            "reason": repair_plan.reason,
            "tool_name": repair_plan.tool_name,
            "suggested_tool_name": repair_plan.suggested_tool_name,
            "patched_args": deepcopy(repair_plan.patched_args),
            "notes": repair_plan.notes,
            "llm_guidance": repair_plan.llm_guidance,
            "current_task": current_task,
            "issue_summary": str((open_tool_issue or {}).get("summary") or "").strip(),
            "issue_details": deepcopy((open_tool_issue or {}).get("details") or {}),
            "progress_fingerprint": repair_plan.progress_fingerprint or repair_plan.fingerprint,
        }

    def build_recovery_system_message(self, recovery_state: Dict[str, Any] | None) -> SystemMessage | None:
        if not isinstance(recovery_state, dict):
            return None
        strategy = recovery_state.get("active_strategy")
        issue = recovery_state.get("active_issue")
        if not isinstance(strategy, dict) or not isinstance(issue, dict):
            return None

        guidance = str(strategy.get("llm_guidance") or "").strip() or (
            "Continue the same task using a different valid approach."
        )
        notes = str(strategy.get("notes") or "").strip()
        suggested_tool = str(strategy.get("suggested_tool_name") or "").strip()
        patched_args = strategy.get("patched_args") if isinstance(strategy.get("patched_args"), dict) else {}
        actionable_parts: List[str] = []
        if suggested_tool:
            actionable_parts.append(f"Preferred next tool: {suggested_tool}")
        if patched_args:
            actionable_parts.append(f"Prepared arguments: {_compact_json_payload(patched_args)}")
        if notes:
            actionable_parts.append(f"Notes: {notes}")
        actionable_text = ("\n" + "\n".join(actionable_parts)) if actionable_parts else ""
        return SystemMessage(
            content=(
                "RECOVERY MODE: continue the current request until you either complete it or hit a real external blocker.\n"
                f"Active issue: {str(issue.get('summary') or '').strip()}\n"
                f"Recovery strategy: {str(strategy.get('strategy_kind') or strategy.get('strategy') or '').strip()}\n"
                f"Guidance: {guidance}"
                f"{actionable_text}\n"
                "Do not ask the user for hints while repository state, tools, or verification steps can still move the task forward.\n"
                "Do not repeat the exact same failing call unchanged."
            )
        )

    def build_tool_issue_handoff_text(
        self,
        open_tool_issue: Dict[str, Any] | None,
        *,
        repair_plan: RepairPlan | None = None,
    ) -> str:
        if not isinstance(open_tool_issue, dict):
            return constants.TOOL_ISSUE_NOT_FOUND_TEXT

        issue_kind = str(open_tool_issue.get("kind") or "").strip().lower()
        details = dict(open_tool_issue.get("details") or {})
        summary = compact_text(str(open_tool_issue.get("summary", "")).strip(), 220)
        tool_names = [
            str(name).strip()
            for name in (open_tool_issue.get("tool_names") or [])
            if str(name).strip()
        ]
        tool_hint = f" (`{tool_names[0]}`)" if tool_names else ""
        summary_line = f"\nDetails: {summary}" if summary else ""
        missing_fields = [
            str(field).strip()
            for field in (details.get("missing_required_fields") or [])
            if str(field).strip()
        ]

        if issue_kind == "approval_denied":
            return constants.TOOL_ISSUE_APPROVAL_DENIED_TEXT

        if details.get("safety_violation"):
            return constants.TOOL_ISSUE_WORKSPACE_BOUNDARY_TEMPLATE.format(
                tool_hint=tool_hint,
                summary_line=summary_line,
            )

        if missing_fields:
            fields_label = ", ".join(missing_fields)
            return constants.TOOL_ISSUE_MISSING_FIELDS_TEMPLATE.format(
                tool_hint=tool_hint,
                summary_line=summary_line,
                fields_label=fields_label,
            )

        return constants.TOOL_ISSUE_STAGNATION_TEMPLATE.format(
            tool_hint=tool_hint,
            summary_line=summary_line,
        )

    @staticmethod
    def build_loop_budget_handoff_text(current_task: str, tool_names: List[str]) -> str:
        tool_hint = f" (last tool: `{tool_names[0]}`)" if tool_names else ""
        task_hint = compact_text(current_task.strip(), 180) if current_task else "the current task"
        return constants.LOOP_BUDGET_HANDOFF_TEMPLATE.format(task_hint=task_hint, tool_hint=tool_hint)

    @staticmethod
    def build_successful_tool_stagnation_handoff_text(
        current_task: str,
        *,
        tool_name: str,
        repeat_count: int,
    ) -> str:
        tool_hint = f" (repeated tool: `{tool_name}`)" if tool_name else ""
        task_hint = compact_text(current_task.strip(), 180) if current_task else "the current task"
        return constants.SUCCESSFUL_TOOL_STAGNATION_HANDOFF_TEMPLATE.format(
            task_hint=task_hint,
            tool_hint=tool_hint,
            repeat_count=max(1, int(repeat_count or 0)),
        )

    @staticmethod
    def build_internal_ui_notice(completion_reason: str) -> str:
        normalized = str(completion_reason or "").strip().lower()
        if normalized.startswith("loop_budget_exhausted"):
            return constants.LOOP_BUDGET_UI_NOTICE
        if normalized == "successful_tool_stagnation":
            return constants.SUCCESSFUL_TOOL_STAGNATION_UI_NOTICE
        if normalized:
            return constants.TOOL_ISSUE_UI_NOTICE
        return constants.DEFAULT_INTERNAL_UI_NOTICE

    @staticmethod
    def _normalize_strategy_kind(strategy: str) -> str:
        normalized = str(strategy or "").strip().lower()
        if normalized in {"normalize_args", "repair_then_rerun", "resume_after_transient_failure"}:
            return "fix_args"
        if normalized in {"refresh_context", "switch_tool"}:
            return "change_tool"
        return "stop" if normalized in {"external_block", "llm_replan"} else "fix_args"

    def plan_recovery(
        self,
        *,
        state: Dict[str, Any],
        messages: List[BaseMessage],
        current_task: str,
        current_turn_id: int,
        open_tool_issue: Dict[str, Any] | None,
        recovery_state: Dict[str, Any],
        last_ai: BaseMessage | None,
        last_message: BaseMessage | None,
        step_count: int,
        max_loops: int,
        hard_loop_ceiling: int,
        auto_repair_enabled: bool,
        max_auto_repairs: int,
        successful_tool_stagnation_limit: int,
    ) -> Dict[str, Any]:
        loop_budget_reached = step_count >= int(max_loops or 0)
        pending_tool_calls = bool(last_ai and getattr(last_ai, "tool_calls", None))
        next_retry_turn_id = current_turn_id
        next_retry_count = int(state.get("self_correction_retry_count", 0) or 0)
        next_fingerprint_history = [
            str(item).strip()
            for item in (state.get("self_correction_fingerprint_history") or [])
            if str(item).strip()
        ]
        next_open_tool_issue = open_tool_issue
        next_recovery_state = deepcopy(recovery_state)
        completion_reason = "no_open_tool_issue"
        handoff_message = ""
        drop_trailing_tool_call = False
        successful_tool_repeat_count = 0
        successful_tool_name = ""
        repair_plan = (
            build_repair_plan(
                open_tool_issue,
                current_task=current_task,
                max_auto_repairs=max_auto_repairs,
            )
            if open_tool_issue and auto_repair_enabled
            else None
        )
        issue_fingerprint = str(
            (open_tool_issue or {}).get("progress_fingerprint")
            or (open_tool_issue or {}).get("fingerprint")
            or ""
        ).strip()

        if loop_budget_reached and pending_tool_calls and not open_tool_issue:
            branch = {
                "completion_reason": "loop_budget_exhausted_pending_tool_call",
                "turn_outcome": "finish_turn",
                "handoff_message": self.build_loop_budget_handoff_text(
                    current_task=current_task,
                    tool_names=[
                        str(tool_call.get("name") or "").strip()
                        for tool_call in (getattr(last_ai, "tool_calls", []) or [])
                        if str(tool_call.get("name") or "").strip()
                    ],
                ),
                "drop_trailing_tool_call": True,
                "next_open_tool_issue": None,
            }
        elif loop_budget_reached:
            next_open_tool_issue = None
            next_recovery_state["external_blocker"] = {
                "reason": "loop_budget_exhausted",
                "issue_summary": "" if not open_tool_issue else open_tool_issue.get("summary", ""),
            }
            branch = {
                "completion_reason": "loop_budget_exhausted",
                "turn_outcome": "finish_turn",
                "next_open_tool_issue": next_open_tool_issue,
                "handoff_message": (
                    self.build_tool_issue_handoff_text(open_tool_issue, repair_plan=repair_plan)
                    if open_tool_issue
                    else ""
                ),
            }
        elif open_tool_issue and int(next_retry_count or 0) >= max(1, int(hard_loop_ceiling or 1)):
            next_recovery_state["active_issue"] = open_tool_issue
            next_recovery_state["active_strategy"] = None
            next_recovery_state["strategy_queue"] = []
            next_recovery_state["external_blocker"] = {
                "reason": "recovery_stagnated",
                "issue_summary": str(open_tool_issue.get("summary", "")),
            }
            branch = {
                "completion_reason": "recovery_stagnated",
                "turn_outcome": "finish_turn",
                "next_open_tool_issue": None,
                "handoff_message": self.build_tool_issue_handoff_text(
                    open_tool_issue,
                    repair_plan=repair_plan,
                ),
            }
        elif not open_tool_issue:
            next_retry_count = 0
            next_fingerprint_history = []
            next_recovery_state["active_issue"] = None
            next_recovery_state["active_strategy"] = None
            next_recovery_state["strategy_queue"] = []
            next_recovery_state["attempts_by_strategy"] = {}
            next_recovery_state["progress_markers"] = []
            next_recovery_state["external_blocker"] = None
            next_recovery_state["llm_replan_attempted_for"] = []
            if isinstance(last_message, ToolMessage):
                next_recovery_state["last_successful_evidence"] = str(
                    state.get("last_tool_result") or stringify_content(last_message.content)
                ).strip()
                successful_tool_repeat_count = self._count_repeated_successful_tool_results(messages)
                successful_tool_name = str(last_message.name or "").strip()
                if successful_tool_repeat_count >= max(2, int(successful_tool_stagnation_limit or 0)):
                    branch = {
                        "completion_reason": "successful_tool_stagnation",
                        "turn_outcome": "continue_agent",
                        "next_open_tool_issue": None,
                    }
                else:
                    branch = {
                        "completion_reason": "tool_result_ready_for_agent",
                        "turn_outcome": "continue_agent",
                        "next_open_tool_issue": None,
                    }
            else:
                branch = {
                    "completion_reason": "no_open_tool_issue",
                    "turn_outcome": "finish_turn",
                    "next_open_tool_issue": None,
                }
        elif repair_plan and (repair_plan.needs_external_input or repair_plan.terminal_reason):
            next_recovery_state["active_issue"] = open_tool_issue
            next_recovery_state["active_strategy"] = None
            next_recovery_state["strategy_queue"] = []
            next_recovery_state["external_blocker"] = {
                "reason": repair_plan.terminal_reason or repair_plan.reason,
                "issue_summary": str(open_tool_issue.get("summary", "")),
            }
            branch = {
                "completion_reason": repair_plan.terminal_reason or "needs_external_input",
                "turn_outcome": "finish_turn",
                "next_open_tool_issue": None,
                "handoff_message": self.build_tool_issue_handoff_text(
                    open_tool_issue,
                    repair_plan=repair_plan,
                ),
            }
        else:
            branch = self._plan_recoverable_issue(
                current_task=current_task,
                open_tool_issue=open_tool_issue,
                repair_plan=repair_plan,
                issue_fingerprint=issue_fingerprint,
                next_recovery_state=next_recovery_state,
            )

        completion_reason = branch["completion_reason"]
        handoff_message = str(branch.get("handoff_message") or "")
        drop_trailing_tool_call = bool(branch.get("drop_trailing_tool_call"))
        next_open_tool_issue = branch.get("next_open_tool_issue", next_open_tool_issue)
        next_retry_count = int(branch.get("next_retry_count", next_retry_count) or next_retry_count)
        next_fingerprint_history = list(branch.get("next_fingerprint_history") or next_fingerprint_history)

        return {
            "turn_id": current_turn_id,
            "turn_outcome": branch["turn_outcome"],
            "current_task": current_task,
            "recovery_state": next_recovery_state,
            "open_tool_issue": next_open_tool_issue,
            "has_protocol_error": False,
            "self_correction_retry_count": next_retry_count,
            "self_correction_retry_turn_id": next_retry_turn_id,
            "self_correction_fingerprint_history": next_fingerprint_history,
            "self_correction_last_reason": completion_reason,
            "handoff_message": handoff_message,
            "completion_reason": completion_reason,
            "drop_trailing_tool_call": drop_trailing_tool_call,
            "had_pending_tool_calls": pending_tool_calls,
            "loop_budget_reached": loop_budget_reached,
            "successful_tool_repeat_count": successful_tool_repeat_count,
            "successful_tool_name": successful_tool_name,
        }

    def apply_recovery(self, recovery_state: Dict[str, Any], *, current_turn_id: int) -> Dict[str, Any]:
        queue = list(recovery_state.get("strategy_queue") or [])
        if not queue:
            return {
                "turn_outcome": "finish_turn",
                "recovery_state": recovery_state,
                "recovery_status": "empty_strategy_queue",
                "turn_id": current_turn_id,
            }

        active_strategy = deepcopy(queue[0])
        recovery_state["active_strategy"] = active_strategy
        recovery_state["strategy_queue"] = deepcopy(queue[1:])
        return {
            "turn_outcome": "",
            "recovery_state": recovery_state,
            "recovery_status": "recovery_prepared",
            "active_strategy": active_strategy,
            "turn_id": current_turn_id,
        }

    def _plan_recoverable_issue(
        self,
        *,
        current_task: str,
        open_tool_issue: Dict[str, Any],
        repair_plan: RepairPlan | None,
        issue_fingerprint: str,
        next_recovery_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        progress_markers = [
            str(item).strip()
            for item in (next_recovery_state.get("progress_markers") or [])
            if str(item).strip()
        ]
        if issue_fingerprint and issue_fingerprint not in progress_markers:
            progress_markers.append(issue_fingerprint)
        next_recovery_state["progress_markers"] = progress_markers

        if not repair_plan:
            repair_plan = RepairPlan(
                strategy="llm_replan",
                reason="missing_repair_plan",
                fingerprint=issue_fingerprint or "missing-plan",
                tool_name=str((open_tool_issue.get("tool_names") or ["unknown_tool"])[0]),
                suggested_tool_name=str((open_tool_issue.get("tool_names") or ["unknown_tool"])[0]),
                original_args=dict(open_tool_issue.get("tool_args") or {}),
                patched_args=dict(open_tool_issue.get("tool_args") or {}),
                notes="Recovery fallback: inspect recent tool output and choose the best next step.",
                llm_guidance="Inspect the failure, gather more context, and continue with a different valid approach.",
            )

        strategy_id = self.repair_plan_strategy_id(repair_plan)
        attempts_by_strategy = dict(next_recovery_state.get("attempts_by_strategy") or {})
        attempt_count = int(attempts_by_strategy.get(strategy_id, 0) or 0) + 1
        attempts_by_strategy[strategy_id] = attempt_count
        next_recovery_state["attempts_by_strategy"] = attempts_by_strategy

        llm_replans = [
            str(item).strip()
            for item in (next_recovery_state.get("llm_replan_attempted_for") or [])
            if str(item).strip()
        ]
        strategy_payload = self.build_recovery_strategy(
            repair_plan=repair_plan,
            open_tool_issue=open_tool_issue,
            current_task=current_task,
            strategy_id=strategy_id,
        )

        if attempt_count == 1:
            next_recovery_state["active_issue"] = open_tool_issue
            next_recovery_state["active_strategy"] = None
            next_recovery_state["strategy_queue"] = [strategy_payload]
            next_recovery_state["external_blocker"] = None
            return {
                "completion_reason": f"recover_{repair_plan.strategy}",
                "turn_outcome": "recover_agent",
                "next_open_tool_issue": open_tool_issue,
                "next_retry_count": attempt_count,
                "next_fingerprint_history": list(progress_markers),
            }

        if repair_plan.strategy != "llm_replan" and issue_fingerprint and issue_fingerprint not in llm_replans:
            llm_replans.append(issue_fingerprint)
            next_recovery_state["llm_replan_attempted_for"] = llm_replans
            llm_replan = RepairPlan(
                strategy="llm_replan",
                reason=f"{repair_plan.reason}_llm_replan",
                fingerprint=repair_plan.fingerprint,
                tool_name=repair_plan.tool_name,
                suggested_tool_name=repair_plan.suggested_tool_name,
                original_args=repair_plan.original_args,
                patched_args=repair_plan.patched_args,
                notes="Deterministic recovery did not clear the issue. Replan from repository state and recent tool output.",
                llm_guidance="Do not stop. Replan using repository state, recent tool failures, and alternative verification or edit paths until you either succeed or hit a real external blocker.",
                progress_fingerprint=repair_plan.progress_fingerprint,
            )
            llm_strategy_id = self.repair_plan_strategy_id(llm_replan)
            attempts_by_strategy[llm_strategy_id] = int(attempts_by_strategy.get(llm_strategy_id, 0) or 0) + 1
            next_recovery_state["attempts_by_strategy"] = attempts_by_strategy
            next_recovery_state["active_issue"] = open_tool_issue
            next_recovery_state["active_strategy"] = None
            next_recovery_state["strategy_queue"] = [
                self.build_recovery_strategy(
                    repair_plan=llm_replan,
                    open_tool_issue=open_tool_issue,
                    current_task=current_task,
                    strategy_id=llm_strategy_id,
                )
            ]
            next_recovery_state["external_blocker"] = None
            return {
                "completion_reason": "recover_llm_replan",
                "turn_outcome": "recover_agent",
                "next_open_tool_issue": open_tool_issue,
                "next_retry_count": max(int(value or 0) for value in attempts_by_strategy.values()),
                "next_fingerprint_history": list(progress_markers),
            }

        next_recovery_state["active_issue"] = open_tool_issue
        next_recovery_state["active_strategy"] = None
        next_recovery_state["strategy_queue"] = []
        next_recovery_state["external_blocker"] = {
            "reason": "recovery_stagnated",
            "issue_summary": str(open_tool_issue.get("summary", "")),
        }
        return {
            "completion_reason": "recovery_stagnated",
            "turn_outcome": "finish_turn",
            "next_open_tool_issue": None,
            "handoff_message": self.build_tool_issue_handoff_text(
                open_tool_issue,
                repair_plan=repair_plan,
            ),
            "next_retry_count": max(int(value or 0) for value in attempts_by_strategy.values()),
            "next_fingerprint_history": list(progress_markers),
        }

    def _count_repeated_successful_tool_results(self, messages: List[BaseMessage]) -> int:
        streak = 0
        target_fingerprint = ""
        for message in reversed(messages or []):
            if isinstance(message, ToolMessage):
                parsed = parse_tool_execution_result(stringify_content(message.content))
                if not parsed.ok:
                    break
                fingerprint = self._successful_tool_result_fingerprint(message)
                if not fingerprint:
                    break
                if not target_fingerprint:
                    target_fingerprint = fingerprint
                    streak = 1
                    continue
                if fingerprint != target_fingerprint:
                    break
                streak += 1
                continue
            if isinstance(message, (AIMessage, AIMessageChunk)):
                if getattr(message, "tool_calls", None):
                    continue
                break
            if isinstance(message, HumanMessage):
                break
        return streak

    @staticmethod
    def _successful_tool_result_fingerprint(message: ToolMessage) -> str:
        metadata = getattr(message, "additional_kwargs", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        payload = {
            "tool_name": str(message.name or "").strip(),
            "tool_args": metadata.get("tool_args") or {},
            "content": compact_text(stringify_content(message.content).strip(), 240),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
