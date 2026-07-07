from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


PlanStepStatus = Literal["pending", "in_progress", "completed", "blocked", "skipped"]
PlanStatus = Literal[
    "draft",
    "pending_approval",
    "approved",
    "needs_changes",
    "rebuild_requested",
    "rejected",
    "executing",
    "completed",
    "replan_pending",
]
PlanComplexity = Literal["low", "medium", "high"]


class RuntimePlanStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Stable step identifier, e.g. '1' or 'inspect-runtime'.")
    title: str
    description: str
    status: PlanStepStatus = "pending"

    @field_validator("id", "title", "description", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return " ".join(str(value or "").replace("\r\n", "\n").split()).strip()

    @model_validator(mode="after")
    def _validate_required_text(self) -> "RuntimePlanStep":
        if not self.id:
            raise ValueError("step.id must be non-empty")
        if not self.title:
            raise ValueError("step.title must be non-empty")
        if not self.description:
            raise ValueError("step.description must be non-empty")
        return self


class RuntimePlanDraft(BaseModel):
    """LLM-facing structured plan payload.

    Runtime fields such as id/version/status are attached after validation so
    model output cannot accidentally overwrite durable bookkeeping.
    """

    model_config = ConfigDict(extra="ignore")

    summary: str
    steps: list[RuntimePlanStep] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    estimated_tools: list[str] = Field(default_factory=list)
    estimated_files: list[str] = Field(default_factory=list)
    complexity: PlanComplexity = "medium"

    @field_validator(
        "summary",
        mode="before",
    )
    @classmethod
    def _normalize_summary(cls, value: Any) -> str:
        return " ".join(str(value or "").replace("\r\n", "\n").split()).strip()

    @field_validator("risks", "assumptions", "verification", "estimated_tools", "estimated_files", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = " ".join(str(item or "").replace("\r\n", "\n").split()).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    @model_validator(mode="after")
    def _validate_summary(self) -> "RuntimePlanDraft":
        if not self.summary:
            raise ValueError("summary must be non-empty")
        return self


class RuntimePlan(RuntimePlanDraft):
    id: str
    version: int = 1
    status: PlanStatus = "draft"
    active_step_id: str = ""


def new_plan_id() -> str:
    return f"plan-{uuid.uuid4().hex[:12]}"


def normalize_plan_action(value: Any) -> str:
    if isinstance(value, dict):
        raw = value.get("action") or value.get("choice") or value.get("value") or value.get("key") or ""
    else:
        raw = value
    text = " ".join(str(raw or "").split()).strip().casefold()
    if text in {"implement", "approve", "approved", "execute", "реализовать", "да, реализовать"}:
        return "implement"
    if text in {
        "revise",
        "edit",
        "change",
        "needs_changes",
        "внести изменения в план",
        "внести правки/дополнения в план",
    }:
        return "revise"
    if text in {"rebuild", "rebuild_plan", "перестроить план"}:
        return "rebuild"
    if text in {"cancel", "reject", "rejected", "decline", "отменить", "нет, отказаться от реализации"}:
        return "cancel"
    return text or "cancel"


def normalize_plan_feedback(value: Any) -> str:
    if isinstance(value, dict):
        raw = value.get("feedback") or value.get("choice") or value.get("value") or ""
    else:
        raw = value
    return str(raw or "").replace("\r\n", "\n").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("empty structured plan response")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else stripped
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("structured plan response must be a JSON object")
    return payload


def build_runtime_plan(
    draft_payload: Any,
    *,
    previous_plan: dict[str, Any] | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    if isinstance(draft_payload, RuntimePlanDraft):
        draft = draft_payload
    elif isinstance(draft_payload, RuntimePlan):
        draft = RuntimePlanDraft.model_validate(draft_payload.model_dump())
    else:
        draft = RuntimePlanDraft.model_validate(draft_payload)

    previous = previous_plan if isinstance(previous_plan, dict) and not rebuild else {}
    plan_id = str(previous.get("id") or "").strip() or new_plan_id()
    previous_version = int(previous.get("version", 0) or 0) if previous else 0
    version = 1 if rebuild or not previous else previous_version + 1

    normalized = RuntimePlan(
        **draft.model_dump(),
        id=plan_id,
        version=max(1, version),
        status="pending_approval",
        active_step_id="",
    )
    data = normalized.model_dump()
    for index, step in enumerate(data["steps"], start=1):
        step["id"] = str(step.get("id") or index)
        step["status"] = "pending"
    return data


def coerce_runtime_plan(plan: Any) -> dict[str, Any] | None:
    if not isinstance(plan, dict):
        return None
    if plan.get("format") == "markdown":
        content = str(plan.get("content") or "").strip()
        if not content:
            return None
        return {
            "id": str(plan.get("id") or new_plan_id()),
            "version": int(plan.get("version", 1) or 1),
            "summary": content.splitlines()[0] if content.splitlines() else "Legacy markdown plan",
            "steps": [
                {
                    "id": "1",
                    "title": "Execute legacy plan",
                    "description": content,
                    "status": "pending",
                }
            ],
            "risks": [],
            "assumptions": ["Legacy markdown plan was loaded from an older checkpoint."],
            "verification": [],
            "estimated_tools": [],
            "estimated_files": [],
            "complexity": "medium",
            "status": str(plan.get("status") or "pending_approval"),
            "active_step_id": str(plan.get("active_step_id") or ""),
        }
    try:
        return RuntimePlan.model_validate(plan).model_dump()
    except ValidationError:
        return None


def update_plan_status(
    plan: dict[str, Any] | None,
    *,
    status: PlanStatus | None = None,
    active_step_id: str | None = None,
) -> dict[str, Any] | None:
    normalized = coerce_runtime_plan(plan)
    if normalized is None:
        return None
    updated = deepcopy(normalized)
    if status is not None:
        updated["status"] = status
    if active_step_id is not None:
        updated["active_step_id"] = active_step_id
    return updated


def set_step_status(plan: dict[str, Any] | None, step_id: str, status: PlanStepStatus) -> dict[str, Any] | None:
    normalized = coerce_runtime_plan(plan)
    if normalized is None:
        return None
    updated = deepcopy(normalized)
    target_id = str(step_id or "").strip()
    for step in updated.get("steps", []):
        if str(step.get("id") or "").strip() == target_id:
            step["status"] = status
            break
    updated["active_step_id"] = target_id if status == "in_progress" else ""
    if status == "in_progress":
        updated["status"] = "executing"
    return updated


def first_pending_step(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    normalized = coerce_runtime_plan(plan)
    if normalized is None:
        return None
    for step in normalized.get("steps", []):
        if str(step.get("status") or "").strip() == "pending":
            return dict(step)
    return None


def active_step(plan: dict[str, Any] | None, active_step_id: str | None = None) -> dict[str, Any] | None:
    normalized = coerce_runtime_plan(plan)
    if normalized is None:
        return None
    target_id = str(active_step_id or normalized.get("active_step_id") or "").strip()
    if not target_id:
        return None
    for step in normalized.get("steps", []):
        if str(step.get("id") or "").strip() == target_id:
            return dict(step)
    return None


def plan_is_complete(plan: dict[str, Any] | None) -> bool:
    normalized = coerce_runtime_plan(plan)
    if normalized is None:
        return False
    return all(str(step.get("status") or "") in {"completed", "skipped"} for step in normalized.get("steps", []))


def render_plan_markdown(plan: dict[str, Any] | None) -> str:
    normalized = coerce_runtime_plan(plan)
    if normalized is None:
        return ""
    lines = [
        f"## План v{normalized.get('version', 1)}",
        "",
        str(normalized.get("summary") or "").strip(),
        "",
        "### Этапы",
    ]
    for step in normalized.get("steps", []):
        status = str(step.get("status") or "pending")
        lines.append(
            f"{step.get('id')}. **{step.get('title')}** [{status}]\n"
            f"   {step.get('description')}"
        )
    if normalized.get("risks"):
        lines.extend(["", "### Риски", *[f"- {item}" for item in normalized["risks"]]])
    if normalized.get("assumptions"):
        lines.extend(["", "### Предположения", *[f"- {item}" for item in normalized["assumptions"]]])
    if normalized.get("verification"):
        lines.extend(["", "### Проверка", *[f"- {item}" for item in normalized["verification"]]])
    if normalized.get("estimated_tools"):
        lines.extend(["", "### Предполагаемые инструменты", *[f"- {item}" for item in normalized["estimated_tools"]]])
    if normalized.get("estimated_files"):
        lines.extend(["", "### Предполагаемые файлы", *[f"- {item}" for item in normalized["estimated_files"]]])
    lines.extend(["", f"Сложность: `{normalized.get('complexity', 'medium')}`"])
    return "\n".join(lines).strip()
