import asyncio
import logging
from typing import Any, Optional, Tuple

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core.logging_config import setup_logging
from core.model_profiles import ModelProfileStore, find_active_profile, find_profile_by_id
from core.multimodal import extract_model_capabilities, resolve_model_capabilities
from core.nodes import AgentNodes
from core.providers import (
    create_llm,
    create_runtime_llm,
    prepare_llm_with_tools,
)
# Back-compat re-exports — tests reference these via ``agent_module._...``.
from core.providers.gemini import (
    gemini_model_supports_thinking_budget as _gemini_model_supports_thinking_budget,
    patch_langchain_google_genai_retry_kwargs as _patch_langchain_google_genai_retry_kwargs,
)
from core.providers.openai_reasoning import (
    extract_openai_reasoning_delta as _extract_openai_reasoning_delta,
)
from core.run_logger import JsonlRunLogger
from core.state import AgentState
from core.turn_outcomes import (
    TURN_OUTCOME_CONTINUE_AGENT,
    TURN_OUTCOME_FINISH_TURN,
    TURN_OUTCOME_RECOVER_AGENT,
    TURN_OUTCOME_RUN_TOOLS,
    normalize_turn_outcome,
)
from tools.tool_registry import ToolRegistry

logger = logging.getLogger("agent")


def _register_llm_cleanup_callback(tool_registry: ToolRegistry, llm: Any) -> bool:
    close_method = getattr(llm, "aclose", None) or getattr(llm, "close", None)
    if callable(close_method):
        tool_registry.register_cleanup_callback(close_method)
        return True
    for target in (
        getattr(llm, "root_async_client", None),
        getattr(llm, "async_client", None),
        getattr(llm, "root_client", None),
        getattr(llm, "client", None),
    ):
        if target is None:
            continue
        target_close = getattr(target, "aclose", None) or getattr(target, "close", None)
        if callable(target_close):
            tool_registry.register_cleanup_callback(target_close)
            return True
    return False


def _resolve_effective_model_capabilities(config: AgentConfig, runtime_capabilities: dict[str, Any]) -> dict[str, Any]:
    profiles_payload = ModelProfileStore(config.model_profile_config_path).load()
    selected_profile_id = str(config.active_model_profile_id or "").strip()
    active_profile = (
        find_profile_by_id(profiles_payload, selected_profile_id)
        if selected_profile_id
        else find_active_profile(profiles_payload)
    )
    return resolve_model_capabilities(active_profile, runtime_capabilities)


# --- Builder ---


def create_agent_workflow(
    nodes: AgentNodes,
    config: AgentConfig,
    tools_enabled: Optional[bool] = None,
) -> StateGraph:
    """Builds the LangGraph workflow with tool-call based routing and bounded recovery."""
    tools_enabled = bool(nodes.tools) and config.model_supports_tools if tools_enabled is None else tools_enabled
    approval_enabled = bool(tools_enabled and config.enable_approvals)

    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", nodes.summarize_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("plan_build", nodes.plan_build_node)
    workflow.add_node("plan_review", nodes.plan_review_node)
    workflow.add_node("plan_revision_input", nodes.plan_revision_input_node)
    workflow.add_node("plan_select_step", nodes.plan_select_step_node)
    workflow.add_node("plan_complete_step", nodes.plan_complete_step_node)
    workflow.add_node("plan_block_step", nodes.plan_block_step_node)
    workflow.add_node("recovery", nodes.recovery_node)
    workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})

    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "update_step")

    def _plan_graph_active(state: AgentState) -> bool:
        return bool(state.get("plan_graph_active") or state.get("current_plan"))

    def _plan_status(state: AgentState) -> str:
        return str(state.get("plan_status") or "").strip().lower()

    def route_after_update_step(state: AgentState):
        if _plan_graph_active(state):
            status = _plan_status(state)
            # If the previous run was interrupted right before ``plan_review``
            # (e.g. by a 429 error), the checkpoint may contain
            # ``plan_status='pending_approval'`` with a valid ``current_plan``
            # but no active interrupt task. A plain text follow-up ("Продолжай")
            # must not bypass approval and jump straight to the agent.
            if status == "pending_approval" and state.get("current_plan"):
                return "plan_review"
            if status == "approved":
                return "plan_select_step"
            if status == "executing" and not str(state.get("active_plan_step_id") or "").strip():
                return "plan_select_step"
        return "agent"

    workflow.add_conditional_edges(
        "update_step", route_after_update_step, ["plan_select_step", "plan_review", "agent"],
    )

    if tools_enabled:
        workflow.add_node("tools", nodes.tools_node)
        if approval_enabled:
            workflow.add_node("approval", nodes.approval_node)

    def route_after_agent(state: AgentState):
        steps = state.get("steps", 0)
        messages = state.get("messages") or []

        if not messages:
            logger.warning("Agent node returned no messages; ending turn safely.")
            return END

        turn_outcome = normalize_turn_outcome(state.get("turn_outcome"))
        has_open_tool_issue = bool(state.get("open_tool_issue"))
        has_protocol_error = bool(state.get("has_protocol_error"))
        plan_graph_active = _plan_graph_active(state)
        plan_status = _plan_status(state)

        if plan_graph_active and plan_status in {"draft", "needs_changes", "rebuild_requested"}:
            if tools_enabled and turn_outcome == TURN_OUTCOME_RUN_TOOLS:
                if steps >= config.max_loops:
                    logger.warning(
                        "Plan-mode loop guard reached at step %s/%s with pending tool calls. Routing to recovery.",
                        steps,
                        config.max_loops,
                    )
                    return "recovery"
                pending_ai_with_tools = nodes._get_last_pending_ai_with_tool_calls(messages)
                if isinstance(pending_ai_with_tools, AIMessage) and pending_ai_with_tools.tool_calls:
                    return "tools"
                return "recovery"
            if turn_outcome == TURN_OUTCOME_RECOVER_AGENT or has_open_tool_issue or has_protocol_error:
                return "recovery"
            return "plan_build"

        if plan_graph_active and plan_status == "executing" and str(state.get("active_plan_step_id") or "").strip():
            if (
                turn_outcome == TURN_OUTCOME_FINISH_TURN
                and not has_open_tool_issue
                and not has_protocol_error
                and nodes.plan_step_completion_requested(state)
            ):
                return "plan_complete_step"

        if tools_enabled and turn_outcome == TURN_OUTCOME_RUN_TOOLS:
            if steps >= config.max_loops:
                logger.warning(
                    "Loop guard reached at step %s/%s with pending tool calls. Routing to recovery.",
                    steps,
                    config.max_loops,
                )
                return "recovery"
            pending_ai_with_tools = nodes._get_last_pending_ai_with_tool_calls(messages)
            if isinstance(pending_ai_with_tools, AIMessage) and pending_ai_with_tools.tool_calls:
                if approval_enabled and nodes.tool_calls_require_approval(pending_ai_with_tools.tool_calls):
                    return "approval"
                return "tools"
            logger.warning(
                "Agent reported run_tools outcome without a valid tool call payload. "
                "Routing to recovery."
            )
            return "recovery"

        if turn_outcome == TURN_OUTCOME_RECOVER_AGENT or has_open_tool_issue or has_protocol_error:
            return "recovery"

        return END

    def route_after_tools(state: AgentState):
        if state.get("open_tool_issue"):
            return "recovery"
        return "update_step"

    def route_after_recovery(state: AgentState):
        normalized = normalize_turn_outcome(state.get("turn_outcome"))
        if (
            _plan_graph_active(state)
            and _plan_status(state) == "executing"
            and str(state.get("active_plan_step_id") or "").strip()
            and normalized == TURN_OUTCOME_FINISH_TURN
        ):
            return "plan_block_step"
        if normalized in (TURN_OUTCOME_RECOVER_AGENT, TURN_OUTCOME_CONTINUE_AGENT):
            return "update_step"
        return END

    def route_after_plan_build(state: AgentState):
        if state.get("open_tool_issue") or state.get("has_protocol_error"):
            return "recovery"
        if _plan_status(state) == "pending_approval" and state.get("current_plan"):
            return "plan_review"
        return "recovery"

    def route_after_plan_review(state: AgentState):
        status = _plan_status(state)
        if status == "approved":
            return "plan_select_step"
        if status == "needs_changes":
            return "plan_revision_input"
        if status == "rebuild_requested":
            return "plan_build"
        return END

    def route_after_plan_select(state: AgentState):
        if state.get("open_tool_issue") or state.get("has_protocol_error"):
            return "recovery"
        status = _plan_status(state)
        if status in {"completed", "rejected"}:
            return END
        return "agent"

    def route_after_plan_block(state: AgentState):
        if _plan_status(state) == "rebuild_requested":
            return "plan_build"
        return END

    if tools_enabled:
        agent_routes = ["tools", "recovery", "plan_build", "plan_complete_step", END]
        if approval_enabled:
            agent_routes.insert(0, "approval")
            workflow.add_edge("approval", "tools")
        workflow.add_conditional_edges("agent", route_after_agent, agent_routes)
        workflow.add_conditional_edges("tools", route_after_tools, ["recovery", "update_step"])
    else:
        workflow.add_conditional_edges("agent", route_after_agent, ["recovery", "plan_build", "plan_complete_step", END])

    workflow.add_conditional_edges("plan_build", route_after_plan_build, ["plan_review", "recovery"])
    workflow.add_conditional_edges("plan_review", route_after_plan_review, ["plan_select_step", "plan_revision_input", "plan_build", END])
    workflow.add_edge("plan_revision_input", "plan_build")
    workflow.add_conditional_edges("plan_select_step", route_after_plan_select, ["agent", "recovery", END])
    workflow.add_edge("plan_complete_step", "plan_select_step")
    workflow.add_conditional_edges("plan_block_step", route_after_plan_block, ["plan_build", END])
    workflow.add_conditional_edges("recovery", route_after_recovery, ["update_step", "plan_block_step", END])

    return workflow


def build_compiled_agent(
    config: AgentConfig,
    tool_registry: ToolRegistry,
    checkpoint_runtime: Any,
    *,
    run_logger: Optional[JsonlRunLogger] = None,
) -> Tuple[Any, ToolRegistry]:
    """Compile an agent app using already-loaded tools and an existing checkpointer."""
    llm = create_runtime_llm(config)
    tool_registry.config = config
    tool_registry.model_capabilities = extract_model_capabilities(llm)
    effective_model_capabilities = _resolve_effective_model_capabilities(
        config,
        tool_registry.model_capabilities,
    )
    tool_registry.checkpoint_info = checkpoint_runtime.to_dict()
    tool_registry.checkpoint_runtime = checkpoint_runtime

    tools = tool_registry.tools
    tool_calling_enabled = bool(tools) and config.model_supports_tools
    llm_with_tools = llm
    if tool_calling_enabled:
        llm_with_tools, tool_calling_enabled, bind_error = prepare_llm_with_tools(llm, tools)
        if tool_calling_enabled:
            logger.info("🛠️ Tools bound to LLM successfully.")
        else:
            tool_registry.loader_status.append(
                {
                    "loader": "llm_tool_binding",
                    "module": config.provider,
                    "loaded_tools": [],
                    "error": bind_error,
                }
            )
            logger.error("Tool calling disabled for this runtime because tool binding failed: %s", bind_error)
    elif not config.model_supports_tools:
        logger.debug("⚠️ Tools disabled: Model does not support tool calling.")

    registered_cleanup_ids: set[int] = set()
    for cleanup_target in (llm, llm_with_tools):
        marker = id(cleanup_target)
        if marker in registered_cleanup_ids:
            continue
        if _register_llm_cleanup_callback(tool_registry, cleanup_target):
            registered_cleanup_ids.add(marker)

    active_tools = tools if tool_calling_enabled else []
    active_tool_metadata = tool_registry.tool_metadata if tool_calling_enabled else {}
    nodes = AgentNodes(
        config=config,
        llm=llm,
        tools=active_tools,
        llm_with_tools=llm_with_tools,
        tool_metadata=active_tool_metadata,
        model_capabilities=effective_model_capabilities,
        run_logger=run_logger,
    )
    workflow = create_agent_workflow(nodes, config, tools_enabled=tool_calling_enabled)
    return workflow.compile(checkpointer=checkpoint_runtime.checkpointer), tool_registry


async def build_agent_app(config: Optional[AgentConfig] = None) -> Tuple[Any, ToolRegistry]:
    """
    Builds the LangGraph application and returns it along with the tool registry.
    """
    # Pydantic AgentConfig автоматически загружает .env.
    config = config or AgentConfig()
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
        reasoning_debug_enabled=config.debug_reasoning_stream,
    )

    logger.info(f"Initializing agent: [bold cyan]{config.provider}[/]", extra={"markup": True})

    # 1. Initialize Resources
    tool_registry = ToolRegistry(config)
    await tool_registry.load_all()
    checkpoint_runtime = await create_checkpoint_runtime(config)
    run_logger = JsonlRunLogger(config.run_log_dir)
    tool_registry.checkpoint_info = checkpoint_runtime.to_dict()
    tool_registry.checkpoint_runtime = checkpoint_runtime
    tool_registry.register_cleanup_callback(checkpoint_runtime.aclose)
    return build_compiled_agent(
        config,
        tool_registry,
        checkpoint_runtime,
        run_logger=run_logger,
    )


if __name__ == "__main__":

    async def main():
        app, registry = await build_agent_app()
        print(f"✔ Agent Ready. Tools: {len(registry.tools)}")
        await registry.cleanup()

    asyncio.run(main())
