import asyncio
import logging
from typing import Any, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from core.api_key_rotation import RotatingChatModel
from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core.logging_config import setup_logging
from core.multimodal import extract_model_capabilities
from core.nodes import AgentNodes
from core.run_logger import JsonlRunLogger
from core.state import AgentState
from tools.tool_registry import ToolRegistry

logger = logging.getLogger("agent")


# --- Factories ---

def prepare_llm_with_tools(
    llm: BaseChatModel,
    tools: list[Any],
) -> tuple[BaseChatModel, bool, str]:
    """Bind tools once and report whether structured tool calling is actually available."""
    if not tools:
        return llm, False, ""

    binder = getattr(llm, "bind_tools", None)
    if not callable(binder):
        return llm, False, "LLM backend does not implement bind_tools()."

    try:
        return binder(tools), True, ""
    except Exception as exc:
        return llm, False, str(exc)

def create_llm(config: AgentConfig, *, api_key_override: str | None = None) -> BaseChatModel:
    """Initializes LLM based on configuration."""
    if config.provider == "gemini":
        # Lazy import to avoid loading both providers on startup.
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Безопасное извлечение ключа (защита от краша, если ключ None)
        if api_key_override is None:
            api_key = config.gemini_api_key.get_secret_value() if config.gemini_api_key else None
        else:
            api_key = str(api_key_override or "")
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=config.temperature,
            google_api_key=api_key,
        )
    if config.provider == "openai":
        # Lazy import to avoid loading both providers on startup.
        from langchain_openai import ChatOpenAI

        if api_key_override is None:
            api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else None
        else:
            api_key = str(api_key_override or "")
        return ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            api_key=api_key,
            base_url=config.openai_base_url,
            max_retries=0,
            stream_usage=True,
        )
    raise ValueError(f"Unknown provider: {config.provider}")


def create_runtime_llm(config: AgentConfig) -> BaseChatModel | RotatingChatModel:
    profile_id = str(config.active_model_profile_id or "").strip()
    if not profile_id:
        return create_llm(config)
    return RotatingChatModel(
        config=config,
        profile_id=profile_id,
        profile_store_path=config.model_profile_config_path,
        llm_factory=create_llm,
    )


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
    workflow.add_node("recovery", nodes.recovery_node)
    workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})

    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "update_step")
    workflow.add_edge("update_step", "agent")

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

        turn_outcome = str(state.get("turn_outcome") or "").strip().lower()
        has_open_tool_issue = bool(state.get("open_tool_issue"))
        has_protocol_error = bool(state.get("has_protocol_error"))

        if tools_enabled and turn_outcome == "run_tools":
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

        if turn_outcome == "recover_agent" or has_open_tool_issue or has_protocol_error:
            return "recovery"

        return END

    def route_after_tools(state: AgentState):
        if state.get("open_tool_issue"):
            return "recovery"
        return "update_step"

    def route_after_recovery(state: AgentState):
        if state.get("turn_outcome") == "recover_agent":
            return "update_step"
        return END

    if tools_enabled:
        agent_routes = ["tools", "recovery", END]
        if approval_enabled:
            agent_routes.insert(0, "approval")
            workflow.add_edge("approval", "tools")
        workflow.add_conditional_edges("agent", route_after_agent, agent_routes)
        workflow.add_conditional_edges("tools", route_after_tools, ["recovery", "update_step"])
    else:
        workflow.add_conditional_edges("agent", route_after_agent, ["recovery", END])

    workflow.add_conditional_edges("recovery", route_after_recovery, ["update_step", END])

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

    active_tools = tools if tool_calling_enabled else []
    active_tool_metadata = tool_registry.tool_metadata if tool_calling_enabled else {}
    nodes = AgentNodes(
        config=config,
        llm=llm,
        tools=active_tools,
        llm_with_tools=llm_with_tools,
        tool_metadata=active_tool_metadata,
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
    setup_logging(level=config.log_level, log_file=config.log_file)

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
