import asyncio
import logging
from typing import Any, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

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

def create_llm(config: AgentConfig) -> BaseChatModel:
    """Initializes LLM based on configuration."""
    if config.provider == "gemini":
        # Lazy import to avoid loading both providers on startup.
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Безопасное извлечение ключа (защита от краша, если ключ None)
        api_key = config.gemini_api_key.get_secret_value() if config.gemini_api_key else None
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            temperature=config.temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )
    if config.provider == "openai":
        # Lazy import to avoid loading both providers on startup.
        from langchain_openai import ChatOpenAI

        api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else None
        return ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            api_key=api_key,
            base_url=config.openai_base_url,
            stream_usage=True,
        )
    raise ValueError(f"Unknown provider: {config.provider}")


# --- Builder ---

def create_agent_workflow(
    nodes: AgentNodes,
    config: AgentConfig,
    tools_enabled: Optional[bool] = None,
) -> StateGraph:
    """Builds the LangGraph workflow with explicit turn classification and bounded recovery."""
    tools_enabled = bool(nodes.tools) and config.model_supports_tools if tools_enabled is None else tools_enabled
    approval_enabled = bool(tools_enabled and config.enable_approvals)

    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", nodes.summarize_node)
    workflow.add_node("classify_turn", nodes.classify_turn_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("recovery", nodes.recovery_node)
    workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})

    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "classify_turn")
    workflow.add_edge("classify_turn", "update_step")
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

async def build_agent_app(config: Optional[AgentConfig] = None) -> Tuple[Any, ToolRegistry]:
    """
    Builds the LangGraph application and returns it along with the tool registry.
    """
    # Pydantic AgentConfig автоматически загружает .env.
    config = config or AgentConfig()
    setup_logging(level=config.log_level, log_file=config.log_file)

    logger.info(f"Initializing agent: [bold cyan]{config.provider}[/]", extra={"markup": True})

    # 1. Initialize Resources
    llm = create_llm(config)
    tool_registry = ToolRegistry(config)
    tool_registry.model_capabilities = extract_model_capabilities(llm)
    await tool_registry.load_all()
    checkpoint_runtime = await create_checkpoint_runtime(config)
    run_logger = JsonlRunLogger(config.run_log_dir)
    tool_registry.checkpoint_info = checkpoint_runtime.to_dict()
    tool_registry.register_cleanup_callback(checkpoint_runtime.aclose)

    # 2. Bind Tools
    tools = tool_registry.tools
    can_use_tools = config.model_supports_tools

    llm_with_tools = llm
    if tools and can_use_tools:
        try:
            llm_with_tools = llm.bind_tools(tools)
            logger.info("🛠️ Tools bound to LLM successfully.")
        except Exception as e:
            logger.error(f"Failed to bind tools: {e}")
    else:
        if not can_use_tools:
            logger.debug("⚠️ Tools disabled: Model does not support tool calling.")

    # 3. Create Nodes
    nodes = AgentNodes(
        config=config,
        llm=llm,
        tools=tools,
        llm_with_tools=llm_with_tools,
        tool_metadata=tool_registry.tool_metadata,
        run_logger=run_logger,
    )
    workflow = create_agent_workflow(nodes, config, tools_enabled=bool(tools) and can_use_tools)
    return workflow.compile(checkpointer=checkpoint_runtime.checkpointer), tool_registry


if __name__ == "__main__":

    async def main():
        app, registry = await build_agent_app()
        print(f"✔ Agent Ready. Tools: {len(registry.tools)}")
        await registry.cleanup()

    asyncio.run(main())
