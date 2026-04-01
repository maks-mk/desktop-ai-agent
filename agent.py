import asyncio
import logging
from typing import Any, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core.logging_config import setup_logging
from core.nodes import AgentNodes
from core.run_logger import JsonlRunLogger
from core.state import AgentState
from tools.tool_registry import ToolRegistry

# Setup logging
try:
    logger = setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)
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
    """Builds the LangGraph workflow where critic validates only final agent answers."""
    tools_enabled = bool(nodes.tools) and config.model_supports_tools if tools_enabled is None else tools_enabled

    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", nodes.summarize_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("critic", nodes.critic_node)
    workflow.add_node("prepare_retry", nodes.prepare_retry_node)
    workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})

    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "update_step")
    workflow.add_edge("update_step", "agent")

    if tools_enabled:
        workflow.add_node("approval", nodes.approval_node)
        workflow.add_node("tools", nodes.tools_node)

    def should_run_critic_after_agent(state: AgentState) -> bool:
        mode = str(getattr(config, "critic_mode", "hybrid") or "hybrid").strip().lower()
        if mode == "always":
            return True
        if mode == "off":
            return False
        # Hybrid mode: run critic only for higher-risk branches.
        critic_source = str(state.get("critic_source", "") or "").strip().lower()
        if critic_source == "tools":
            return True
        if state.get("open_tool_issue"):
            return True
        if bool(state.get("has_protocol_error")):
            return True
        return False

    def route_after_agent(state: AgentState):
        steps = state.get("steps", 0)

        if steps >= config.max_loops:
            logger.debug(f"🛑 Loop Guard: {steps} steps reached.")
            return END

        messages = state.get("messages") or []
        if not messages:
            logger.warning("Agent node returned no messages; routing to critic for a safe fallback.")
            return "critic"

        last_msg = messages[-1]

        if tools_enabled and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            if nodes.tool_calls_require_approval(last_msg.tool_calls):
                return "approval"
            return "tools"

        return "critic" if should_run_critic_after_agent(state) else END

    def route_after_critic(state: AgentState):
        if state.get("turn_outcome") == "finish_turn":
            return END
        if state.get("turn_outcome") == "retry_agent":
            return "prepare_retry"
        return END

    if tools_enabled:
        workflow.add_conditional_edges("agent", route_after_agent, ["approval", "tools", "critic", END])
        workflow.add_edge("approval", "tools")
        workflow.add_edge("tools", "update_step")
    else:
        workflow.add_conditional_edges("agent", route_after_agent, ["critic", END])

    workflow.add_conditional_edges("critic", route_after_critic, ["prepare_retry", END])
    workflow.add_edge("prepare_retry", "update_step")

    return workflow

async def build_agent_app(config: Optional[AgentConfig] = None) -> Tuple[Any, ToolRegistry]:
    """
    Builds the LangGraph application and returns it along with the tool registry.
    """
    # Pydantic AgentConfig автоматически загружает .env.
    config = config or AgentConfig()

    logger.info(f"Initializing agent: [bold cyan]{config.provider}[/]", extra={"markup": True})

    # 1. Initialize Resources
    llm = create_llm(config)
    tool_registry = ToolRegistry(config)
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
