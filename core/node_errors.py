class ProviderContextError(RuntimeError):
    """Raised when the agent context violates provider message-ordering constraints."""


class EmptyLLMResponseError(RuntimeError):
    """Raised when the provider returns no content, tool calls, or invalid-call details."""
