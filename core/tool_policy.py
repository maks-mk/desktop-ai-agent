from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    read_only: bool = False
    mutating: bool = False
    destructive: bool = False
    requires_approval: bool = False
    networked: bool = False
    source: str = "local"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_tool_metadata(name: str, source: str = "local") -> ToolMetadata:
    if source == "mcp":
        return ToolMetadata(name=name, mutating=True, requires_approval=True, networked=True, source=source)
    return ToolMetadata(name=name, read_only=True, source=source)
