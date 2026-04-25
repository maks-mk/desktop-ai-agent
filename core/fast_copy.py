from __future__ import annotations

from copy import deepcopy
from typing import Any

_IMMUTABLE_SCALARS = (str, int, float, bool, bytes, complex, type(None))


def copy_jsonish(value: Any) -> Any:
    """Fast-path copy for the JSON-like payloads we pass through tool/runtime state."""
    value_type = type(value)
    if value_type in _IMMUTABLE_SCALARS:
        return value
    if value_type is dict:
        return {key: copy_jsonish(item) for key, item in value.items()}
    if value_type is list:
        return [copy_jsonish(item) for item in value]
    if value_type is tuple:
        return tuple(copy_jsonish(item) for item in value)
    if value_type is set:
        return {copy_jsonish(item) for item in value}
    if value_type is frozenset:
        return frozenset(copy_jsonish(item) for item in value)
    return deepcopy(value)
