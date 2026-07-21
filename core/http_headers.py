from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.constants import BASE_DIR

logger = logging.getLogger("agent")


def load_openai_headers(path: Path | None = None) -> dict[str, str]:
    """Load editable headers for OpenAI-compatible requests without failing startup.

    When the headers file is absent, an empty mapping is returned so the
    underlying SDK sends its own standard headers (no spoofing). When the file
    exists, only its string-valued entries are applied as overrides.
    """
    headers_path = path or BASE_DIR / "headers.json"
    try:
        payload: Any = json.loads(headers_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Could not load OpenAI-compatible headers from %s: %s", headers_path, exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("Ignoring OpenAI-compatible headers from %s: expected a JSON object", headers_path)
        return {}

    return {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and key.strip() and isinstance(value, str)
    }
