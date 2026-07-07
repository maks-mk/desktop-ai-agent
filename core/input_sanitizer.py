from __future__ import annotations

from dataclasses import dataclass
import unicodedata


DEFAULT_USER_INPUT_LIMIT = 10_000
_ALLOWED_CONTROL_CHARS = {"\n", "\t"}


@dataclass(frozen=True)
class SanitizedUserInput:
    text: str
    original_length: int
    removed_control_chars: int = 0
    truncated: bool = False

    @property
    def changed(self) -> bool:
        return (
            self.removed_control_chars > 0
            or self.truncated
            or len(self.text) != self.original_length
        )


def sanitize_user_text(raw_text: str, *, max_chars: int = DEFAULT_USER_INPUT_LIMIT) -> SanitizedUserInput:
    text = str(raw_text or "")
    original_length = len(text)
    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )

    cleaned_chars: list[str] = []
    removed_control_chars = 0
    for char in normalized:
        if _should_strip_character(char):
            removed_control_chars += 1
            continue
        cleaned_chars.append(char)

    cleaned = "".join(cleaned_chars).strip()
    truncated = False
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
        truncated = True

    return SanitizedUserInput(
        text=cleaned,
        original_length=original_length,
        removed_control_chars=removed_control_chars,
        truncated=truncated,
    )


def build_user_input_notice(result: SanitizedUserInput) -> str:
    messages: list[str] = []
    if result.removed_control_chars:
        messages.append("Removed unsupported control characters before sending the request.")
    if result.truncated:
        messages.append(
            f"Input was truncated to {DEFAULT_USER_INPUT_LIMIT} characters before sending to the runtime."
        )
    return " ".join(messages)


def _should_strip_character(char: str) -> bool:
    if char in _ALLOWED_CONTROL_CHARS:
        return False
    return unicodedata.category(char).startswith("C")
