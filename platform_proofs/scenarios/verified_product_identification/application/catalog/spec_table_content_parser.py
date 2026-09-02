"""Conservative deterministic parsing for WDC ``specTableContent``."""

from __future__ import annotations

from dataclasses import dataclass

_MAX_SPEC_ATTRIBUTE_KEY_LENGTH = 120
_VERTICAL_TAB = "\x0b"


@dataclass(frozen=True, slots=True)
class ParsedSpecAttribute:
    """One attribute pair extracted from unambiguous ``specTableContent``."""

    source_key: str
    source_value: str


def parse_spec_table_content(text: str) -> tuple[ParsedSpecAttribute, ...]:
    """Extract attribute pairs only when the source format is unambiguous."""
    stripped = text.strip()
    if not stripped:
        return ()

    vertical_tab_pairs = _try_vertical_tab_colon_pairs(stripped)
    if vertical_tab_pairs is not None:
        return vertical_tab_pairs

    newline_pairs = _try_newline_colon_key_value(stripped)
    if newline_pairs is not None:
        return newline_pairs

    return ()


def _try_vertical_tab_colon_pairs(text: str) -> tuple[ParsedSpecAttribute, ...] | None:
    if _VERTICAL_TAB not in text:
        return None

    parts = [part.strip() for part in text.split(_VERTICAL_TAB) if part.strip()]
    if len(parts) < 4:
        return None

    pairs: list[ParsedSpecAttribute] = []
    index = 0
    while index < len(parts):
        segment = parts[index]
        if ":" not in segment:
            return None

        source_key, _, inline_value = segment.partition(":")
        source_key = source_key.strip()
        inline_value = inline_value.strip()
        if not _is_valid_spec_attribute_key(source_key):
            return None

        if inline_value:
            pairs.append(
                ParsedSpecAttribute(
                    source_key=source_key,
                    source_value=inline_value,
                )
            )
            index += 1
            continue

        if index + 1 >= len(parts):
            return None
        source_value = parts[index + 1].strip()
        if not source_value:
            return None
        pairs.append(
            ParsedSpecAttribute(
                source_key=source_key,
                source_value=source_value,
            )
        )
        index += 2

    if len(pairs) < 2:
        return None
    return tuple(pairs)


def _try_newline_colon_key_value(text: str) -> tuple[ParsedSpecAttribute, ...] | None:
    if _VERTICAL_TAB in text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    pairs: list[ParsedSpecAttribute] = []
    for line in lines:
        if line.count(":") != 1:
            return None
        source_key, _, source_value = line.partition(":")
        source_key = source_key.strip()
        source_value = source_value.strip()
        if not _is_valid_spec_attribute_key(source_key) or not source_value:
            return None
        pairs.append(
            ParsedSpecAttribute(
                source_key=source_key,
                source_value=source_value,
            )
        )
    return tuple(pairs)


def _is_valid_spec_attribute_key(source_key: str) -> bool:
    return bool(source_key) and len(source_key) <= _MAX_SPEC_ATTRIBUTE_KEY_LENGTH
