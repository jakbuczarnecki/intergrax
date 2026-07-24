# © Artur Czarnecki. All rights reserved.

"""Declarative Slack command metadata, discovery, registry, and help rendering."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAddress,
    InboundConversationEvent,
)
from local_workspace_application.slack_companion.models import (
    AuthorizedSlackAskContext,
    SlackDedupeRecord,
)

_COMMAND_ATTR = "__lkw_slack_command__"

SlackCommandParser = Callable[[str], "SlackCommandMatch | None"]
SlackCommandHandler = Callable[
    ["SlackCommandContext", "SlackCommandMatch"],
    Awaitable[None],
]


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 for ch in value)


def _require_nonempty_public_text(field: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"SlackCommandMetadata.{field} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"SlackCommandMetadata.{field} must be non-empty")
    if _has_control_chars(normalized):
        raise ValueError(f"SlackCommandMetadata.{field} must not contain control characters")
    return normalized


@dataclass(frozen=True, slots=True)
class SlackCommandMetadata:
    """Immutable public metadata for one formal Slack command."""

    command_id: str
    syntax: str
    description: str
    example: str
    priority: int
    visible_in_help: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError("SlackCommandMetadata.priority must be int")
        if not isinstance(self.visible_in_help, bool):
            raise TypeError("SlackCommandMetadata.visible_in_help must be bool")

        command_id = _require_nonempty_public_text("command_id", self.command_id)
        syntax = _require_nonempty_public_text("syntax", self.syntax)
        description = _require_nonempty_public_text("description", self.description)

        if self.visible_in_help:
            example = _require_nonempty_public_text("example", self.example)
        else:
            if not isinstance(self.example, str):
                raise TypeError("SlackCommandMetadata.example must be str")
            example = self.example.strip()
            if example and _has_control_chars(example):
                raise ValueError(
                    "SlackCommandMetadata.example must not contain control characters"
                )

        object.__setattr__(self, "command_id", command_id)
        object.__setattr__(self, "syntax", syntax)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "example", example)


@dataclass(frozen=True, slots=True)
class SlackCommandMatch:
    """Successful parse of a formal command; payload may be None for no-arg commands."""

    payload: object | None = None


@dataclass(frozen=True, slots=True)
class SlackCommandContext:
    """Minimal product context for a formal Slack command handler."""

    event: InboundConversationEvent
    address: ConversationAddress
    authorized: AuthorizedSlackAskContext
    claim: SlackDedupeRecord
    actor_key: str


@dataclass(frozen=True, slots=True)
class _SlackCommandAnnotation:
    metadata: SlackCommandMetadata
    parser: SlackCommandParser


@dataclass(frozen=True, slots=True)
class SlackCommandDefinition:
    """One discovered command: metadata + parser + bound async handler."""

    metadata: SlackCommandMetadata
    parser: SlackCommandParser
    handler: SlackCommandHandler


@dataclass(frozen=True, slots=True)
class SlackResolvedCommand:
    """First matching command for a message (handler not yet invoked)."""

    command_id: str
    match: SlackCommandMatch
    handler: SlackCommandHandler
    definition: SlackCommandDefinition


def slack_command(
    *,
    command_id: str,
    syntax: str,
    description: str,
    example: str,
    priority: int,
    parser: SlackCommandParser,
    visible_in_help: bool = True,
) -> Callable[[Any], Any]:
    """
    Attach immutable command metadata + parser to a method.

    Does not register globally; discovery binds methods from a concrete owner.
    """
    if not callable(parser):
        raise TypeError("slack_command parser must be callable")
    metadata = SlackCommandMetadata(
        command_id=command_id,
        syntax=syntax,
        description=description,
        example=example,
        priority=priority,
        visible_in_help=visible_in_help,
    )
    annotation = _SlackCommandAnnotation(metadata=metadata, parser=parser)

    def decorator(fn: Any) -> Any:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"slack_command handler {getattr(fn, '__name__', fn)!r} must be async"
            )
        setattr(fn, _COMMAND_ATTR, annotation)
        return fn

    return decorator


class SlackCommandRegistry:
    """Immutable ordered registry of formal Slack commands."""

    def __init__(self, definitions: Sequence[SlackCommandDefinition]) -> None:
        ordered = tuple(
            sorted(
                definitions,
                key=lambda d: (d.metadata.priority, d.metadata.command_id),
            )
        )
        seen: set[str] = set()
        for definition in ordered:
            command_id = definition.metadata.command_id
            if command_id in seen:
                raise ValueError(f"duplicate Slack command_id: {command_id}")
            seen.add(command_id)
            if definition.parser is None or not callable(definition.parser):
                raise ValueError(f"missing parser for command_id={command_id}")
            if not inspect.iscoroutinefunction(definition.handler):
                raise TypeError(
                    f"handler for command_id={command_id} must be async"
                )
        self._definitions = ordered

    @property
    def definitions(self) -> tuple[SlackCommandDefinition, ...]:
        return self._definitions

    def match(self, text: str) -> SlackResolvedCommand | None:
        for definition in self._definitions:
            matched = definition.parser(text)
            if matched is None:
                continue
            if not isinstance(matched, SlackCommandMatch):
                raise TypeError(
                    f"parser for {definition.metadata.command_id} must return "
                    "SlackCommandMatch | None"
                )
            return SlackResolvedCommand(
                command_id=definition.metadata.command_id,
                match=matched,
                handler=definition.handler,
                definition=definition,
            )
        return None

    def visible_commands(self) -> tuple[SlackCommandDefinition, ...]:
        return tuple(
            definition
            for definition in self._definitions
            if definition.metadata.visible_in_help
        )


def discover_slack_commands(handler_owner: object) -> SlackCommandRegistry:
    """
    Discover commands from annotated methods on ``handler_owner`` only.

    Does not scan modules, packages, or HTTP endpoints.
    """
    collected: list[SlackCommandDefinition] = []
    seen_names: set[str] = set()

    for cls in type(handler_owner).__mro__:
        for name, attr in vars(cls).items():
            if name in seen_names:
                continue
            if not callable(attr):
                continue
            annotation = getattr(attr, _COMMAND_ATTR, None)
            if annotation is None:
                continue
            if not isinstance(annotation, _SlackCommandAnnotation):
                raise TypeError(
                    f"invalid { _COMMAND_ATTR } on {name!r}"
                )
            bound = getattr(handler_owner, name)
            collected.append(
                SlackCommandDefinition(
                    metadata=annotation.metadata,
                    parser=annotation.parser,
                    handler=bound,
                )
            )
            seen_names.add(name)

    return SlackCommandRegistry(collected)


def render_command_help(definitions: Sequence[SlackCommandDefinition]) -> str:
    """Render plain-text help solely from registry metadata (no hardcoded command names)."""
    lines = ["Available commands:", ""]
    for definition in definitions:
        meta = definition.metadata
        if not meta.visible_in_help:
            continue
        lines.append(f"`{meta.syntax}`")
        lines.append(meta.description)
        lines.append(f"Example: `{meta.example}`")
        lines.append("")
    lines.append(
        "You can also send a normal question to search the active workspace."
    )
    return "\n".join(lines).rstrip() + "\n"
