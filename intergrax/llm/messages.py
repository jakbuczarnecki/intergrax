# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import uuid

from intergrax.utils.time_provider import SystemTimeProvider

MessageRole = Literal["system", "user", "assistant", "tool"]

MODEL_INPUT_MESSAGES_METADATA_KEY = "model_input_messages.v1"
MODEL_INPUT_MESSAGES_SCHEMA_VERSION = "model_input_messages.v1"
STRUCTURED_MODEL_INPUT_REQUIRED_REASON = (
    "structured_model_input_requires_message_capable_consumer"
)

_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


class StructuredModelInputRequiredError(RuntimeError):
    reason = STRUCTURED_MODEL_INPUT_REQUIRED_REASON

    def __init__(self) -> None:
        super().__init__(self.reason)


@dataclass
class AttachmentRef:
    """
    Lightweight reference to an attachment associated with a message or session.

    The actual binary content is stored elsewhere (filesystem, object storage,
    database BLOB, etc.). Here we only keep stable identifiers and metadata.
    """

    id: str
    type: str  # e.g. "pdf", "docx", "image", "audio", "video", "code", "json"
    uri: str   # e.g. "file://...", "s3://...", "db://attachments/<id>"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# ChatMessage – extended with tool_calls and to_dict() method
# ============================================================
@dataclass
class ChatMessage:
    """
    Universal chat message compatible with the OpenAI Responses API.
    Supports fields:
      - tool_call_id  → for single tool calls (from field 'id'),
      - tool_calls    → list of calls (for assistant.tool_calls).
    """

    role: MessageRole
    content: str    
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    deleted: bool = False
    modified: bool = False    
    created_at: str = field(default_factory=lambda: SystemTimeProvider.utc_now().isoformat())
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    attachments: List[AttachmentRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Converts the object to a dict compatible with OpenAI Responses API / ChatCompletions.
        """
        msg = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    def __repr__(self):
        extras = []
        if self.name:
            extras.append(f"name={self.name}")
        if self.tool_call_id:
            extras.append(f"tool_call_id={self.tool_call_id}")
        if self.tool_calls:
            extras.append(f"tool_calls={len(self.tool_calls)}")
        extras_str = ", ".join(extras)
        return f"<ChatMessage role={self.role} {extras_str}>"


def compute_model_facing_messages_hash(messages: Sequence[ChatMessage]) -> str:
    """SHA-256 over canonical model-facing message sequence (``to_dict()`` payloads)."""
    payload = [message.to_dict() for message in messages]
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256()
    digest.update(canonical_json.encode("utf-8"))
    return digest.hexdigest()


def _json_safe_value(value: object) -> object:
  """Validate JSON serializability without ``default=str``."""
  if value is None:
    return None
  if isinstance(value, bool):
    return value
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    if not math.isfinite(value):
      raise ValueError("non-finite float in tool_calls")
    return value
  if isinstance(value, str):
    return value
  if isinstance(value, list):
    return [_json_safe_value(item) for item in value]
  if isinstance(value, dict):
    result: dict[str, object] = {}
    for key, item in value.items():
      if type(key) is not str:
        raise ValueError("non-string key in tool_calls")
      result[key] = _json_safe_value(item)
    return result
  raise ValueError("non-json-safe value in tool_calls")


def _json_safe_tool_calls(tool_calls: Optional[List[dict]]) -> Optional[List[dict]]:
  if tool_calls is None:
    return None
  safe = [_json_safe_value(dict(item)) for item in tool_calls]
  json.dumps(safe)
  return safe


def _message_to_envelope_row(message: ChatMessage) -> dict[str, object]:
  return {
    "entry_id": message.entry_id,
    "role": message.role,
    "content": message.content,
    "name": message.name,
    "tool_call_id": message.tool_call_id,
    "tool_calls": _json_safe_tool_calls(message.tool_calls),
  }


def build_model_input_messages_envelope(
  messages: Sequence[ChatMessage],
) -> dict[str, object]:
  rows = [_message_to_envelope_row(message) for message in messages]
  return {
    "schema_version": MODEL_INPUT_MESSAGES_SCHEMA_VERSION,
    "messages": rows,
    "messages_hash": compute_model_facing_messages_hash(messages),
  }


def _chat_message_from_envelope_row(row: dict[str, object]) -> ChatMessage:
  role = row.get("role")
  if role not in _ALLOWED_MESSAGE_ROLES:
    raise ValueError("invalid model input message role")
  content = row.get("content")
  if type(content) is not str:
    raise ValueError("invalid model input message content")
  entry_id = row.get("entry_id")
  if type(entry_id) is not str or not entry_id.strip():
    raise ValueError("invalid model input message entry_id")
  name = row.get("name")
  if name is not None and (type(name) is not str or not name.strip()):
    raise ValueError("invalid model input message name")
  tool_call_id = row.get("tool_call_id")
  if tool_call_id is not None and (type(tool_call_id) is not str or not tool_call_id.strip()):
    raise ValueError("invalid model input message tool_call_id")
  raw_tool_calls = row.get("tool_calls")
  tool_calls: Optional[List[dict]] = None
  if raw_tool_calls is not None:
    if type(raw_tool_calls) is not list:
      raise ValueError("invalid model input message tool_calls")
    tool_calls = []
    for item in raw_tool_calls:
      if not isinstance(item, dict):
        raise ValueError("invalid model input message tool_calls row")
      tool_calls.append(dict(item))
    tool_calls = _json_safe_tool_calls(tool_calls)
  return ChatMessage(
    role=role,
    content=content,
    entry_id=entry_id,
    name=name,
    tool_call_id=tool_call_id,
    tool_calls=tool_calls,
  )


def model_input_messages_from_envelope(
  raw: object,
) -> tuple[ChatMessage, ...]:
  if not isinstance(raw, dict):
    raise ValueError("model input envelope must be a dict")
  schema_version = raw.get("schema_version")
  if schema_version != MODEL_INPUT_MESSAGES_SCHEMA_VERSION:
    raise ValueError("invalid model input envelope schema_version")
  raw_messages = raw.get("messages")
  if type(raw_messages) is not list:
    raise ValueError("model input envelope messages must be a list")
  stored_hash = raw.get("messages_hash")
  if type(stored_hash) is not str or not stored_hash.strip():
    raise ValueError("invalid model input envelope messages_hash")
  seen_entry_ids: set[str] = set()
  messages: list[ChatMessage] = []
  for item in raw_messages:
    if not isinstance(item, dict):
      raise ValueError("model input envelope message row must be a dict")
    message = _chat_message_from_envelope_row(item)
    if message.entry_id in seen_entry_ids:
      raise ValueError("duplicate model input message entry_id")
    seen_entry_ids.add(message.entry_id)
    messages.append(message)
  computed_hash = compute_model_facing_messages_hash(messages)
  if computed_hash != stored_hash:
    raise ValueError("model input messages hash mismatch")
  return tuple(messages)


def model_input_messages_from_metadata(
  metadata: Mapping[str, object],
) -> tuple[ChatMessage, ...]:
  if MODEL_INPUT_MESSAGES_METADATA_KEY not in metadata:
    return ()
  return model_input_messages_from_envelope(metadata[MODEL_INPUT_MESSAGES_METADATA_KEY])


def replace_final_user_message(
  messages: Sequence[ChatMessage],
  prompt: str,
) -> tuple[ChatMessage, ...]:
  if type(prompt) is not str:
    raise ValueError("prompt must be a str")
  if not messages:
    raise ValueError("messages must not be empty")
  final_message = messages[-1]
  if final_message.role != "user":
    raise ValueError("final message must be user")
  updated_final = ChatMessage(
    role=final_message.role,
    content=prompt,
    entry_id=final_message.entry_id,
    name=final_message.name,
    tool_call_id=final_message.tool_call_id,
    tool_calls=list(final_message.tool_calls) if final_message.tool_calls is not None else None,
  )
  return tuple(list(messages[:-1]) + [updated_final])


def requires_structured_model_input(
  messages: Sequence[ChatMessage],
) -> bool:
  if not messages:
    return False
  for index, message in enumerate(messages):
    is_final = index == len(messages) - 1
    if message.role == "system":
      content = message.content or ""
      if not content.startswith("[context:"):
        return True
      if message.name is not None or message.tool_call_id is not None or message.tool_calls:
        return True
      continue
    if is_final and message.role == "user":
      if message.name is not None or message.tool_call_id is not None or message.tool_calls:
        return True
      continue
    return True
  return False


def final_user_message_content(
  messages: Sequence[ChatMessage],
) -> str:
  if not messages:
    raise ValueError("messages must not be empty")
  final_message = messages[-1]
  if final_message.role != "user":
    raise ValueError("final message must be user")
  return final_message.content


def append_chat_messages(
        existing: Optional[List[ChatMessage]],
        new: List[ChatMessage]
) -> List[ChatMessage]:
    """
    Append-only reducer for message lists (optional LangGraph adapter compatibility).

    When used as a LangGraph state reducer, merging works as:
      - `existing`: the current list of messages in the state (may be None)
      - `new`: the list of messages provided by a node update

    We simply append the new messages to the existing ones.
    """
    if existing is None:
        return list(new)
    return [*existing, *new]
