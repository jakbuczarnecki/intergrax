# # © Artur Czarnecki. All rights reserved.
# # Integrax framework – proprietary and confidential.
# # Use, modification, or distribution without written permission is prohibited.

# from __future__ import annotations
# from typing import Protocol, Sequence, Iterable, Optional, Any, Dict, Union, List

# # If it's only for types – remove runtime dependency (eliminates import cycles)
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from .conversational_memory import ChatMessage
# else:
#     ChatMessage = Any  # runtime stub

# import json
# import re

# __all__ = [
#     "LLMAdapter",
#     "OpenAIChatCompletionsAdapter",
#     "GeminiChatAdapter",
#     "LangChainOllamaAdapter",
#     "LLMAdapterRegistry",
# ]

# # ============================================================
# # Pydantic compat (v2/v1/fallback) – for structured output only
# # ============================================================
# try:
#     from pydantic import BaseModel  # type: ignore
#     _HAS_PYDANTIC = True
# except Exception:
#     class BaseModel: ...
#     _HAS_PYDANTIC = False


# def _strip_code_fences(text: str) -> str:
#     """
#     Removes wrappers like ```json ... ``` or ``` ... ``` if present.
#     """
#     if not text:
#         return text
#     # Remove known fence variants
#     fence_re = r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$"
#     m = re.match(fence_re, text, flags=re.DOTALL)
#     return m.group(1) if m else text


# def _extract_json_object(text: str) -> str:
#     """
#     Returns the first sensible {} as a JSON string (tolerantly trims noise).
#     When absent, returns an empty string.
#     """
#     if not text:
#         return ""
#     text = _strip_code_fences(text).strip()
#     start = text.find("{")
#     end = text.rfind("}")
#     if start == -1 or end == -1 or end <= start:
#         return ""
#     return text[start:end + 1]


# def _model_json_schema(model_cls: type) -> Dict[str, Any]:
#     """
#     Returns JSON Schema for the model class (Pydantic v2/v1). If unavailable, a minimal schema.
#     """
#     # pydantic v2
#     if hasattr(model_cls, "model_json_schema"):
#         try:
#             return model_cls.model_json_schema()  # type: ignore[attr-defined]
#         except Exception:
#             pass
#     # pydantic v1
#     if hasattr(model_cls, "schema"):
#         try:
#             return model_cls.schema()  # type: ignore[attr-defined]
#         except Exception:
#             pass
#     # fallback
#     return {"type": "object"}


# def _validate_with_model(model_cls: type, json_str: str):
#     """
#     Validates and creates a model instance from JSON.
#     Supports Pydantic v2, v1, and a fallback **dict constructor.
#     """
#     if not json_str or not json_str.strip():
#         raise ValueError("Empty JSON content for structured output.")
#     data = json.loads(json_str)

#     # pydantic v2
#     if hasattr(model_cls, "model_validate_json"):
#         try:
#             return model_cls.model_validate_json(json_str)  # type: ignore[attr-defined]
#         except Exception:
#             pass
#     if hasattr(model_cls, "model_validate"):
#         try:
#             return model_cls.model_validate(data)  # type: ignore[attr-defined]
#         except Exception:
#             pass

#     # pydantic v1
#     if hasattr(model_cls, "parse_raw"):
#         try:
#             return model_cls.parse_raw(json_str)  # type: ignore[attr-defined]
#         except Exception:
#             pass
#     if hasattr(model_cls, "parse_obj"):
#         try:
#             return model_cls.parse_obj(data)  # type: ignore[attr-defined]
#         except Exception:
#             pass

#     # fallback (plain class/dataclass with a compatible __init__)
#     try:
#         return model_cls(**data)
#     except Exception as e:
#         raise ValueError(f"Cannot validate structured output with {model_cls}: {e}")


# # ============================================================
# # Universal interface
# # ============================================================
# class LLMAdapter(Protocol):
#     def generate_messages(
#         self, messages: Sequence[ChatMessage], *, temperature: float = 0.2, max_tokens: Optional[int] = None
#     ) -> str: ...

#     def stream_messages(
#         self, messages: Sequence[ChatMessage], *, temperature: float = 0.2, max_tokens: Optional[int] = None
#     ) -> Iterable[str]: ...

#     # ---- Tools (optional) ----
#     def supports_tools(self) -> bool: ...
#     def generate_with_tools(
#         self,
#         messages: Sequence[ChatMessage],
#         tools_schema: List[Dict[str, Any]],
#         *,
#         temperature: float = 0.2,
#         max_tokens: Optional[int] = None,
#         tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
#     ) -> Dict[str, Any]: ...
#     def stream_with_tools(
#         self,
#         messages: Sequence[ChatMessage],
#         tools_schema: List[Dict[str, Any]],
#         *,
#         temperature: float = 0.2,
#         max_tokens: Optional[int] = None,
#         tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
#     ) -> Iterable[Dict[str, Any]]: ...

#     # ---- Structured output (optional) ----
#     def generate_structured(
#         self,
#         messages: Sequence[ChatMessage],
#         output_model: type,
#         *,
#         temperature: float = 0.2,
#         max_tokens: Optional[int] = None,
#     ): ...


# # ============================================================
# # Helper – convert ChatMessage → OpenAI schema
# # ============================================================
# def _map_messages_to_openai(msgs: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
#     out: List[Dict[str, Any]] = []
#     for m in msgs:
#         d: Dict[str, Any] = {"role": m.role, "content": m.content}
#         # Tool messages must have correct id/name fields
#         if m.role == "tool":
#             if getattr(m, "tool_call_id", None) is not None:
#                 d["tool_call_id"] = m.tool_call_id
#             if getattr(m, "name", None) is not None:
#                 d["name"] = m.name
#         if getattr(m, "tool_calls", None):
#             d["tool_calls"] = m.tool_calls
#         out.append(d)
#     return out


# # ============================================================
# # OpenAI Chat Completions
# # ============================================================
# class OpenAIChatCompletionsAdapter:
#     """
#     client = openai.OpenAI()
#     OpenAIChatAdapter(client, model="gpt-4o-mini")
#     """

#     def __init__(self, client, model: str, **defaults):
#         self.client = client
#         self.model = model
#         self.defaults = defaults

#     # --- plain chat ---
#     def generate_messages(self, messages, *, temperature=0.2, max_tokens=None) -> str:
#         payload = dict(model=self.model, messages=_map_messages_to_openai(messages), temperature=temperature)
#         if max_tokens is not None:
#             payload["max_tokens"] = max_tokens
#         res = self.client.chat.completions.create(**payload, **self.defaults)
#         return res.choices[0].message.content or ""

#     def stream_messages(self, messages, *, temperature=0.2, max_tokens=None) -> Iterable[str]:
#         payload = dict(model=self.model, messages=_map_messages_to_openai(messages), temperature=temperature, stream=True)
#         if max_tokens is not None:
#             payload["max_tokens"] = max_tokens
#         stream = self.client.chat.completions.create(**payload, **self.defaults)
#         for ev in stream:
#             delta = getattr(ev.choices[0].delta, "content", None)
#             if delta:
#                 yield delta

#     # --- tools ---
#     def supports_tools(self) -> bool:
#         return True  # signal for the agent to use the native tools path

#     def generate_with_tools(
#         self, messages, tools_schema, *, temperature=0.2, max_tokens=None, tool_choice=None
#     ) -> Dict[str, Any]:
#         payload = dict(
#             model=self.model,
#             messages=_map_messages_to_openai(messages),
#             temperature=temperature,
#             tools=tools_schema,
#         )
#         if tool_choice is not None:
#             payload["tool_choice"] = tool_choice
#         if max_tokens is not None:
#             payload["max_tokens"] = max_tokens

#         res = self.client.chat.completions.create(**payload, **self.defaults)
#         choice = res.choices[0]
#         msg = choice.message
#         tool_calls = getattr(msg, "tool_calls", None) or []

#         # We RETURN tool_calls in the native OpenAI shape (with 'type' and 'function')
#         native_tool_calls = []
#         for tc in tool_calls:
#             native_tool_calls.append({
#                 "id": getattr(tc, "id", None),
#                 "type": "function",
#                 "function": {
#                     "name": getattr(tc.function, "name", None),
#                     "arguments": getattr(tc.function, "arguments", "{}"),
#                 },
#             })

#         return {
#             "content": msg.content or "",
#             "tool_calls": native_tool_calls,
#             "finish_reason": choice.finish_reason,
#         }

#     def stream_with_tools(self, *args, **kwargs):
#         # Streaming tool_calls is rarely used → fallback to single call
#         yield self.generate_with_tools(*args, **kwargs)

#     # --- structured output (new) ---
#     def generate_structured(
#         self,
#         messages,
#         output_model: type,
#         *,
#         temperature: float = 0.2,
#         max_tokens: Optional[int] = None,
#     ):
#         """
#         Returns a model *instance* conforming to the given schema (supports Pydantic v2/v1).
#         Enforces pure JSON via response_format=json_object and appends a system description with the schema.
#         """
#         schema = _model_json_schema(output_model)

#         sys_extra = {
#             "role": "system",
#             "content": (
#                 "Return ONLY a single JSON object that strictly conforms to this JSON Schema. "
#                 "No prose, no markdown, no backticks. If a field is optional and unknown, omit it.\n"
#                 f"JSON_SCHEMA: {json.dumps(schema, ensure_ascii=False)}"
#             ),
#         }
#         mapped = _map_messages_to_openai(messages)
#         mapped = [sys_extra] + mapped

#         payload = dict(
#             model=self.model,
#             messages=mapped,
#             temperature=temperature,
#             response_format={"type": "json_object"},
#         )
#         if max_tokens is not None:
#             payload["max_tokens"] = max_tokens

#         res = self.client.chat.completions.create(**payload, **self.defaults)
#         txt = res.choices[0].message.content or ""
#         json_str = _extract_json_object(txt) or txt.strip()
#         if not json_str:
#             raise ValueError("Model did not return JSON content for structured output.")

#         return _validate_with_model(output_model, json_str)


# # ============================================================
# # OpenAI Chat Responses
# # ============================================================
# class OpenAIChatResponsesAdapter:
#     """
#     Drop-in replacement for the previous Chat Completions-based adapter,
#     now using the new OpenAI Responses API under the hood.

#     Public interface is preserved:
#     - generate_messages
#     - stream_messages
#     - generate_with_tools
#     - stream_with_tools
#     - generate_structured

#     No calling code changes required.
#     """

#     def __init__(self, client, model: str, **defaults):
#         self.client = client
#         self.model = model
#         self.defaults = defaults

#     # ---------------------------------------------------------------------
#     # INTERNAL HELPERS (PRIVATE METHODS)
#     # ---------------------------------------------------------------------

#     def _messages_to_responses_input(self, mapped_messages):
#         """
#         Convert legacy Chat Completion style messages:
#             { "role": "user", "content": "Hello" }

#         Into the Responses API "input items" format:
#             { "type": "message", "role": "user", "content": "Hello" }

#         Required because Responses API expects explicit typed input blocks.
#         """
#         items = []
#         for m in mapped_messages:
#             items.append({
#                 "type": "message",
#                 "role": m.get("role", "user"),
#                 "content": m.get("content", ""),
#             })
#         return items

#     def _collect_output_text(self, response):
#         """
#         Extract the final assistant output text from a Responses response object.

#         The SDK provides response.output_text, but not all models return it.
#         If it's missing, we manually aggregate text from:
#             response.output[*].content[*] where type == "output_text".

#         This ensures consistent return behavior across all models.
#         """
#         # Fast path: preferred attribute
#         txt = getattr(response, "output_text", None)
#         if txt:
#             return txt

#         # Fallback: walk through the structured response
#         chunks = []
#         for item in getattr(response, "output", []) or []:
#             if getattr(item, "type", None) == "message":
#                 for c in getattr(item, "content", []) or []:
#                     if getattr(c, "type", None) == "output_text":
#                         chunks.append(getattr(c, "text", "") or "")

#         return "".join(chunks)

#     # ---------------------------------------------------------------------
#     # PUBLIC: Plain chat
#     # ---------------------------------------------------------------------

#     def generate_messages(self, messages, *, temperature=0.2, max_tokens=None) -> str:
#         """
#         Standard single-shot assistant response (non-streaming).
#         Equivalent to the old chat.completions.create(...) adapter behavior.
#         """
#         mapped = _map_messages_to_openai(messages)
#         input_items = self._messages_to_responses_input(mapped)

#         payload = dict(
#             model=self.model,
#             input=input_items,
#             temperature=temperature,
#         )

#         # Responses API uses `max_output_tokens`, not `max_tokens`
#         if max_tokens is not None:
#             payload["max_output_tokens"] = max_tokens

#         response = self.client.responses.create(**payload, **self.defaults)
#         return self._collect_output_text(response)

#     def stream_messages(self, messages, *, temperature=0.2, max_tokens=None):
#         """
#         Streaming response generator.

#         Yields incremental text chunks extracted from the streaming
#         Responses API delta events.
#         """
#         mapped = _map_messages_to_openai(messages)
#         input_items = self._messages_to_responses_input(mapped)

#         payload = dict(
#             model=self.model,
#             input=input_items,
#             temperature=temperature,
#             stream=True,
#         )

#         if max_tokens is not None:
#             payload["max_output_tokens"] = max_tokens

#         stream = self.client.responses.create(**payload, **self.defaults)

#         for ev in stream:
#             # Relevant event type: "response.output_text.delta"
#             if getattr(ev, "type", None) == "response.output_text.delta":
#                 delta = getattr(ev, "delta", None)
#                 if delta:
#                     yield delta

#     # ---------------------------------------------------------------------
#     # PUBLIC: Tools
#     # ---------------------------------------------------------------------

#     def supports_tools(self) -> bool:
#         """
#         Signal to higher-level agent framework code:
#         this adapter supports calling functions / tools natively.
#         """
#         return True

#     def generate_with_tools(self, messages, tools_schema, *, temperature=0.2, max_tokens=None, tool_choice=None):
#         """
#         Equivalent to generate_messages, but enabling tool call extraction.

#         The returned structure preserves backward compatibility:
#         - `content`
#         - `tool_calls[]`
#         - `finish_reason`
#         """
#         mapped = _map_messages_to_openai(messages)
#         input_items = self._messages_to_responses_input(mapped)

#         payload = dict(
#             model=self.model,
#             input=input_items,
#             temperature=temperature,
#             tools=tools_schema,
#         )

#         if tool_choice is not None:
#             payload["tool_choice"] = tool_choice

#         if max_tokens is not None:
#             payload["max_output_tokens"] = max_tokens

#         response = self.client.responses.create(**payload, **self.defaults)

#         # Extract assistant text (if any)
#         content = self._collect_output_text(response)

#         # Extract tool calls
#         native_tool_calls = []
#         for item in getattr(response, "output", []) or []:
#             if getattr(item, "type", None) == "function_call":
#                 # Normalize tool call arguments to JSON string
#                 args = getattr(item, "arguments", "{}")
#                 if not isinstance(args, str):
#                     args = json.dumps(args, ensure_ascii=False)

#                 native_tool_calls.append({
#                     "id": getattr(item, "call_id", None),
#                     "type": "function",
#                     "function": {
#                         "name": getattr(item, "name", None),
#                         "arguments": args,
#                     },
#                 })

#         # Responses API doesn't expose a direct finish_reason field;
#         # we approximate via status for compatibility.
#         finish_reason = getattr(response, "status", None) or "completed"

#         return {
#             "content": content or "",
#             "tool_calls": native_tool_calls,
#             "finish_reason": finish_reason,
#         }

#     def stream_with_tools(self, *args, **kwargs):
#         """
#         Streaming tool-calls is rarely useful.
#         For now we fallback to generate_with_tools for compatibility.
#         """
#         yield self.generate_with_tools(*args, **kwargs)

#     # ---------------------------------------------------------------------
#     # PUBLIC: Structured JSON output
#     # ---------------------------------------------------------------------

#     def generate_structured(self, messages, output_model, *, temperature=0.2, max_tokens=None):
#         """
#         Structured output wrapper using Responses API + JSON Schema validation.

#         The API surface matches the previous implementation and returns:
#             validated_instance = output_model(...)
#         """
#         schema = _model_json_schema(output_model)

#         # Prepend schema instruction as a system message
#         sys_extra = {
#             "role": "system",
#             "content": (
#                 "Return ONLY a single JSON object that strictly conforms to this JSON Schema. "
#                 "No prose, no markdown, no backticks. If a field is optional and unknown, omit it.\n"
#                 f"JSON_SCHEMA: {json.dumps(schema, ensure_ascii=False)}"
#             )
#         }

#         mapped = [_map_messages_to_openai(messages)[0]] if messages else []
#         mapped = [sys_extra] + _map_messages_to_openai(messages)

#         input_items = self._messages_to_responses_input(mapped)

#         payload = dict(
#             model=self.model,
#             input=input_items,
#             temperature=temperature,
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": {
#                     "name": getattr(output_model, "__name__", "OutputModel"),
#                     "schema": schema,
#                     "strict": True,
#                 },
#             },
#         )

#         if max_tokens is not None:
#             payload["max_output_tokens"] = max_tokens

#         response = self.client.responses.create(**payload, **self.defaults)

#         txt = self._collect_output_text(response)
#         json_str = _extract_json_object(txt) or txt.strip()
#         if not json_str:
#             raise ValueError("Model did not return valid JSON.")

#         return _validate_with_model(output_model, json_str)


# # ============================================================
# # Gemini (skeleton)
# # ============================================================
# class GeminiChatAdapter:
#     """
#     Skeleton - depends on the Google SDK version. Versions from 2025 support function calling.
#     """
#     def __init__(self, model, **defaults):
#         self.model = model
#         self.defaults = defaults

#     def _split_system(self, messages: Sequence[ChatMessage]):
#         sys_txt = "\n".join(m.content for m in messages if m.role == "system").strip() or None
#         convo = [m for m in messages if m.role != "system"]
#         return sys_txt, convo

#     def generate_messages(self, messages, *, temperature=0.2, max_tokens=None) -> str:
#         sys_txt, convo = self._split_system(messages)
#         chat = self.model.start_chat(history=[{"role": m.role, "parts": [m.content]} for m in convo[:-1]])
#         user_last = convo[-1].content if convo else ""
#         kwargs = {"temperature": temperature, **self.defaults}
#         if max_tokens is not None:
#             kwargs["max_output_tokens"] = max_tokens
#         if sys_txt:
#             user_last = f"[SYSTEM]\n{sys_txt}\n\n[USER]\n{user_last}"
#         res = chat.send_message(user_last, **kwargs)
#         return res.text or ""

#     def stream_messages(self, messages, *, temperature=0.2, max_tokens=None) -> Iterable[str]:
#         yield self.generate_messages(messages, temperature=temperature, max_tokens=max_tokens)

#     def supports_tools(self) -> bool: return False
#     def generate_with_tools(self, *a, **k): raise NotImplementedError("Gemini tools not wired here")
#     def stream_with_tools(self, *a, **k): raise NotImplementedError("Gemini tools not wired here")

#     # (optionally implement generate_structured similar to the Ollama path)


# # ============================================================
# # LangChain + Ollama (no native tools → planner JSON)
# # ============================================================
# class LangChainOllamaAdapter:
#     def __init__(self, chat, **defaults):
#         self.chat = chat
#         self.defaults = defaults

#     def _to_lc_messages(self, messages):
#         from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
#         out = []
#         for m in messages:
#             if m.role == "system":
#                 out.append(SystemMessage(content=m.content))
#             elif m.role == "user":
#                 out.append(HumanMessage(content=m.content))
#             elif m.role == "assistant":
#                 out.append(AIMessage(content=m.content))
#             elif m.role == "tool":
#                 # no native tools → inject tool result as context
#                 out.append(SystemMessage(content=f"[TOOL RESULT]\n{m.content}"))
#             else:
#                 out.append(SystemMessage(content=f"[{m.role.upper()}]\n{m.content}"))
#         return out

#     @staticmethod
#     def _with_ollama_options(base_kwargs: Dict[str, Any], *, temperature: float=None, max_tokens: Optional[int]=None):
#         """
#         Ollama (via langchain_ollama) expects generation parameters inside `options`.
#         Mapping:
#           - temperature -> options["temperature"]
#           - max_tokens  -> options["num_predict"]
#         """
#         kwargs = dict(base_kwargs or {})
#         opts = dict(kwargs.get("options") or {})
#         if temperature is not None:
#             opts["temperature"] = temperature
#         if max_tokens is not None:
#             opts["num_predict"] = max_tokens
#         kwargs["options"] = opts
#         return kwargs

#     def generate_messages(self, messages, *, temperature=None, max_tokens=None):
#         lc_msgs = self._to_lc_messages(messages)
#         kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
#         res = self.chat.invoke(lc_msgs, **kwargs)
#         return getattr(res, "content", None) or str(res)

#     def stream_messages(self, messages, *, temperature=0.2, max_tokens=None):
#         lc_msgs = self._to_lc_messages(messages)
#         kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
#         try:
#             for chunk in self.chat.stream(lc_msgs, **kwargs):
#                 c = getattr(chunk, "content", None)
#                 if c:
#                     yield c
#         except Exception:
#             # fallback to single call
#             yield self.generate_messages(messages, temperature=temperature, max_tokens=max_tokens)

#     def supports_tools(self) -> bool:
#         return False  # key point – the agent will take the “planner” branch

#     # --- structured output (new) ---
#     def generate_structured(
#         self,
#         messages,
#         output_model: type,
#         *,
#         temperature: float = 0.2,
#         max_tokens: Optional[int] = None,
#     ):
#         """
#         Enforces returning a single JSON object conforming to the schema (strict JSON prompt + validation).
#         """
#         schema = _model_json_schema(output_model)

#         from langchain_core.messages import SystemMessage, HumanMessage
#         lc_msgs = self._to_lc_messages(messages)

#         strict = SystemMessage(
#             content=(
#                 "Return ONLY a single JSON object that strictly conforms to the JSON Schema below. "
#                 "Do not add any commentary, markdown, or backticks. "
#                 "If a field is optional and unknown, omit it."
#             )
#         )
#         schema_msg = HumanMessage(content=f"JSON_SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}")
#         lc_msgs = [strict, schema_msg] + lc_msgs

#         kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
#         res = self.chat.invoke(lc_msgs, **kwargs)
#         txt = getattr(res, "content", None) or str(res)
#         json_str = _extract_json_object(txt) or txt.strip()
#         if not json_str:
#             raise ValueError("Model did not return JSON content for structured output (Ollama).")

#         return _validate_with_model(output_model, json_str)


# # ============================================================
# # Adapter registry
# # ============================================================
# class LLMAdapterRegistry:
#     _registry = {}

#     @classmethod
#     def register(cls, name: str, factory):
#         cls._registry[name.lower()] = factory

#     @classmethod
#     def create(cls, name: str, **kwargs) -> LLMAdapter:
#         key = name.lower()
#         if key not in cls._registry:
#             raise ValueError(f"Unknown adapter: {name}")
#         return cls._registry[key](**kwargs)

# # Default adapter registrations
# LLMAdapterRegistry.register("openai", lambda **kw: OpenAIChatCompletionsAdapter(**kw))
# LLMAdapterRegistry.register("gemini", lambda **kw: GeminiChatAdapter(**kw))
# LLMAdapterRegistry.register("ollama", lambda **kw: LangChainOllamaAdapter(**kw))
