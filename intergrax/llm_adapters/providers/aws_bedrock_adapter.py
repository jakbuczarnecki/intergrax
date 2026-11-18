# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, cast

import boto3
from botocore.config import Config

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters._shared.anthropic_messages import map_anthropic_messages
from intergrax.llm_adapters._shared.bedrock_converse import (
    build_converse_request,
    converse_stream_supported,
    converse_supported,
    extract_converse_text,
    extract_converse_tool_calls,
    iter_converse_stream_text,
    parse_converse_stream_tool_event,
)
from intergrax.llm_adapters._shared.messages import split_system_messages
from intergrax.llm_adapters._shared.openai_tool_call_interop import tool_calls_from_openai_dicts
from intergrax.llm_adapters._shared.stream_finish_signal import (
    StreamFinishSignal,
    converse_stop_signal,
    finish_reason_from_raw,
    invoke_stop_signal,
)
from intergrax.llm_adapters._shared.tool_schema import (
    openai_tools_to_anthropic,
    openai_tools_to_bedrock_converse,
)
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.capabilities import LLMFeature
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.native_tool_choice import NativeToolChoice
from intergrax.llm_adapters.contracts.provider_error import (
    LLMProviderErrorCategory,
    LLMProviderErrorSignal,
    category_from_status,
)
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.providers._per_call_client import (
    CallClient,
    call_on_client,
    stream_on_client,
)
from intergrax.llm_adapters.registry.context_window import (
    init_adapter_context_window_tokens,
    resolve_context_window_tokens,
)
from intergrax.utils import attribute_access

#: Bedrock Runtime error codes (``ClientError.response["Error"]["Code"]``) -> typed category (EX-D3-03).
_BEDROCK_ERROR_CATEGORIES: dict[str, LLMProviderErrorCategory] = {
    "AccessDeniedException": LLMProviderErrorCategory.PERMISSION,
    "UnrecognizedClientException": LLMProviderErrorCategory.AUTHENTICATION,
    "ExpiredTokenException": LLMProviderErrorCategory.AUTHENTICATION,
    "ValidationException": LLMProviderErrorCategory.INVALID_REQUEST,
    "ResourceNotFoundException": LLMProviderErrorCategory.MODEL_NOT_FOUND,
    "ThrottlingException": LLMProviderErrorCategory.RATE_LIMIT,
    "ServiceQuotaExceededException": LLMProviderErrorCategory.QUOTA_EXHAUSTED,
    "ModelTimeoutException": LLMProviderErrorCategory.TIMEOUT,
    "ModelNotReadyException": LLMProviderErrorCategory.OVERLOADED,
    "ServiceUnavailableException": LLMProviderErrorCategory.OVERLOADED,
    "InternalServerException": LLMProviderErrorCategory.SERVER_ERROR,
    "ModelErrorException": LLMProviderErrorCategory.SERVER_ERROR,
    "ModelStreamErrorException": LLMProviderErrorCategory.SERVER_ERROR,
}


def _bedrock_error_signal(exc: Exception) -> LLMProviderErrorSignal | None:
    """Category and HTTP status of a botocore ``ClientError`` (read by attribute, no SDK import)."""
    response = attribute_access.optional(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    code = error.get("Code") if isinstance(error, dict) else None
    metadata = response.get("ResponseMetadata")
    raw_status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    status = raw_status if isinstance(raw_status, int) and not isinstance(raw_status, bool) else None
    category = _BEDROCK_ERROR_CATEGORIES.get(code) if isinstance(code, str) else None
    if category is None and status is not None:
        category = category_from_status(status)
    if category is None:
        return None
    return LLMProviderErrorSignal(category=category, status_code=status)


def _invoke_stop(payload: object) -> object:
    """Stop reason of an InvokeModel body or stream chunk, including Mistral's ``outputs[0].stop_reason``."""
    direct = invoke_stop_signal(payload)
    if direct:
        return direct
    if isinstance(payload, dict):
        outputs = payload.get("outputs")
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            return outputs[0].get("stop_reason")
    return None


def _invoke_served_model(payload: object) -> object:
    """Model reported by an InvokeModel body (``model``) or an Anthropic ``message_start`` chunk."""
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if isinstance(model, str):
        return model
    message = payload.get("message")
    if isinstance(message, dict):
        return message.get("model")
    return None


def _observe_converse_stop(events: Iterable[Any], signal: StreamFinishSignal) -> Iterator[Any]:
    """Pass Converse stream events through while recording the ``messageStop`` stop reason."""
    for event in events:
        signal.observe_finish(converse_stop_signal(event))
        yield event


class BedrockRuntimeClient(Protocol):
    """Minimal Bedrock Runtime client surface used by this adapter."""

    def invoke_model(self, **kwargs: Any) -> Any: ...

    def invoke_model_with_response_stream(self, **kwargs: Any) -> Any: ...

    def converse(self, **kwargs: Any) -> Any: ...

    def converse_stream(self, **kwargs: Any) -> Any: ...


class _ClosableBedrockRuntimeClient(BedrockRuntimeClient, Protocol):
    """A Bedrock Runtime client the adapter built itself and closes (a per-call client)."""

    def close(self) -> None: ...


class BedrockModelFamily(str, Enum):  # noqa: UP042 - str-mixin kept for compatibility
    """Model family inferred from Bedrock modelId prefix."""

    ANTHROPIC = "anthropic"
    META = "meta"
    MISTRAL = "mistral"
    AMAZON = "amazon"
    UNKNOWN = "unknown"


class BedrockNativeCodec(Protocol):
    """
    A model-family specific codec for Bedrock InvokeModel / InvokeModelWithResponseStream.

    Responsibilities:
      - Build native request body for the given model family
      - Extract final text from InvokeModel response JSON
      - Extract streamed text chunks from InvokeModelWithResponseStream event payload JSON
    """

    def build_body(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
        defaults: dict,
        model_id: str,
    ) -> dict: ...

    def extract_text(self, parsed: dict) -> str: ...

    def extract_stream_text(self, payload: dict) -> str: ...


# -----------------------------
# Codecs (official formats)
# -----------------------------


class AnthropicClaudeCodec:
    """
    Anthropic Claude Messages API format on Bedrock InvokeModel.
    """

    def build_body(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
        defaults: dict,
        model_id: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: NativeToolChoice | None = None,
    ) -> dict:
        temp = temperature if temperature is not None else defaults.get("temperature")
        out_tokens = max_tokens if max_tokens is not None else defaults.get("max_tokens", 1024)

        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": map_anthropic_messages(convo),
            "max_tokens": int(out_tokens),
        }
        if system_text:
            body["system"] = system_text
        if temp is not None:
            body["temperature"] = float(temp)
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        return body

    def extract_tool_calls(self, parsed: dict) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for block in parsed.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            out.append(
                {
                    "id": block.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        return out

    def extract_text(self, parsed: dict) -> str:
        # Typical: {"content":[{"type":"text","text":"..."}], ...}
        content = parsed.get("content")
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text")
                if isinstance(txt, str) and txt:
                    parts.append(txt)
        return "".join(parts)

    def extract_stream_text(self, payload: dict) -> str:
        # Bedrock Anthropic streaming deltas often contain "delta" text blocks.
        delta = payload.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            txt = delta.get("text")
            if isinstance(txt, str) and txt:
                return txt

        cbd = payload.get("content_block_delta")
        if isinstance(cbd, dict):
            d = cbd.get("delta")
            if isinstance(d, dict) and d.get("type") == "text_delta":
                txt = d.get("text")
                if isinstance(txt, str) and txt:
                    return txt

        return ""


class MetaLlamaCodec:
    """
    Meta Llama native completion format on Bedrock InvokeModel.
    Request: {prompt, temperature, top_p, max_gen_len}
    Response: {"generation": "..."} :contentReference[oaicite:3]{index=3}
    """

    def build_body(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
        defaults: dict,
        model_id: str,
    ) -> dict:
        # Llama uses "max_gen_len" instead of max_tokens.
        temp = temperature if temperature is not None else defaults.get("temperature", 0.5)
        max_gen_len = max_tokens if max_tokens is not None else defaults.get("max_gen_len", 512)
        top_p = defaults.get("top_p", 0.9)

        prompt = self._format_llama_prompt(system_text=system_text, convo=convo)

        body: dict = {"prompt": prompt, "temperature": float(temp), "max_gen_len": int(max_gen_len)}
        # top_p is optional
        if top_p is not None:
            body["top_p"] = float(top_p)
        return body

    def extract_text(self, parsed: dict) -> str:
        gen = parsed.get("generation")
        return gen if isinstance(gen, str) else ""

    def extract_stream_text(self, payload: dict) -> str:
        # Streaming chunk contains "generation" segments in AWS examples. :contentReference[oaicite:4]{index=4}
        gen = payload.get("generation")
        return gen if isinstance(gen, str) else ""

    def _format_llama_prompt(self, *, system_text: str, convo: Sequence[ChatMessage]) -> str:
        # Official prompt template uses special tokens and headers. :contentReference[oaicite:5]{index=5}
        parts: list[str] = ["<|begin_of_text|>"]
        if system_text:
            parts.append("<|start_header_id|>system<|end_header_id|>\n")
            parts.append(system_text.strip())
            parts.append("<|eot_id|>\n")

        for m in convo:
            if not m.content:
                continue
            role = m.role if m.role in ("user", "assistant") else "assistant"
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n")
            parts.append(m.content)
            parts.append("<|eot_id|>\n")

        parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
        return "".join(parts)


class MistralCodec:
    """
    Mistral native format on Bedrock InvokeModel.
    Request: {prompt:"<s>[INST] ... [/INST]", max_tokens, temperature}
    Response: {"outputs":[{"text":"..."}]} :contentReference[oaicite:6]{index=6}
    """

    def build_body(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
        defaults: dict,
        model_id: str,
    ) -> dict:
        temp = temperature if temperature is not None else defaults.get("temperature", 0.5)
        out_tokens = max_tokens if max_tokens is not None else defaults.get("max_tokens", 512)

        prompt = self._format_mistral_inst(system_text=system_text, convo=convo)

        return {"prompt": prompt, "max_tokens": int(out_tokens), "temperature": float(temp)}

    def extract_text(self, parsed: dict) -> str:
        outputs = parsed.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return ""
        first = outputs[0]
        if not isinstance(first, dict):
            return ""
        txt = first.get("text")
        return txt if isinstance(txt, str) else ""

    def extract_stream_text(self, payload: dict) -> str:
        # Streaming example extracts outputs[0].text per event. :contentReference[oaicite:7]{index=7}
        return self.extract_text(payload)

    def _format_mistral_inst(self, *, system_text: str, convo: Sequence[ChatMessage]) -> str:
        # AWS example uses "<s>[INST] ... [/INST]" wrapper. :contentReference[oaicite:8]{index=8}
        # Keep it deterministic: flatten to a single instruction block.
        lines: list[str] = []
        if system_text:
            lines.append(system_text.strip())

        for m in convo:
            if not m.content:
                continue
            if m.role == "user":
                lines.append(m.content.strip())
            elif m.role == "assistant":
                # Preserve assistant context for multi-turn by prefixing.
                lines.append(f"Assistant: {m.content.strip()}")
            else:
                lines.append(m.content.strip())

        joined = "\n\n".join([x for x in lines if x])
        return f"<s>[INST] {joined} [/INST]"


class AmazonTitanTextCodec:
    """
    Amazon Titan Text native format on Bedrock InvokeModel.
    Request: {"inputText": "...", "textGenerationConfig": {...}}
    Response: {"results":[{"outputText":"..."}]} :contentReference[oaicite:9]{index=9}
    """

    def build_body(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
        defaults: dict,
        model_id: str,
    ) -> dict:
        temp = temperature if temperature is not None else defaults.get("temperature", 0.7)
        top_p = defaults.get("top_p", 0.9)
        max_count = max_tokens if max_tokens is not None else defaults.get("maxTokenCount", 512)

        # Titan suggests "User: ...\nBot:" conversational prompt format. :contentReference[oaicite:10]{index=10}
        prompt = self._format_titan(system_text=system_text, convo=convo)

        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "temperature": float(temp),
                "topP": float(top_p),
                "maxTokenCount": int(max_count),
                "stopSequences": defaults.get("stopSequences", []),
            },
        }

    def extract_text(self, parsed: dict) -> str:
        results = parsed.get("results")
        if not isinstance(results, list) or not results:
            return ""
        first = results[0]
        if not isinstance(first, dict):
            return ""
        out = first.get("outputText")
        return out if isinstance(out, str) else ""

    def extract_stream_text(self, payload: dict) -> str:
        # For Titan streaming chunks, docs show "outputText" in the decoded bytes.
        out = payload.get("outputText")
        return out if isinstance(out, str) else ""

    def _format_titan(self, *, system_text: str, convo: Sequence[ChatMessage]) -> str:
        # Minimal deterministic transcript.
        lines: list[str] = []
        if system_text:
            lines.append(f"System: {system_text.strip()}")

        for m in convo:
            if not m.content:
                continue
            if m.role == "user":
                lines.append(f"User: {m.content.strip()}")
            elif m.role == "assistant":
                lines.append(f"Bot: {m.content.strip()}")
            else:
                lines.append(m.content.strip())

        # Titan conversational format ends with "Bot:" to cue completion.
        return "\n".join(lines + ["Bot:"])


# -----------------------------
# Adapter
# -----------------------------


@dataclass(frozen=True)
class BedrockAdapterConfig:
    """Small config object to keep adapter initialization explicit."""

    region: str
    model_id: str
    family: BedrockModelFamily


class BedrockChatAdapter(BaseLLMAdapter):
    """
    AWS Bedrock adapter using InvokeModel / InvokeModelWithResponseStream.
    Supports multiple model families by dispatching to native codecs.

    Contract:
      - __init__(client=None, model_id=None, region=None, family=None, **defaults)
      - generate_messages(...) -> str
      - stream_messages(...)   -> Iterable[str]
    """

    _CODECS: dict[BedrockModelFamily, BedrockNativeCodec] = {
        BedrockModelFamily.ANTHROPIC: AnthropicClaudeCodec(),
        BedrockModelFamily.META: MetaLlamaCodec(),
        BedrockModelFamily.MISTRAL: MistralCodec(),
        BedrockModelFamily.AMAZON: AmazonTitanTextCodec(),
    }

    def __init__(
        self,
        client: BedrockRuntimeClient | None = None,
        model_id: str | None = None,
        model: str | None = None,
        region: str | None = None,
        family: BedrockModelFamily | None = None,
        **defaults,
    ):
        super().__init__()
        self._apply_defaults_call_config(defaults)

        resolved_region = region or os.getenv("INTERGRAX_DEFAULT_AWS_REGION", "")
        resolved_model_id = (model_id or model or "").strip()

        if not resolved_region:
            raise RuntimeError("INTERGRAX_DEFAULT_AWS_REGION must be configured for Bedrock adapter.")

        if not resolved_model_id:
            raise RuntimeError(
                "Bedrock adapter requires an explicit model_id (set INTERGRAX_LLM_MODEL "
                "via LLMProfile / llm_profile_from_env)."
            )

        inferred_family = family or self._infer_family_from_model_id(resolved_model_id)
        if inferred_family == BedrockModelFamily.UNKNOWN:
            raise RuntimeError(
                f"Cannot infer Bedrock model family from model_id='{resolved_model_id}'. "
                "Pass family=BedrockModelFamily.<FAMILY> explicitly or register a codec."
            )

        # boto3 takes its timeouts from the client ``Config`` only, so a call that carries an effective
        # timeout runs on a client built for that call and closed after it (EX-D3-01). An injected client
        # cannot be rebuilt; with one, a call that carries a timeout is refused (fail closed).
        self._client_factory: Callable[[float], CallClient[BedrockRuntimeClient]] | None = None
        if client is None:
            client = cast(
                BedrockRuntimeClient,
                boto3.client(
                    service_name="bedrock-runtime",
                    region_name=resolved_region,
                ),
            )

            def _client_with_timeout(timeout: float) -> CallClient[BedrockRuntimeClient]:
                built = cast(
                    _ClosableBedrockRuntimeClient,
                    boto3.client(
                        service_name="bedrock-runtime",
                        region_name=resolved_region,
                        config=Config(connect_timeout=timeout, read_timeout=timeout),
                    ),
                )
                return CallClient[BedrockRuntimeClient](client=built, release=built.close)

            self._client_factory = _client_with_timeout
        self.client: BedrockRuntimeClient = client
        self.config = BedrockAdapterConfig(
            region=resolved_region,
            model_id=resolved_model_id,
            family=inferred_family,
        )
        self.defaults = defaults

        # Keep existing contract compatibility: a single name for token estimation if needed elsewhere.
        self.model_name_for_token_estimation: str = self.config.model_id

        # Conservative default; you can refine per family if you want.
        self._context_window_tokens: int = init_adapter_context_window_tokens(
            provider=LLMProvider.AWS_BEDROCK,
            model=self.config.model_id,
            constructor_kwargs=defaults,
            legacy_windows=self._CONTEXT_WINDOWS,
        )

        self.provider = LLMProvider.AWS_BEDROCK
        self.model = resolved_model_id

        use_converse = defaults.get("use_converse")
        if use_converse is None:
            use_converse = os.getenv("INTERGRAX_BEDROCK_USE_CONVERSE", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
        self._use_converse = bool(use_converse)

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _estimate_extensions(self) -> LLMProviderExtensions:
        return LLMProviderExtensions(usage_source="estimate")

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            if self._use_converse and converse_supported(self.client):
                temp = temperature if temperature is not None else self.defaults.get("temperature")
                req = build_converse_request(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                resp = self._execute(lambda: self._call(lambda c: c.converse(modelId=self.config.model_id, **req)))
                text = extract_converse_text(resp)
                out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
                response = build_adapter_response(
                    content=text,
                    finish_reason=finish_reason_from_raw(converse_stop_signal(resp)),
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                    provider_extensions=self._estimate_extensions(),
                )
                success = True
                return response

            system_text, convo = split_system_messages(messages)
            codec = self._get_codec(self.config.family)

            body = codec.build_body(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                defaults=self.defaults,
                model_id=self.config.model_id,
            )

            raw_body = self._execute(
                lambda: self._call(
                    lambda c: c.invoke_model(
                        modelId=self.config.model_id,
                        body=json.dumps(body).encode("utf-8"),
                        accept="application/json",
                        contentType="application/json",
                    )["body"].read()
                )
            )

            parsed = json.loads(raw_body.decode("utf-8"))

            text = codec.extract_text(parsed)
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
            response = self._served_response(
                call,
                build_adapter_response(
                    content=text,
                    finish_reason=finish_reason_from_raw(_invoke_stop(parsed)),
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                    provider_extensions=self._estimate_extensions(),
                ),
                _invoke_served_model(parsed),
            )
            success = True
            return response

        except Exception as e:
            err_type = type(e).__name__
            raise

        finally:
            usage = response.usage if (success and response and response.usage) else LLMTokenUsage()
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=err_type,
            )

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> Iterable[LLMStreamEvent]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0
        buf: list[str] = []
        signal = StreamFinishSignal()

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            if self._use_converse and converse_stream_supported(self.client):
                temp = temperature if temperature is not None else self.defaults.get("temperature")
                req = build_converse_request(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                )

                def _converse_body_stream() -> Iterable[object]:
                    def _open(c: BedrockRuntimeClient) -> Iterable[object]:
                        resp = c.converse_stream(modelId=self.config.model_id, **req)
                        stream = resp.get("stream") if isinstance(resp, dict) else resp
                        return iter_converse_stream_text(_observe_converse_stop(stream, signal))

                    return self._stream(_open)

                for text in self._execute_streaming(_converse_body_stream):
                    buf.append(text)
                    yield partial_stream_event(delta_content=text)
                out_tok = int(
                    self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation)
                )
                success = True
                finish, refusal = signal.resolve()
                yield final_stream_event(
                    response=build_adapter_response(
                        content="".join(buf),
                        finish_reason=finish,
                        refusal=refusal,
                        usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                        model=self.model,
                        provider=self._provider_slug(),
                        provider_extensions=self._estimate_extensions(),
                    )
                )
                return

            system_text, convo = split_system_messages(messages)
            codec = self._get_codec(self.config.family)

            body = codec.build_body(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                defaults=self.defaults,
                model_id=self.config.model_id,
            )

            served_model: object = None

            def _invoke_stream_body() -> Iterable[object]:
                return self._stream(
                    lambda c: c.invoke_model_with_response_stream(
                        modelId=self.config.model_id,
                        body=json.dumps(body).encode("utf-8"),
                        accept="application/json",
                        contentType="application/json",
                    )["body"]
                )

            for event in self._execute_streaming(_invoke_stream_body):
                chunk = event.get("chunk")
                if not isinstance(chunk, dict):
                    continue
                data = chunk.get("bytes")
                if not data:
                    continue

                payload = json.loads(data.decode("utf-8"))
                signal.observe_finish(_invoke_stop(payload))
                served_model = _invoke_served_model(payload) or served_model
                text = codec.extract_stream_text(payload)
                if text:
                    buf.append(text)
                    yield partial_stream_event(delta_content=text)

            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
            success = True
            finish, refusal = signal.resolve()
            yield final_stream_event(
                response=self._served_response(
                    call,
                    build_adapter_response(
                        content="".join(buf),
                        finish_reason=finish,
                        refusal=refusal,
                        usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                        model=self.model,
                        provider=self._provider_slug(),
                        provider_extensions=self._estimate_extensions(),
                    ),
                    served_model,
                )
            )

        except Exception as e:
            err_type = type(e).__name__
            raise

        finally:
            self.usage.end_call(
                call,
                input_tokens=in_tok,
                output_tokens=out_tok,
                success=success,
                error_type=err_type,
            )

    def supports_streaming(self) -> bool:
        return True

    def supports_call_timeout(self) -> bool:
        """Only an adapter-built client can be rebuilt with the call timeout; an injected one cannot.

        boto3 takes its connect and read timeouts from the client ``Config`` only (EX-D3-01), so an
        adapter with an injected client refuses a call that carries an effective timeout (fail closed).
        """
        return self._client_factory is not None

    def _call[R](self, call: Callable[[BedrockRuntimeClient], R]) -> R:
        """Run ``call`` on the client of this attempt (derived for the effective timeout when one is set)."""
        return call_on_client(
            shared=self.client,
            derive=self._client_factory,
            timeout=self._effective_call_timeout_sec(),
            call=call,
        )

    def _stream[E](self, open_stream: Callable[[BedrockRuntimeClient], Iterable[E]]) -> Iterable[E]:
        """Open a stream on the client of this attempt; a derived client is closed when the stream ends."""
        return stream_on_client(
            shared=self.client,
            derive=self._client_factory,
            timeout=self._effective_call_timeout_sec(),
            open_stream=open_stream,
        )

    def _classify_provider_error(self, exc: Exception) -> LLMProviderErrorSignal | None:
        return _bedrock_error_signal(exc)

    def supports_tool_streaming(self) -> bool:
        """``stream_with_tools`` requires the same Converse/Anthropic route as ``supports_tools``."""
        return self.supports_tools()

    def supports_tools(self) -> bool:
        if self._use_converse and converse_supported(self.client):
            return True
        return self.config.family == BedrockModelFamily.ANTHROPIC

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: NativeToolChoice | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        if not self.supports_tools():
            raise self._capability_mismatch(
                LLMFeature.TOOLS,
                "Tools require Bedrock Converse API or an Anthropic model on InvokeModel.",
            )

        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            if self._use_converse and converse_supported(self.client):
                temp = temperature if temperature is not None else self.defaults.get("temperature")
                req = build_converse_request(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    tools=openai_tools_to_bedrock_converse(tools_schema),
                )
                resp = self._execute(lambda: self._call(lambda c: c.converse(modelId=self.config.model_id, **req)))
                text = extract_converse_text(resp)
                tool_calls = tool_calls_from_openai_dicts(extract_converse_tool_calls(resp))
                out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
                finish = finish_reason_from_raw(converse_stop_signal(resp), has_tool_calls=bool(tool_calls))
                response = build_adapter_response(
                    content=text,
                    finish_reason=finish,
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                    tool_calls=tool_calls,
                    provider_extensions=self._estimate_extensions(),
                )
                success = True
                return response

            system_text, convo = split_system_messages(messages)
            codec = self._get_codec(self.config.family)
            assert isinstance(codec, AnthropicClaudeCodec)

            body = codec.build_body(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                defaults=self.defaults,
                model_id=self.config.model_id,
                tools=openai_tools_to_anthropic(tools_schema),
                tool_choice=tool_choice,
            )

            raw_body = self._execute(
                lambda: self._call(
                    lambda c: c.invoke_model(
                        modelId=self.config.model_id,
                        body=json.dumps(body).encode("utf-8"),
                        accept="application/json",
                        contentType="application/json",
                    )["body"].read()
                )
            )
            parsed = json.loads(raw_body.decode("utf-8"))
            text = codec.extract_text(parsed)
            tool_calls = tool_calls_from_openai_dicts(codec.extract_tool_calls(parsed))
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
            finish = finish_reason_from_raw(_invoke_stop(parsed), has_tool_calls=bool(tool_calls))
            response = self._served_response(
                call,
                build_adapter_response(
                    content=text,
                    finish_reason=finish,
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                    tool_calls=tool_calls,
                    provider_extensions=self._estimate_extensions(),
                ),
                _invoke_served_model(parsed),
            )
            success = True
            return response
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            usage = response.usage if (success and response and response.usage) else LLMTokenUsage()
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=err_type,
            )

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: NativeToolChoice | None = None,
        run_id: str | None = None,
    ) -> Iterable[LLMStreamEvent]:
        if not self.supports_tools():
            raise self._capability_mismatch(
                LLMFeature.TOOL_STREAMING,
                "Tools are not supported for this Bedrock configuration.",
            )

        call = self.usage.begin_call(run_id=run_id, adapter=self)
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0
        buf: list[str] = []
        signal = StreamFinishSignal()
        tool_calls: list[dict[str, Any]] = []

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            if self._use_converse and converse_stream_supported(self.client):
                temp = temperature if temperature is not None else self.defaults.get("temperature")
                req = build_converse_request(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    tools=openai_tools_to_bedrock_converse(tools_schema),
                )

                def _converse_tool_stream() -> Iterable[object]:
                    def _open(c: BedrockRuntimeClient) -> Iterable[object]:
                        resp = c.converse_stream(modelId=self.config.model_id, **req)
                        stream = resp.get("stream") if isinstance(resp, dict) else resp
                        return stream

                    return self._stream(_open)

                active_tools: dict[str, dict[str, Any]] = {}
                for event in self._execute_streaming(_converse_tool_stream):
                    signal.observe_finish(converse_stop_signal(event))
                    for text in iter_converse_stream_text([event]):
                        buf.append(text)
                        yield partial_stream_event(delta_content=text)
                    completed = parse_converse_stream_tool_event(event, active_tools=active_tools)
                    if completed:
                        tool_calls.append(completed)

                for tu in active_tools.values():
                    tool_calls.append(tu)

                typed_calls = tool_calls_from_openai_dicts(tool_calls)
                finish, refusal = signal.resolve(has_tool_calls=bool(typed_calls))
                out_tok = int(
                    self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation)
                )
                success = True
                yield final_stream_event(
                    response=build_adapter_response(
                        content="".join(buf),
                        finish_reason=finish,
                        refusal=refusal,
                        usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                        model=self.model,
                        provider=self._provider_slug(),
                        tool_calls=typed_calls,
                        provider_extensions=self._estimate_extensions(),
                    )
                )
                return

            result = self.generate_with_tools(
                messages,
                tools_schema,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                run_id=run_id,
            )
            content = result.content or ""
            if content:
                yield partial_stream_event(delta_content=content)
            yield final_stream_event(response=result)
            success = True
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(
                call,
                input_tokens=in_tok,
                output_tokens=out_tok,
                success=success,
                error_type=err_type,
            )

    # ------------------------------------------------------------------
    # Extensibility
    # ------------------------------------------------------------------

    @classmethod
    def register_codec(cls, family: BedrockModelFamily, codec: BedrockNativeCodec) -> None:
        """Allow adding support for additional Bedrock model families without editing adapter internals."""
        cls._CODECS[family] = codec

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_codec(self, family: BedrockModelFamily) -> BedrockNativeCodec:
        codec = self._CODECS.get(family)
        if codec is None:
            raise ValueError(
                f"Unsupported Bedrock model family '{family}'. "
                f"Provide family=... or register a codec via BedrockChatAdapter.register_codec()."
            )
        return codec

    _CONTEXT_WINDOWS: dict[str, int] = {
        "anthropic.claude-3-5-sonnet-20241022-v2:0": 200_000,
        "anthropic.claude-3-5-haiku-20241022-v1:0": 200_000,
        "anthropic.claude-3-opus-20240229-v1:0": 200_000,
        "meta.llama3-1-70b-instruct-v1:0": 128_000,
        "meta.llama3-1-8b-instruct-v1:0": 128_000,
        "mistral.mistral-large-2402-v1:0": 128_000,
        "amazon.titan-text-express-v1": 8_000,
    }

    def _estimate_context_window(self, model_id: str) -> int:
        """Resolve through the ModelCatalog; the local table is only a catalog-resolved fallback."""
        return resolve_context_window_tokens(
            LLMProvider.AWS_BEDROCK,
            model_id,
            legacy_windows=self._CONTEXT_WINDOWS,
        )

    def _infer_family_from_model_id(self, model_id: str) -> BedrockModelFamily:
        # Bedrock model ids are usually prefixed as "<provider>.<model...>".
        prefix = model_id.split(".", 1)[0].strip().lower()
        if prefix == "anthropic":
            return BedrockModelFamily.ANTHROPIC
        if prefix == "meta":
            return BedrockModelFamily.META
        if prefix == "mistral":
            return BedrockModelFamily.MISTRAL
        if prefix == "amazon":
            return BedrockModelFamily.AMAZON
        return BedrockModelFamily.UNKNOWN
