# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import boto3
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.llm_adapter import ChatMessage, LLMAdapter
from intergrax.llm_adapters.llm_provider import LLMProvider


class BedrockModelFamily(str, Enum):
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
        temperature: Optional[float],
        max_tokens: Optional[int],
        defaults: Dict,
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
    See 'anthropic_version=bedrock-2023-05-31' requirement and messages structure. :contentReference[oaicite:2]{index=2}
    """

    def build_body(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: Optional[float],
        max_tokens: Optional[int],
        defaults: Dict,
        model_id: str,
    ) -> dict:
        # Resolve parameters
        temp = temperature if temperature is not None else defaults.get("temperature")
        out_tokens = max_tokens if max_tokens is not None else defaults.get("max_tokens", 1024)

        messages: List[dict] = []
        for m in convo:
            if not m.content:
                continue
            role = m.role if m.role in ("user", "assistant") else "assistant"
            messages.append({"role": role, "content": [{"type": "text", "text": m.content}]})

        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": int(out_tokens),
        }
        if system_text:
            body["system"] = system_text
        if temp is not None:
            body["temperature"] = float(temp)

        return body

    def extract_text(self, parsed: dict) -> str:
        # Typical: {"content":[{"type":"text","text":"..."}], ...}
        content = parsed.get("content")
        if not isinstance(content, list):
            return ""
        parts: List[str] = []
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
        temperature: Optional[float],
        max_tokens: Optional[int],
        defaults: Dict,
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
        parts: List[str] = ["<|begin_of_text|>"]
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
        temperature: Optional[float],
        max_tokens: Optional[int],
        defaults: Dict,
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
        lines: List[str] = []
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
        temperature: Optional[float],
        max_tokens: Optional[int],
        defaults: Dict,
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
        # For Titan streaming chunks, docs show "outputText" in the decoded bytes. :contentReference[oaicite:11]{index=11}
        out = payload.get("outputText")
        return out if isinstance(out, str) else ""

    def _format_titan(self, *, system_text: str, convo: Sequence[ChatMessage]) -> str:
        # Minimal deterministic transcript.
        lines: List[str] = []
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


class BedrockChatAdapter(LLMAdapter):
    """
    AWS Bedrock adapter using InvokeModel / InvokeModelWithResponseStream.
    Supports multiple model families by dispatching to native codecs.

    Contract:
      - __init__(client: Optional[BedrockRuntimeClient] = None, model_id: Optional[str] = None, region: Optional[str] = None, family: Optional[BedrockModelFamily] = None, **defaults)
      - generate_messages(...) -> str
      - stream_messages(...)   -> Iterable[str]
    """

    _CODECS: Dict[BedrockModelFamily, BedrockNativeCodec] = {
        BedrockModelFamily.ANTHROPIC: AnthropicClaudeCodec(),
        BedrockModelFamily.META: MetaLlamaCodec(),
        BedrockModelFamily.MISTRAL: MistralCodec(),
        BedrockModelFamily.AMAZON: AmazonTitanTextCodec(),
    }

    def __init__(
        self,
        client: Optional[BedrockRuntimeClient] = None,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        family: Optional[BedrockModelFamily] = None,
        **defaults,
    ):
        super().__init__()

        resolved_region = region or GLOBAL_SETTINGS.aws_region
        resolved_model_id = model_id or GLOBAL_SETTINGS.default_bedrock_model_id

        inferred_family = family or self._infer_family_from_model_id(resolved_model_id)

        self.client: BedrockRuntimeClient = client or boto3.client(
            service_name="bedrock-runtime",
            region_name=resolved_region,
        )
        self.config = BedrockAdapterConfig(
            region=resolved_region,
            model_id=resolved_model_id,
            family=inferred_family,
        )
        self.defaults = defaults

        # Keep existing contract compatibility: a single name for token estimation if needed elsewhere.
        self.model_name_for_token_estimation: str = self.config.model_id

        # Conservative default; you can refine per family if you want.
        self._context_window_tokens: int = self._estimate_context_window(self.config.model_id)

        self.provider = LLMProvider.AWS_BEDROCK

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = self._split_system(messages)
            codec = self._get_codec(self.config.family)

            body = codec.build_body(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                defaults=self.defaults,
                model_id=self.config.model_id,
            )

            res = self.client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(body).encode("utf-8"),
                accept="application/json",
                contentType="application/json",
            )

            raw = res["body"].read().decode("utf-8")
            parsed = json.loads(raw)

            text = codec.extract_text(parsed)
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))

            success = True
            return text

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


    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[str]:
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        buf: List[str] = []

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = self._split_system(messages)
            codec = self._get_codec(self.config.family)

            body = codec.build_body(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                defaults=self.defaults,
                model_id=self.config.model_id,
            )

            res = self.client.invoke_model_with_response_stream(
                modelId=self.config.model_id,
                body=json.dumps(body).encode("utf-8"),
                accept="application/json",
                contentType="application/json",
            )

            stream = res["body"]
            for event in stream:
                chunk = event.get("chunk")
                if not isinstance(chunk, dict):
                    continue
                data = chunk.get("bytes")
                if not data:
                    continue

                payload = json.loads(data.decode("utf-8"))
                text = codec.extract_stream_text(payload)
                if text:
                    buf.append(text)
                    yield text

            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
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

    def _estimate_context_window(self, model_id: str) -> int:
        # TODO: Optional: refine using a lookup table per model_id.
        return 32_000

    def _split_system(self, messages: Sequence[ChatMessage]) -> Tuple[str, List[ChatMessage]]:
        system_parts: List[str] = []
        convo: List[ChatMessage] = []

        for m in messages:
            if m.role == "system":
                if m.content:
                    system_parts.append(m.content)
                continue
            convo.append(m)

        return ("\n\n".join(system_parts).strip(), convo)

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
