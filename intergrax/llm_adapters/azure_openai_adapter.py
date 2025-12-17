# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openai import AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.base import BaseLLMAdapter, ChatMessage


class AzureOpenAIChatAdapter(BaseLLMAdapter):
    """
    Azure OpenAI adapter based on the official OpenAI Python SDK (AzureOpenAI).

    Contract (aligned with other adapters):
      - __init__(client: Optional[AzureOpenAI] = None, deployment: Optional[str] = None, **defaults)
      - generate_messages(...) -> str
      - stream_messages(...)   -> Iterable[str]

    Notes:
      - On Azure, the 'model' field in chat completions must be set to the DEPLOYMENT NAME,
        not the underlying model id. :contentReference[oaicite:2]{index=2}
      - Tools / structured output are not wired here (yet).
    """

    # Conservative context window estimates (input + output).
    # Keep safe unless you add a token counter per deployment.
    _AZURE_CONTEXT_WINDOWS: Dict[str, int] = {
        # deployments are tenant-specific; keep fallback conservative
    }

    def __init__(
        self,
        client: Optional[AzureOpenAI] = None,
        deployment: Optional[str] = None,
        **defaults,
    ):
        super().__init__()

        # Framework-wide defaults should be routed via GLOBAL_SETTINGS.
        # Keep these names consistent with your settings pattern.
        endpoint = GLOBAL_SETTINGS.azure_openai_endpoint
        api_version = GLOBAL_SETTINGS.azure_openai_api_version
        default_deployment = GLOBAL_SETTINGS.default_azure_openai_deployment

        # Official SDK can read API key from env (AZURE_OPENAI_API_KEY),
        # but we keep instantiation explicit and consistent with other adapters.
        self.client: AzureOpenAI = client or AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
        )

        # Azure: "model" = deployment name.
        self.deployment: str = deployment or default_deployment
        self.model_name_for_token_estimation: str = self.deployment

        self.defaults = defaults
        self._context_window_tokens: int = self._estimate_context_window(self.deployment)

    @property
    def context_window_tokens(self) -> int:
        """
        Cached maximum context window (input + output tokens) for the configured deployment.
        """
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
    ) -> str:
        system_text, convo = self._split_system(messages)

        payload = self._build_chat_params(
            system_text=system_text,
            convo=convo,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        res: ChatCompletion = self.client.chat.completions.create(**payload)

        if not res.choices:
            return ""

        msg = res.choices[0].message
        return msg.content or ""

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        system_text, convo = self._split_system(messages)

        payload = self._build_chat_params(
            system_text=system_text,
            convo=convo,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        stream = self.client.chat.completions.create(**payload)

        # The OpenAI SDK returns an iterator of typed ChatCompletionChunk.
        for chunk in stream:
            c: ChatCompletionChunk = chunk
            if not c.choices:
                continue

            delta = c.choices[0].delta
            if delta is None:
                continue

            if delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # Tools / structured output (not wired)
    # ------------------------------------------------------------------

    def supports_tools(self) -> bool:
        return False

    def generate_with_tools(self, *a, **k):
        raise NotImplementedError("AzureOpenAI tools are not wired in this adapter.")

    def stream_with_tools(self, *a, **k):
        raise NotImplementedError("AzureOpenAI tools are not wired in this adapter.")

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        raise NotImplementedError("Structured output is not implemented for AzureOpenAIChatAdapter.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_context_window(self, deployment: str) -> int:
        # Deployments are user-defined names; keep fallback conservative.
        return self._AZURE_CONTEXT_WINDOWS.get(deployment, 32_000)

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

    def _build_chat_params(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> dict:
        """
        Build a minimal, explicit Azure Chat Completions payload.

        Azure expects:
          - model: deployment name (not model id) :contentReference[oaicite:3]{index=3}
          - messages: list of {role, content}
        """
        temp = temperature if temperature is not None else self.defaults.get("temperature", None)
        out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)

        mapped = self._map_messages(system_text=system_text, convo=convo)

        payload: dict = {
            "model": self.deployment,
            "messages": mapped,
            "stream": stream,
        }

        if temp is not None:
            payload["temperature"] = float(temp)
        if out_tokens is not None:
            payload["max_tokens"] = int(out_tokens)

        return payload

    def _map_messages(self, *, system_text: str, convo: Sequence[ChatMessage]) -> List[dict]:
        """
        Map ChatMessage -> Azure chat completion message dicts.
        """
        out: List[dict] = []

        if system_text:
            out.append({"role": "system", "content": system_text})

        for m in convo:
            if not m.content:
                continue

            role = m.role
            if role not in ("user", "assistant"):
                # Tools are not wired; treat other roles as assistant text.
                role = "assistant"

            out.append({"role": role, "content": m.content})

        return out
