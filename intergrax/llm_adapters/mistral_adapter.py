from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from mistralai import Mistral
from mistralai.models import ChatCompletionResponse

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.base import ChatMessage, LLMAdapter


# -----------------------------
# Typed streaming contracts
# -----------------------------
class _MistralDelta(Protocol):
    content: Optional[str]


class _MistralStreamChoice(Protocol):
    delta: _MistralDelta


class _MistralStreamChunk(Protocol):
    choices: List[_MistralStreamChoice]


class MistralChatAdapter(LLMAdapter):
    """
    Mistral adapter based on the official Mistral Python SDK (mistralai).

    - Uses Mistral (official client type).
    - Supports:
        - generate_messages
        - stream_messages
    - Tools + structured output are intentionally not wired here (yet).
    """

    _MISTRAL_CONTEXT_WINDOWS: Dict[str, int] = {
        "mistral-small-latest": 32_000,
        "mistral-medium-latest": 32_000,
        "mistral-large-latest": 32_000,
        "codestral-latest": 32_000,
    }

    def __init__(
        self,
        client: Optional[Mistral] = None,
        model: Optional[str] = None,
        **defaults,
    ):
        super().__init__()

        # Framework-wide defaults should be routed via GLOBAL_SETTINGS.
        self.client: Mistral = client or Mistral()
        self.model: str = model or GLOBAL_SETTINGS.default_mistral_model
        self.model_name_for_token_estimation: str = self.model
        self.defaults = defaults

        self._context_window_tokens: int = self._estimate_mistral_context_window(self.model)

    @property
    def context_window_tokens(self) -> int:
        """
        Cached maximum context window (input + output tokens) for the configured model.
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

            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            res: ChatCompletionResponse = self.client.chat.complete(**payload)

            if not res.choices:
                success = True
                return ""

            text = res.choices[0].message.content or ""
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

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = self._split_system(messages)

            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            stream: Iterable[_MistralStreamChunk] = self.client.chat.complete(**payload)

            buf: List[str] = []

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    buf.append(delta.content)
                    yield delta.content

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
    # Internals
    # ------------------------------------------------------------------

    def _estimate_mistral_context_window(self, model: str) -> int:
        return self._MISTRAL_CONTEXT_WINDOWS.get(model, 32_000)

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
        Build a minimal, explicit Mistral Chat payload.

        We force response_format={"type":"text"} to keep extraction deterministic
        (content as a plain string, not a list of chunks).
        """
        temp = temperature if temperature is not None else self.defaults.get("temperature", None)
        out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)

        mapped = self._map_messages(system_text=system_text, convo=convo)

        payload: dict = {
            "model": self.model,
            "messages": mapped,
            "stream": stream,
            "response_format": {"type": "text"},
        }

        if temp is not None:
            payload["temperature"] = float(temp)
        if out_tokens is not None:
            payload["max_tokens"] = int(out_tokens)

        return payload

    def _map_messages(self, *, system_text: str, convo: Sequence[ChatMessage]) -> List[dict]:
        """
        Map ChatMessage -> Mistral chat completion message dicts.
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
