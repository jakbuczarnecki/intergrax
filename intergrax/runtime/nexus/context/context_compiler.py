# © Artur Czarnecki. All rights reserved.

"""Unified Context Compiler — global budget allocator (Phase MEM-DEPTH-1.1)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from intergrax.contracts.host_profile_slices import ContextDecisionProfile
from intergrax.llm.messages import ChatMessage

from intergrax.runtime.nexus.context.context_budget import estimate_tokens, resolve_input_budget_tokens

if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_compiler_models import (
    ContextCandidate,
    ContextCandidateSource,
    ContextCompileResult,
    DegradationStepKind,
)
from intergrax.runtime.nexus.context.degradation_ladder import (
    LADDER_ORDER,
    apply_degradation_step,
)


def _default_count_tokens(text: str) -> int:
    return estimate_tokens(len(text))


def _resolve_decision_profile(config: "RuntimeConfig") -> ContextDecisionProfile:
    raw: Optional[Dict[str, Any]] = config.context_decision_profile
    if raw:
        return ContextDecisionProfile.model_validate(raw)
    return ContextDecisionProfile()


def _last_user_index(messages: Sequence[ChatMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return max(0, len(messages) - 1)


_CE_CONTEXT_TAG = re.compile(
    r"^\[context:(?P<source>[a-z_]+):[^\]]+\]\s",
    re.IGNORECASE,
)


def _detect_injection_source(content: str) -> ContextCandidateSource:
    """Prefer CE-FMT-1 ``[context:source:id]`` tags over legacy string heuristics (CE-10.3)."""
    match = _CE_CONTEXT_TAG.match(content or "")
    if match:
        from intergrax.context.contracts import ContextFragmentSource
        from intergrax.runtime.nexus.context.fragment_bridge import candidate_source_from_fragment

        try:
            fragment_source = ContextFragmentSource(match.group("source"))
            return candidate_source_from_fragment(fragment_source)
        except ValueError:
            pass
    lowered = content.lower()
    if "long-term memory" in lowered or "user memory" in lowered or "ltm:" in lowered:
        return ContextCandidateSource.LONGTERM_MEMORY
    if "rag context" in lowered or "retrieved documents" in lowered:
        return ContextCandidateSource.RAG
    if "web search" in lowered or "websearch" in lowered:
        return ContextCandidateSource.WEBSEARCH
    if "attachments" in lowered or "session attachments" in lowered:
        return ContextCandidateSource.ATTACHMENTS
    if "tool" in lowered and "context" in lowered:
        return ContextCandidateSource.TOOLS
    return ContextCandidateSource.OTHER


def classify_candidates(
    messages: Sequence[ChatMessage],
    *,
    count_tokens: Callable[[str], int],
) -> List[ContextCandidate]:
    if not messages:
        return []

    last_user = _last_user_index(messages)
    candidates: List[ContextCandidate] = []

    for index, message in enumerate(messages):
        content = message.content or ""
        token_estimate = count_tokens(content)
        mandatory = index == last_user or (index == 0 and message.role == "system")

        if index == last_user:
            source = ContextCandidateSource.USER_TURN
            score = 1.0
        elif index == 0 and message.role == "system":
            source = ContextCandidateSource.SYSTEM_INSTRUCTIONS
            score = 1.0
        elif message.role == "system" and index < last_user:
            source = _detect_injection_source(content)
            score = 0.75
            mandatory = False
        elif message.role in {"user", "assistant"}:
            source = ContextCandidateSource.SESSION_HISTORY
            score = 0.65
            mandatory = index >= last_user - 1
        else:
            source = ContextCandidateSource.OTHER
            score = 0.5

        candidates.append(
            ContextCandidate(
                source=source,
                message_index=index,
                score=score,
                token_estimate=token_estimate,
                mandatory=mandatory,
            )
        )
    return candidates


class ContextCompiler:
  """Collect, rank, budget, and degrade context before LLM invocation."""

  def __init__(
      self,
      *,
      count_tokens: Callable[[str], int] | None = None,
      margin_tokens: int = 256,
  ) -> None:
      self._count_tokens = count_tokens or _default_count_tokens
      self._margin_tokens = margin_tokens

  def count_tokens(self, text: str) -> int:
      """Public token estimator for CE planning and compilation."""
      return self._count_tokens(text)

  def resolve_global_input_budget(
      self,
      config: "RuntimeConfig",
      *,
      max_output_tokens: Optional[int] = None,
  ) -> int:
      """Canonical global model-input budget resolver."""
      adapter = config.llm_adapter
      budget_tokens = resolve_input_budget_tokens(
          adapter,
          max_output_tokens=max_output_tokens,
          margin_tokens=self._margin_tokens,
      )
      if config.context_budget_policy is not None:
          budget_tokens = min(budget_tokens, config.context_budget_policy.max_tokens_estimate)
      return budget_tokens

  def compile(
      self,
      messages: List[ChatMessage],
      config: "RuntimeConfig",
      *,
      max_output_tokens: Optional[int] = None,
  ) -> ContextCompileResult:
      decision = _resolve_decision_profile(config)

      working = list(messages)
      if not decision.include_session_history:
          last_user = _last_user_index(working)
          preserved: List[ChatMessage] = []
          for index, message in enumerate(working):
              if index == 0 and message.role == "system":
                  preserved.append(message)
              elif index == last_user:
                  preserved.append(message)
              elif message.role == "system":
                  preserved.append(message)
          working = preserved

      budget_tokens = self.resolve_global_input_budget(
          config,
          max_output_tokens=max_output_tokens,
      )

      candidates = classify_candidates(working, count_tokens=self._count_tokens)
      total_tokens = sum(candidate.token_estimate for candidate in candidates)

      if total_tokens <= budget_tokens:
          return ContextCompileResult(
              messages=working,
              total_tokens=total_tokens,
              budget_tokens=budget_tokens,
              degradation_steps=(DegradationStepKind.FULL.value,),
              trimmed=False,
          )

      applied_steps: list[str] = []
      bytes_removed = 0
      trimmed = False

      for step in LADDER_ORDER:
          if step == DegradationStepKind.FULL:
              continue
          if step == DegradationStepKind.REDUCE_INJECTION_BLOCKS:
              step = DegradationStepKind.DROP_LOWEST_SCORED

          candidates = classify_candidates(working, count_tokens=self._count_tokens)
          if sum(c.token_estimate for c in candidates) <= budget_tokens:
              break

          result = apply_degradation_step(
              messages=working,
              candidates=candidates,
              step=step,
              budget_tokens=budget_tokens,
              prefer_longterm_memory=decision.prefer_longterm_memory,
              prefer_rag_when_enabled=decision.prefer_rag_when_enabled,
              count_tokens=self._count_tokens,
          )
          if result is None:
              continue

          working = result.messages
          applied_steps.append(result.step.value)
          bytes_removed += result.bytes_removed
          trimmed = True
          candidates = classify_candidates(working, count_tokens=self._count_tokens)
          if sum(c.token_estimate for c in candidates) <= budget_tokens:
              break

      working = self._enforce_hard_budget(working, budget_tokens)
      final_candidates = classify_candidates(working, count_tokens=self._count_tokens)
      final_tokens = sum(candidate.token_estimate for candidate in final_candidates)

      return ContextCompileResult(
          messages=working,
          total_tokens=min(final_tokens, budget_tokens),
          budget_tokens=budget_tokens,
          degradation_steps=tuple(applied_steps) if applied_steps else (DegradationStepKind.FULL.value,),
          trimmed=trimmed or final_tokens > budget_tokens,
          bytes_removed=bytes_removed,
      )

  def _enforce_hard_budget(
      self,
      messages: List[ChatMessage],
      budget_tokens: int,
  ) -> List[ChatMessage]:
      """Last-resort trim until estimated tokens fit budget."""
      from intergrax.runtime.nexus.context.context_budget import (
          ContextBudgetPolicy,
          trim_message_to_budget_tokenizer_aware,
      )

      def total(msgs: List[ChatMessage]) -> int:
          return sum(self._count_tokens(message.content or "") for message in msgs)

      if total(messages) <= budget_tokens:
          return messages

      policy = ContextBudgetPolicy(
          max_chars=budget_tokens * 4,
          max_tokens_estimate=budget_tokens,
      )
      last_user = _last_user_index(messages)
      trimmed: List[ChatMessage] = []
      for index, message in enumerate(messages):
          if index == last_user:
              trimmed.append(message)
              continue
          result = trim_message_to_budget_tokenizer_aware(
              message.content or "",
              policy,
              count_tokens=self._count_tokens,
          )
          trimmed.append(
              ChatMessage(role=message.role, content=result.message, metadata=message.metadata)
          )

      while total(trimmed) > budget_tokens and len(trimmed) > 1:
          drop_index = 1 if trimmed[0].role == "system" else 0
          if drop_index >= len(trimmed) - 1:
              break
          trimmed.pop(drop_index)

      return trimmed
