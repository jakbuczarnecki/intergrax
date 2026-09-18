# © Artur Czarnecki. All rights reserved.

"""Unified Context Compiler — global budget allocator (Phase MEM-DEPTH-1.1)."""

from __future__ import annotations

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
from intergrax.context.budget.contracts import ContextBudgetUnsatisfiableError
from intergrax.context.budget.mandatory_base_messages import (
    last_user_message_index,
    mandatory_base_message_indices,
    estimate_mandatory_base_message_tokens,
)
from intergrax.context.contracts import ProviderFragmentIdentityMap
from intergrax.context.budget.degradation import ContextDegradationPolicy, DefaultContextDegradationPolicy
from intergrax.runtime.nexus.context.degradation_ladder import apply_degradation_step


def _default_count_tokens(text: str) -> int:
    return estimate_tokens(len(text))


def _resolve_decision_profile(config: "RuntimeConfig") -> ContextDecisionProfile:
    raw: Optional[Dict[str, Any]] = config.context_decision_profile
    if raw:
        return ContextDecisionProfile.model_validate(raw)
    return ContextDecisionProfile()


def classify_candidates(
    messages: Sequence[ChatMessage],
    *,
    count_tokens: Callable[[str], int],
    provider_fragment_identity: ProviderFragmentIdentityMap | None = None,
) -> List[ContextCandidate]:
    if not messages:
        return []

    from intergrax.runtime.nexus.context.fragment_bridge import candidate_source_from_fragment

    last_user = last_user_message_index(messages)
    provider_entry_ids = (
        provider_fragment_identity.entry_ids()
        if provider_fragment_identity is not None
        else frozenset()
    )
    mandatory_indices = mandatory_base_message_indices(
        messages,
        provider_fragment_entry_ids=provider_entry_ids,
    )
    candidates: List[ContextCandidate] = []

    for index, message in enumerate(messages):
        content = message.content or ""
        token_estimate = count_tokens(content)
        provider_entry = (
            provider_fragment_identity.lookup(message.entry_id)
            if provider_fragment_identity is not None
            else None
        )
        if provider_entry is not None:
            source = candidate_source_from_fragment(provider_entry.source)
            mandatory = provider_entry.mandatory
            score = 1.0 if mandatory else 0.75
        elif index == last_user:
            source = ContextCandidateSource.USER_TURN
            mandatory = index in mandatory_indices
            score = 1.0
        elif message.role == "system":
            source = ContextCandidateSource.SYSTEM_INSTRUCTIONS
            mandatory = index in mandatory_indices
            score = 1.0
        elif message.role in {"user", "assistant"}:
            source = ContextCandidateSource.SESSION_HISTORY
            mandatory = index in mandatory_indices
            score = 0.65
        else:
            source = ContextCandidateSource.OTHER
            mandatory = index in mandatory_indices
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
      degradation_policy: ContextDegradationPolicy | None = None,
  ) -> None:
      self._count_tokens = count_tokens or _default_count_tokens
      self._margin_tokens = margin_tokens
      self._degradation_policy = degradation_policy or DefaultContextDegradationPolicy()

  def count_tokens(self, text: str) -> int:
      """Public token estimator for CE planning and compilation."""
      return self._count_tokens(text)

  @property
  def margin_tokens(self) -> int:
      return self._margin_tokens

  @property
  def degradation_policy(self) -> ContextDegradationPolicy:
      return self._degradation_policy

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
      input_budget_tokens: Optional[int] = None,
      provider_fragment_identity: ProviderFragmentIdentityMap | None = None,
  ) -> ContextCompileResult:
      decision = _resolve_decision_profile(config)

      working = list(messages)
      if not decision.include_session_history:
          last_user = last_user_message_index(working)
          preserved: List[ChatMessage] = []
          for index, message in enumerate(working):
              if index == 0 and message.role == "system":
                  preserved.append(message)
              elif index == last_user:
                  preserved.append(message)
              elif message.role == "system":
                  preserved.append(message)
          working = preserved

      if input_budget_tokens is not None:
          budget_tokens = input_budget_tokens
      else:
          budget_tokens = self.resolve_global_input_budget(
              config,
              max_output_tokens=max_output_tokens,
          )

      candidates = classify_candidates(
          working,
          count_tokens=self._count_tokens,
          provider_fragment_identity=provider_fragment_identity,
      )
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

      for step in self._degradation_policy.ladder_order():
          if step == DegradationStepKind.FULL:
              continue
          if step == DegradationStepKind.REDUCE_INJECTION_BLOCKS:
              step = DegradationStepKind.DROP_LOWEST_SCORED

          candidates = classify_candidates(
              working,
              count_tokens=self._count_tokens,
              provider_fragment_identity=provider_fragment_identity,
          )
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
          candidates = classify_candidates(
              working,
              count_tokens=self._count_tokens,
              provider_fragment_identity=provider_fragment_identity,
          )
          if sum(c.token_estimate for c in candidates) <= budget_tokens:
              break

      working = self._enforce_hard_budget(
          working,
          budget_tokens,
          provider_fragment_identity=provider_fragment_identity,
      )
      final_candidates = classify_candidates(
          working,
          count_tokens=self._count_tokens,
          provider_fragment_identity=provider_fragment_identity,
      )
      final_tokens = sum(candidate.token_estimate for candidate in final_candidates)

      if final_tokens > budget_tokens:
          mandatory_tokens = sum(
              candidate.token_estimate for candidate in final_candidates if candidate.mandatory
          )
          raise ContextBudgetUnsatisfiableError(
              detail="compiled_context_exceeds_budget",
              mandatory_tokens=mandatory_tokens,
              available_tokens=budget_tokens,
          )

      return ContextCompileResult(
          messages=working,
          total_tokens=final_tokens,
          budget_tokens=budget_tokens,
          degradation_steps=tuple(applied_steps) if applied_steps else (DegradationStepKind.FULL.value,),
          trimmed=trimmed,
          bytes_removed=bytes_removed,
      )

  def _enforce_hard_budget(
      self,
      messages: List[ChatMessage],
      budget_tokens: int,
      *,
      provider_fragment_identity: ProviderFragmentIdentityMap | None = None,
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

      last_user = last_user_message_index(messages)
      provider_entry_ids = (
          provider_fragment_identity.entry_ids()
          if provider_fragment_identity is not None
          else frozenset()
      )
      last_user_tokens = self._count_tokens(messages[last_user].content or "")
      if last_user_tokens > budget_tokens:
          raise ContextBudgetUnsatisfiableError(
              detail="mandatory_user_turn_exceeds_budget",
              mandatory_tokens=last_user_tokens,
              available_tokens=budget_tokens,
          )
      mandatory_base_tokens = estimate_mandatory_base_message_tokens(
          messages,
          count_text=self._count_tokens,
          provider_fragment_entry_ids=provider_entry_ids,
      )
      if mandatory_base_tokens > budget_tokens:
          raise ContextBudgetUnsatisfiableError(
              detail="mandatory_system_instructions_exceed_budget",
              mandatory_tokens=mandatory_base_tokens,
              available_tokens=budget_tokens,
          )

      mandatory_indices = mandatory_base_message_indices(
          messages,
          provider_fragment_entry_ids=provider_entry_ids,
      )
      policy = ContextBudgetPolicy(
          max_chars=budget_tokens * 4,
          max_tokens_estimate=budget_tokens,
      )
      working: List[ChatMessage] = []
      for index, message in enumerate(messages):
          if index in mandatory_indices:
              working.append(message)
              continue
          trim_result = trim_message_to_budget_tokenizer_aware(
              message.content or "",
              policy,
              count_tokens=self._count_tokens,
          )
          working.append(
              ChatMessage(
                  role=message.role,
                  content=trim_result.message,
                  entry_id=message.entry_id,
                  tool_calls=message.tool_calls,
                  tool_call_id=message.tool_call_id,
                  metadata=message.metadata,
              )
          )

      while total(working) > budget_tokens:
          mandatory_now = mandatory_base_message_indices(
              working,
              provider_fragment_entry_ids=provider_entry_ids,
          )
          drop_index: int | None = None
          for index in range(len(working) - 1, -1, -1):
              if index not in mandatory_now:
                  drop_index = index
                  break
          if drop_index is None:
              mandatory_tokens = estimate_mandatory_base_message_tokens(
                  working,
                  count_text=self._count_tokens,
                  provider_fragment_entry_ids=provider_entry_ids,
              )
              raise ContextBudgetUnsatisfiableError(
                  detail="compiled_context_exceeds_budget",
                  mandatory_tokens=mandatory_tokens,
                  available_tokens=budget_tokens,
              )
          working.pop(drop_index)

      return working
