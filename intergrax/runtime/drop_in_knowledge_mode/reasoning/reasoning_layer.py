# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.drop_in_knowledge_mode.config import ReasoningConfig, ReasoningMode, RuntimeConfig
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState


@dataclass(frozen=True)
class ReasoningApplyResult:
    """
    Result of applying reasoning policy to system instructions.
    """

    mode: ReasoningMode
    applied: bool
    final_instructions: str
    extra: Dict[str, Any]


class ReasoningLayer:
    """
    Reasoning / Chain-of-Thought layer for Drop-In Knowledge Runtime.

    v1 responsibilities:
      - Augment system instructions based on ReasoningMode.
      - Never store or expose raw chain-of-thought content.
      - Write lightweight observability info into RuntimeState.debug_trace.

    v1 does NOT:
      - Run separate planning calls.
      - Change tools/websearch/RAG flow.
      - Perform self-critique passes.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def apply_reasoning_to_instructions(
        self,
        state: RuntimeState,
        base_system_instructions: str,
    ) -> ReasoningApplyResult:
        """
        Returns the final system instructions after applying the configured reasoning mode.
        Updates RuntimeState.reasoning_mode and debug_trace["reasoning"].

        Note: This method must be a no-op in DIRECT mode.
        """

        reasoning_config = self._get_reasoning_config()
        mode = reasoning_config.mode

        # Keep state consistent even if engine forgets to set it.
        state.reasoning_mode = mode

        if not base_system_instructions:
            base_system_instructions = ""

        if mode == ReasoningMode.DIRECT:
            result = ReasoningApplyResult(
                mode=mode,
                applied=False,
                final_instructions=base_system_instructions,
                extra={"note": "DIRECT mode (no reasoning augmentation)."},
            )
            self._write_reasoning_debug(state, result)
            return result

        if mode == ReasoningMode.COT_INTERNAL:
            augmentation = self._cot_internal_instructions()
            final = self._join_instructions(base_system_instructions, augmentation)

            result = ReasoningApplyResult(
                mode=mode,
                applied=True,
                final_instructions=final,
                extra={
                    "note": "COT_INTERNAL mode (internal reasoning; do not reveal chain-of-thought).",
                    "capture_reasoning_trace": bool(reasoning_config.capture_reasoning_trace),
                },
            )
            self._write_reasoning_debug(state, result)
            return result

        # Forward-compatible default: treat unknown modes as DIRECT.
        result = ReasoningApplyResult(
            mode=ReasoningMode.DIRECT,
            applied=False,
            final_instructions=base_system_instructions,
            extra={
                "note": f"Unknown ReasoningMode '{mode}'. Fallback to DIRECT.",
                "requested_mode": str(mode),
            },
        )
        state.reasoning_mode = ReasoningMode.DIRECT
        self._write_reasoning_debug(state, result)
        return result

    # -----------------------
    # Internals
    # -----------------------

    def _get_reasoning_config(self) -> ReasoningConfig:
        """
        Normalizes missing config to a default ReasoningConfig(DIRECT).
        """
        if self._config.reasoning_config is None:
            return ReasoningConfig(mode=ReasoningMode.DIRECT)
        return self._config.reasoning_config

    def _cot_internal_instructions(self) -> str:
        """
        A minimal, safe instruction block for internal step-by-step reasoning.

        Important: We instruct the model to keep chain-of-thought private
        and produce only the final answer to the user.
        """
        return (
            "Reason internally step by step before answering.\n"
            "Do not reveal your chain-of-thought, internal steps, or hidden reasoning.\n"
            "Provide only the final answer, clearly and concisely.\n"
            "If uncertainty exists, state it explicitly and explain what is missing."
        )

    def _join_instructions(self, base: str, addition: str) -> str:
        base = (base or "").strip()
        addition = (addition or "").strip()
        if not base:
            return addition
        if not addition:
            return base
        return f"{base}\n\n{addition}"

    def _write_reasoning_debug(self, state: RuntimeState, result: ReasoningApplyResult) -> None:
        """
        Writes reasoning metadata to state.debug_trace and (optionally) to state.reasoning_trace.
        Never stores raw chain-of-thought.
        """
        if state.debug_trace is None:
            state.debug_trace = {}

        state.debug_trace["reasoning"] = {
            "mode": result.mode.value,
            "applied": result.applied,
            "extra": result.extra,
        }

        # Optional lightweight per-request trace object (still without CoT content).
        # This is reserved for future PLAN_EXECUTE / structured reasoning modes.
        if state.reasoning_trace is None:
            state.reasoning_trace = {}
        state.reasoning_trace.update(
            {
                "mode": result.mode.value,
                "applied": result.applied,
            }
        )
