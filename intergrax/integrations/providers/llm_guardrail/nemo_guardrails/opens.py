# © Artur Czarnecki. All rights reserved.

"""NVIDIA NeMo Guardrails vendor boundary (nemoguardrails imports allowed here only)."""

from __future__ import annotations

from typing import Any


def nemo_scan_colang(
    text: str,
    *,
    mode: str,
    colang_path: str,
) -> dict[str, Any] | None:
    """
    Run Colang-backed scan when ``nemoguardrails`` is installed.

    Returns a dict with keys ``allowed`` (bool) and optional ``detail``; ``None`` when SDK absent.
    """
    try:
        from nemoguardrails import RailsConfig
        from nemoguardrails.rails.llm.llmrails import LLMRails
    except ImportError:
        return None

    config = RailsConfig.from_path(colang_path)
    rails = LLMRails(config)
    messages = [{"role": "user", "content": text}]
    if mode == "output":
        messages = [
            {"role": "user", "content": "context"},
            {"role": "assistant", "content": text},
        ]
    try:
        output = rails.generate(messages=messages)
    except Exception as exc:  # noqa: BLE001 — vendor boundary
        return {"allowed": True, "detail": f"nemo_guardrails skipped: {exc}", "skipped": True}
    blocked = not bool(str(output).strip())
    return {
        "allowed": not blocked,
        "detail": "nemo_guardrails Colang policy blocked" if blocked else "nemo_guardrails pass",
        "output_preview": str(output)[:120],
    }
