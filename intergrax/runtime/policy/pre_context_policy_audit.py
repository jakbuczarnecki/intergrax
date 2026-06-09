# © Artur Czarnecki. All rights reserved.

"""Pre-context policy hook wiring audit helpers (IDEAL-5.1)."""

from __future__ import annotations

from pathlib import Path

_REQUIRED_HOOK_MARKERS: tuple[str, ...] = (
    "HookPoint.BEFORE_CONTEXT_BUILD",
    "RuntimeEventType.POLICY_DECISION",
)


def audit_pre_context_policy_wiring(source_root: Path) -> list[str]:
    """Return missing hook marker ids when pre-context policy wiring is incomplete."""
    scan_roots = [
        source_root / "intergrax" / "runtime",
        source_root / "intergrax" / "agents" / "uaep.py",
    ]
    corpus_parts: list[str] = []
    for root in scan_roots:
        if root.is_file():
            corpus_parts.append(root.read_text(encoding="utf-8"))
            continue
        if not root.is_dir():
            return [f"missing scan root: {root}"]
        corpus_parts.extend(
            path.read_text(encoding="utf-8") for path in root.rglob("*.py")
        )
    corpus = "\n".join(corpus_parts)
    return [marker for marker in _REQUIRED_HOOK_MARKERS if marker not in corpus]
