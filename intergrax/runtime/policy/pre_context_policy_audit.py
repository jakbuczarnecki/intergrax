# © Artur Czarnecki. All rights reserved.

"""Pre-context policy hook wiring audit helpers (IDEAL-5.1)."""

from __future__ import annotations

from pathlib import Path

_REQUIRED_HOOK_MARKERS: tuple[str, ...] = (
    "HookPoint.BEFORE_CONTEXT_BUILD",
    "RuntimeEventType.POLICY_DECISION",
)


def audit_pre_context_policy_wiring(source_root: Path) -> list[str]:
    """Return missing hook marker ids when pre-context policy wiring is incomplete.

    The UAEP executor lives under ``intergrax/runtime/nexus/uaep`` since 42cbe7e1a (it used to be
    ``intergrax/agents/uaep.py``); both scan roots are required, a missing root fails closed.
    """
    runtime_root = source_root / "intergrax" / "runtime"
    scan_roots = [
        runtime_root,
        runtime_root / "nexus" / "uaep",
    ]
    corpus_parts: dict[Path, str] = {}
    for root in scan_roots:
        if root.is_file():
            corpus_parts[root] = root.read_text(encoding="utf-8")
            continue
        if not root.is_dir():
            return [f"missing scan root: {root}"]
        for path in root.rglob("*.py"):
            corpus_parts[path] = path.read_text(encoding="utf-8")
    corpus = "\n".join(corpus_parts[path] for path in sorted(corpus_parts))
    return [marker for marker in _REQUIRED_HOOK_MARKERS if marker not in corpus]
