#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Generate docs/guides/SYMBOL_INDEX.md — symbol to path map (F5)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guides" / "SYMBOL_INDEX.md"

# High-value harness symbols — prefer grep hit in intergrax/ and agents/
SYMBOLS: tuple[str, ...] = (
    "HarnessKernel",
    "NexusLoop",
    "StepOutcome",
    "AgentStepContext",
    "AgentRunRequest",
    "AgentRunTrace",
    "ToolRuntime",
    "PolicyEngine",
    "RuntimeEvent",
    "UnifiedTaskRunner",
    "HarnessApplication",
    "ApplicationHost",
    "AgentRegistry",
    "CapabilityGraph",
    "CognitiveAgent",
    "DecisionRecord",
    "IntegrationProfile",
    "ContextEngine",
    "RetrievalEngine",
    "StepLLMRouter",
    "EffectiveAgentRunEnvironment",
)

CLASS_DEF = re.compile(r"^class\s+({})\b".format("|".join(re.escape(s) for s in SYMBOLS)))


def find_definition(symbol: str) -> str | None:
    for base in (ROOT / "intergrax", ROOT / "agents"):
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "tests" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if re.search(rf"\bclass\s+{re.escape(symbol)}\b", line):
                        rel = path.relative_to(ROOT).as_posix()
                        return f"`{rel}` L{i}"
                    if re.search(rf"\bdef\s+{re.escape(symbol)}\b", line):
                        rel = path.relative_to(ROOT).as_posix()
                        return f"`{rel}` L{i}"
            except OSError:
                continue
    return None


def main() -> None:
    rows = [
        "# Symbol index (F5 — token-efficient code lookup)",
        "",
        "Use this **before** repo-wide semantic search. Grep the path directly.",
        "",
        "| Symbol | Primary definition |",
        "|--------|-------------------|",
    ]
    for sym in SYMBOLS:
        loc = find_definition(sym) or "_grep repo_"
        rows.append(f"| `{sym}` | {loc} |")

    rows.extend(
        [
            "",
            "## Common paths",
            "",
            "| Area | Path |",
            "|------|------|",
            "| Nexus core | `intergrax/runtime/nexus/` |",
            "| Orchestration | `intergrax/runtime/nexus/orchestration/` |",
            "| Tool runtime | `intergrax/tools/` |",
            "| Agent contracts | `intergrax/runtime/nexus/agent/` |",
            "| Tier-3 hosts | `applications/` |",
            "| Tier-2 agents | `agents/` |",
            "",
            "Regenerate: `uv run python scripts/generate_symbol_index.py`",
            "",
        ]
    )
    OUT.write_text("\n".join(rows), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
