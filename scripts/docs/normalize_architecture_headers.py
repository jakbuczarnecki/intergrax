# © Artur Czarnecki. All rights reserved.
"""Normalize architecture/*.md headers to 1:1 plan link pattern."""

from __future__ import annotations

import re
from pathlib import Path

ARCH = Path(__file__).resolve().parents[2] / "docs" / "architecture"

AUDIT_LAYERS: dict[str, str] = {
    "PLATFORM_FOUNDATION": "1–2, 32",
    "UNIFIED_EXECUTION_RUNTIME": "4–5, 8, 23–24",
    "ORCHESTRATION": "3, 7, 9",
    "NEXUS_EXECUTION_FLOW": "7–10",
    "AGENT_CONTRACTS_AND_ASSEMBLY": "17–20, 31",
    "INTEGRATIONS": "13–14",
    "TOOLS": "11",
    "SKILLS": "12",
    "LLM_ADAPTERS": "6",
    "MEMORY": "15–16",
    "MODALITY": "29",
    "OBSERVABILITY": "21, 30",
    "RELIABILITY_FAILURE_AND_HITL": "22",
    "TIER3_APPLICATION_ENVIRONMENT": "28",
    "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE": "25–27, 30",
    "ADAPTIVE_HARNESS_INTELLIGENCE": "L4 AHI",
    "CRITIC_VERIFICATION": "25 (verify depth)",
}


def header_for(name: str, title: str | None = None) -> str:
    display = title or name.replace("_", " ").title()
    layers = AUDIT_LAYERS.get(name, "")
    layer_line = f"**Audit layers:** {layers}  \n" if layers else ""
    return f"""# {display}

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/{name}.md`](../plan/{name}.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
{layer_line}---
"""


def main() -> None:
    for path in sorted(ARCH.glob("*.md")):
        name = path.stem
        text = path.read_text(encoding="utf-8")
        # Extract title from first # line
        m = re.match(r"^# (.+?)\n", text)
        title = m.group(1) if m else None
        # Body starts after first --- following header block
        parts = re.split(r"\n---\n", text, maxsplit=2)
        if len(parts) >= 2:
            body = parts[-1] if len(parts) == 2 else parts[2]
            # If only 2 parts, body is everything after first ---
            if len(parts) == 2:
                body = parts[1]
            else:
                body = parts[2]
        else:
            body = re.sub(r"^#.*?\n\n?", "", text, count=1, flags=re.DOTALL)

        # Strip duplicate leading --- from body
        body = body.lstrip("\n")
        new_text = header_for(name, title) + "\n" + body
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"normalized {path.name}")


if __name__ == "__main__":
    main()
