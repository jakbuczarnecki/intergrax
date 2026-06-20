#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Insert Cursor read-scope blocks into large architecture domain docs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "docs" / "architecture"
MARKER = "## Cursor read scope (token budget)"
MIN_LINES = 800

SCOPES: dict[str, str] = {
    "PLATFORM_FOUNDATION": (
        "Default: §1–§5 + §5.2 reuse + §5.3 terminology. "
        "Skip historical audit registers unless gap ID cites them."
    ),
    "AGENT_CONTRACTS_AND_ASSEMBLY": (
        "Default: §12–§21 (contract, registry, capability, ACP). "
        "Implementation audit: add §37 routing. Skip §40+ production gate history unless cited."
    ),
    "TIER3_APPLICATION_ENVIRONMENT": (
        "Default: §1–§15 host profile + manifest wiring. "
        "Skip per-product deployment appendices unless Tier-3 task."
    ),
    "TOOLS": (
        "Default: ToolRuntime path + plugin model + policy invoke. "
        "Skip full catalog tables — use `intergrax/tools/` registry grep."
    ),
    "UNIFIED_EXECUTION_RUNTIME": (
        "Default: UAEP + PolicyEngine + RuntimeEvent spine. "
        "Skip §42 long narrative unless governance task."
    ),
    "ORCHESTRATION": (
        "Default: intake + NexusLoop + graph executor. "
        "Skip strategy catalog §50+ unless ORCH-STRAT task."
    ),
    "NEXUS_EXECUTION_FLOW": (
        "Default: §1–§20 flow narrative + §23 gap register. "
        "Skip §27+ scenario catalog unless FLOW task."
    ),
    "INTEGRATIONS": (
        "Default: manifest registration + IntegrationProfile. "
        "Skip slug inventory — use catalog grep."
    ),
    "CRITIC_VERIFICATION": (
        "Default: CVL contracts + orchestrator + wiring. Skip historical LC narrative unless cited."
    ),
}


def block(domain: str, scope: str) -> str:
    return f"""
{MARKER}

**Do not read this entire file in one session** ({domain} canon).

- **Implement / audit default:** {scope}
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/{domain}.md`](../plan/{domain}.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/{domain}.md`](../guides/audit_slices/{domain}.md).

---
"""


def main() -> None:
    for path in sorted(ARCH.glob("*.md")):
        domain = path.stem
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < MIN_LINES and domain not in SCOPES:
            continue
        scope = SCOPES.get(domain)
        if not scope:
            scope = "Read TOC sections matching current task only; skip appendices and paydown logs."
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            # replace existing block
            start = text.index(MARKER)
            end = text.find("\n---\n", start)
            if end == -1:
                continue
            text = text[:start] + block(domain, scope).strip() + "\n\n" + text[end + 5 :]
        else:
            # insert after first --- following header metadata
            insert_at = text.find("\n---\n")
            if insert_at == -1:
                continue
            insert_at += len("\n---\n")
            text = text[:insert_at] + block(domain, scope) + text[insert_at:]
        path.write_text(text, encoding="utf-8")
        print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
