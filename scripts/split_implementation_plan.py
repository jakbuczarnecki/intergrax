# © Artur Czarnecki. All rights reserved.
"""Split INTERGRAX_IMPLEMENTATION_PLAN monolith into hub + phase/appendix files."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "INTERGRAX_IMPLEMENTATION_PLAN.md"
PLAN = REPO / "docs" / "plan"
BACKUP = REPO / "docs" / "_archive_INTERGRAX_IMPLEMENTATION_PLAN_monolith.md"

PHASE_H2 = re.compile(r"^## Phase [A-Z0-9][^\n]*$", re.M)
APPENDIX = re.compile(r"^## Appendix [A-Z][^\n]*$", re.M)

PHASE_FILE_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Phase V-REM|Phase ORCH|Phase FLOW|Phase CLEAN|Phase AS|Phase LEG"), "core-runtime"),
    (re.compile(r"Phase INT|M\.6"), "integrations"),
    (re.compile(r"Phase TS|Phase O\b|P-Ext|Plugin"), "tools-skills"),
    (re.compile(r"Phase W-ML|Phase M-LLM|M-LLM-R"), "llm-and-modality"),
    (re.compile(r"Phase RAG|Phase CTX|Phase MEM"), "rag-context-memory"),
    (re.compile(r"Phase GOV|Phase SEC|Phase COST"), "governance-security"),
    (re.compile(r"Phase OBS|Phase REL"), "observability-reliability"),
    (re.compile(r"Phase REG|Phase CG|Phase PE"), "registry-capability"),
    (re.compile(r"Phase EVAL|Phase W-ADAPT|Phase CRIT"), "evaluation-adaptive-critic"),
    (re.compile(r"Phase H-APP|Phase DX|Phase AA|Phase N\b|Phase S\b"), "tier3-dx-aa"),
    (re.compile(r"Phase W-OPS|Phase FAUDIT|Phase Q|Phase U|Phase T|Phase L\b|Phase V\b"), "platform-quality"),
]


def classify_phase(heading: str) -> str:
    for pattern, fname in PHASE_FILE_MAP:
        if pattern.search(heading):
            return fname
    return "misc-phases"


def extract_blocks(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    matches = list(pattern.finditer(text))
    blocks: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((m.group(0), text[start:end].rstrip() + "\n"))
    return blocks


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    if not BACKUP.exists():
        BACKUP.write_text(text, encoding="utf-8")

    phases_dir = PLAN / "phases"
    appendices_dir = PLAN / "appendices"
    phases_dir.mkdir(parents=True, exist_ok=True)
    appendices_dir.mkdir(parents=True, exist_ok=True)

    # Historical §3 block (### Phase A … before first ## Phase)
    sec3 = re.search(r"^## 3\. Implementation Phases", text, re.M)
    first_h2_phase = PHASE_H2.search(text)
    if sec3 and first_h2_phase and first_h2_phase.start() > sec3.start():
        historical = text[sec3.start() : first_h2_phase.start()].rstrip() + "\n"
        (phases_dir / "historical-phases.md").write_text(
            "# Implementation Phases — Historical (A–V)\n\n"
            "**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)\n\n"
            "---\n\n"
            + historical,
            encoding="utf-8",
        )

    phase_buckets: dict[str, list[str]] = {}
    for heading, chunk in extract_blocks(text, PHASE_H2):
        fname = classify_phase(heading)
        phase_buckets.setdefault(fname, []).append(chunk)

    for fname, chunks in phase_buckets.items():
        (phases_dir / f"{fname}.md").write_text(
            f"# Implementation Phases — {fname.replace('-', ' ').title()}\n\n"
            f"**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)\n\n"
            f"---\n\n"
            + "\n".join(chunks)
            + "\n",
            encoding="utf-8",
        )

    for heading, chunk in extract_blocks(text, APPENDIX):
        letter = re.search(r"Appendix ([A-Z])", heading)
        if letter:
            (appendices_dir / f"appendix-{letter.group(1).lower()}.md").write_text(
                f"**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)\n\n"
                f"---\n\n"
                + chunk,
                encoding="utf-8",
            )

    # Build hub: everything except ## Phase blocks and appendices
    parts: list[str] = []
    markers = sorted(
        [(m.start(), m.end(), "skip") for m in PHASE_H2.finditer(text)]
        + [(m.start(), m.end(), "skip") for m in APPENDIX.finditer(text)],
        key=lambda x: x[0],
    )
    pos = 0
    for start, end, _ in markers:
        if start > pos:
            parts.append(text[pos:start])
        pos = end
    if pos < len(text):
        parts.append(text[pos:])
    hub = "".join(parts).rstrip() + "\n"

    # Replace bloated §3 with pointer
    hub = re.sub(
        r"## 3\. Implementation Phases.*?(?=## 4\. Priority Order)",
        """## 3. Implementation Phases

Historical phase registers (A–V) and closeout phases are decomposed under [`plan/phases/`](plan/phases/).

| Domain | File |
|--------|------|
| Historical A–V | [`plan/phases/historical-phases.md`](plan/phases/historical-phases.md) |
| Core runtime | [`plan/phases/core-runtime.md`](plan/phases/core-runtime.md) |
| Integrations | [`plan/phases/integrations.md`](plan/phases/integrations.md) |
| Tools & skills | [`plan/phases/tools-skills.md`](plan/phases/tools-skills.md) |
| LLM & modality | [`plan/phases/llm-and-modality.md`](plan/phases/llm-and-modality.md) |
| RAG, context, memory | [`plan/phases/rag-context-memory.md`](plan/phases/rag-context-memory.md) |
| Governance & security | [`plan/phases/governance-security.md`](plan/phases/governance-security.md) |
| Observability & reliability | [`plan/phases/observability-reliability.md`](plan/phases/observability-reliability.md) |
| Registry & capability graph | [`plan/phases/registry-capability.md`](plan/phases/registry-capability.md) |
| Evaluation, AHI, critic | [`plan/phases/evaluation-adaptive-critic.md`](plan/phases/evaluation-adaptive-critic.md) |
| Tier-3, DX, conformance | [`plan/phases/tier3-dx-aa.md`](plan/phases/tier3-dx-aa.md) |
| Platform quality | [`plan/phases/platform-quality.md`](plan/phases/platform-quality.md) |
| Other | [`plan/phases/misc-phases.md`](plan/phases/misc-phases.md) |

Appendices: [`plan/appendices/`](plan/appendices/) · Full backup: [`_archive_INTERGRAX_IMPLEMENTATION_PLAN_monolith.md`](_archive_INTERGRAX_IMPLEMENTATION_PLAN_monolith.md)

""",
        hub,
        count=1,
        flags=re.S,
    )

    # Update documentation model table references
    hub = hub.replace(
        "Full architecture specification | `intergrax_runtime_architecture.md`",
        "Full architecture specification | [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) (hub) · [`architecture/`](architecture/)",
    )

    SRC.write_text(hub, encoding="utf-8")
    print(f"Hub lines: {len(hub.splitlines())}")
    print(f"Phase files: {len(list(phases_dir.glob('*.md')))}")
    print(f"Appendix files: {len(list(appendices_dir.glob('*.md')))}")


if __name__ == "__main__":
    main()
