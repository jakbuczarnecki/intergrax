# © Artur Czarnecki. All rights reserved.
"""Migrate implementation plan to flat docs/plan/*.md (1:1 with docs/architecture/)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MONOLITH = DOCS / "INTERGRAX_IMPLEMENTATION_PLAN.md"
PLAN_DIR = DOCS / "plan"
ARCH_DIR = DOCS / "architecture"
GUIDES = DOCS / "guides"
OLD_PHASES = PLAN_DIR / "phases"
OLD_APPENDICES = PLAN_DIR / "appendices"

ARCH_NAMES = [
    "PLATFORM_FOUNDATION",
    "UNIFIED_EXECUTION_RUNTIME",
    "ORCHESTRATION",
    "NEXUS_EXECUTION_FLOW",
    "AGENT_CONTRACTS_AND_ASSEMBLY",
    "INTEGRATIONS",
    "TOOLS",
    "SKILLS",
    "LLM_ADAPTERS",
    "MEMORY",
    "MODALITY",
    "OBSERVABILITY",
    "RELIABILITY_FAILURE_AND_HITL",
    "TIER3_APPLICATION_ENVIRONMENT",
    "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "ADAPTIVE_HARNESS_INTELLIGENCE",
    "CRITIC_VERIFICATION",
]

PHASE_TO_PLAN: dict[str, str | list[str]] = {
    "V-REM": "UNIFIED_EXECUTION_RUNTIME",
    "CLEAN": "UNIFIED_EXECUTION_RUNTIME",
    "GOV-AUDIT": "UNIFIED_EXECUTION_RUNTIME",
    "SEC": "UNIFIED_EXECUTION_RUNTIME",
    "COST": "UNIFIED_EXECUTION_RUNTIME",
    "P4": "UNIFIED_EXECUTION_RUNTIME",
    "G": "UNIFIED_EXECUTION_RUNTIME",
    "J": "UNIFIED_EXECUTION_RUNTIME",
    "ORCH": "ORCHESTRATION",
    "B": "ORCHESTRATION",
    "C": "ORCHESTRATION",
    "FLOW": "NEXUS_EXECUTION_FLOW",
    "AS": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "REG": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "CG": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "PE": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "L": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "INT": "INTEGRATIONS",
    "RAG": "INTEGRATIONS",
    "M": "INTEGRATIONS",
    "M-RAG": "INTEGRATIONS",
    "H": ["INTEGRATIONS", "TIER3_APPLICATION_ENVIRONMENT"],
    "LEG": "TOOLS",
    "O": "TOOLS",
    "TS": ["TOOLS", "SKILLS"],
    "R": "SKILLS",
    "M-LLM": "LLM_ADAPTERS",
    "M-LLM-R": "LLM_ADAPTERS",
    "MEM": "MEMORY",
    "MEM-DEPTH": "MEMORY",
    "CTX": "MEMORY",
    "I": "MEMORY",
    "W-ML": "MODALITY",
    "OBS": "OBSERVABILITY",
    "OBS-BUS": "OBSERVABILITY",
    "D": ["OBSERVABILITY", "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE"],
    "REL": "RELIABILITY_FAILURE_AND_HITL",
    "F": "RELIABILITY_FAILURE_AND_HITL",
    "H-APP": "TIER3_APPLICATION_ENVIRONMENT",
    "N": "TIER3_APPLICATION_ENVIRONMENT",
    "DX": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "AA": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "W-OPS": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "EVAL": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "P-Ext": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "E": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "A": ["PLATFORM_FOUNDATION", "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE"],
    "W-ADAPT": "ADAPTIVE_HARNESS_INTELLIGENCE",
    "CRIT-V": "CRITIC_VERIFICATION",
    "FAUDIT-32": "PLATFORM_FOUNDATION",
    "Q": "PLATFORM_FOUNDATION",
    "Q+": "PLATFORM_FOUNDATION",
    "S": "PLATFORM_FOUNDATION",
    "T": "PLATFORM_FOUNDATION",
    "U": "PLATFORM_FOUNDATION",
    "K": "PLATFORM_FOUNDATION",
    "V": "PLATFORM_FOUNDATION",
}

V_SUB_TO_PLAN = {
    "V-CG": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "V-ALG": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "V-PE": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "V-CE": "MEMORY",
    "V-EVAL": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "V-AM": "PLATFORM_FOUNDATION",
    "V-SEC": "UNIFIED_EXECUTION_RUNTIME",
    "V-COST": "UNIFIED_EXECUTION_RUNTIME",
    "V-MA": "ORCHESTRATION",
    "V-KG": "INTEGRATIONS",
    "V-V6": "PLATFORM_FOUNDATION",
}

SECTION_61_TO_PLAN = {
    "6.1": "PLATFORM_FOUNDATION",
    "6.1a": "PLATFORM_FOUNDATION",
    "6.1z": "PLATFORM_FOUNDATION",
    "6.1u": "PLATFORM_FOUNDATION",
    "6.1s": "PLATFORM_FOUNDATION",
    "6.1ah": "PLATFORM_FOUNDATION",
    "6.1ai": "PLATFORM_FOUNDATION",
    "6.1aa": "MEMORY",
    "6.1am": "MEMORY",
    "6.1f": "MEMORY",
    "6.1aj": "NEXUS_EXECUTION_FLOW",
    "6.1ak": "CRITIC_VERIFICATION",
    "6.1al": "OBSERVABILITY",
    "6.1b": "ORCHESTRATION",
    "6.1c": ["TOOLS", "SKILLS"],
    "6.1d": "INTEGRATIONS",
    "6.1e": "INTEGRATIONS",
    "6.1w": "INTEGRATIONS",
    "6.1x": "INTEGRATIONS",
    "6.1y": "INTEGRATIONS",
    "6.1g": "UNIFIED_EXECUTION_RUNTIME",
    "6.1h": "TOOLS",
    "6.1j": "UNIFIED_EXECUTION_RUNTIME",
    "6.1q": "UNIFIED_EXECUTION_RUNTIME",
    "6.1r": "UNIFIED_EXECUTION_RUNTIME",
    "6.1i": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "6.1k": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "6.1l": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "6.1m": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "6.1n": "OBSERVABILITY",
    "6.1o": "RELIABILITY_FAILURE_AND_HITL",
    "6.1p": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "6.1t": "ADAPTIVE_HARNESS_INTELLIGENCE",
    "6.1v": "LLM_ADAPTERS",
}

SECTION_62_TO_PLAN = {
    "6.2": "PLATFORM_FOUNDATION",
    "6.2v": "UNIFIED_EXECUTION_RUNTIME",
    "6.2bb": "ORCHESTRATION",
    "6.2aj": "NEXUS_EXECUTION_FLOW",
    "6.2bc": ["TOOLS", "SKILLS"],
    "6.2bd": "INTEGRATIONS",
    "6.2ae": "INTEGRATIONS",
    "6.2af": "INTEGRATIONS",
    "6.2ag": "INTEGRATIONS",
    "6.2be": "INTEGRATIONS",
    "6.2bf": "MEMORY",
    "6.2aa": "MEMORY",
    "6.2ab": "MEMORY",
    "6.2bg": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "6.2bh": "UNIFIED_EXECUTION_RUNTIME",
    "6.2bi": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "6.2bj": "AGENT_CONTRACTS_AND_ASSEMBLY",
    "6.2bk": "OBSERVABILITY",
    "6.2bl": "RELIABILITY_FAILURE_AND_HITL",
    "6.2bm": "UNIFIED_EXECUTION_RUNTIME",
    "6.2bn": "UNIFIED_EXECUTION_RUNTIME",
    "6.2bo": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "6.2ac": "ADAPTIVE_HARNESS_INTELLIGENCE",
    "6.2ad": "LLM_ADAPTERS",
    "6.2ak": "CRITIC_VERIFICATION",
    "6.2w": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "6.2x": "TIER3_APPLICATION_ENVIRONMENT",
    "6.2y": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "6.2z": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
}

APPENDIX_TO_PLAN = {
    "a": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "b": "PLATFORM_FOUNDATION",
    "c": "PLATFORM_FOUNDATION",
    "d": "PLATFORM_FOUNDATION",
    "e": "PLATFORM_FOUNDATION",
    "f": "PLATFORM_FOUNDATION",
    "g": "PLATFORM_FOUNDATION",
    "h": "UNIFIED_EXECUTION_RUNTIME",
    "i": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "j": "UNIFIED_EXECUTION_RUNTIME",
    "k": "INTEGRATIONS",
    "l": "MEMORY",
    "m": "PLATFORM_FOUNDATION",
    "n": "NEXUS_EXECUTION_FLOW",
}


def _targets(key: str, mapping: dict) -> list[str]:
    val = mapping.get(key)
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _add(bucket: dict[str, list[str]], name: str, block: str) -> None:
    block = block.strip()
    if not block:
        return
    bucket.setdefault(name, []).append(block)


def split_by_pattern(text: str, pattern: str) -> list[tuple[str, str]]:
    parts = re.split(pattern, text, flags=re.MULTILINE)
    if not parts or parts[0].strip() == "":
        parts = parts[1:]
    result: list[tuple[str, str]] = []
    i = 0
    while i < len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1]
        result.append((header, body))
        i += 2
    if i < len(parts) and parts[i].strip():
        result.append(("", parts[i]))
    return result


def extract_blocks(text: str, header_prefix: str) -> list[str]:
    """Extract blocks starting with ## Phase or ### Phase headers."""
    if header_prefix == "### Phase":
        stop = r"(?=\n### Phase |\n## Phase |\Z)"
    else:
        stop = r"(?=\n## Phase |\n### Phase |\Z)"
    pattern = rf"^{re.escape(header_prefix)} .+?{stop}"
    return [m.group(0).strip() for m in re.finditer(pattern, text, re.MULTILINE | re.DOTALL)]


def git_show(path: str) -> str | None:
    import subprocess

    rel = path.replace("\\", "/")
    if rel.startswith("docs/"):
        git_path = rel
    else:
        git_path = f"docs/{rel}" if not rel.startswith("docs") else rel
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{git_path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def phase_key_from_header(header: str) -> str | None:
    m = re.match(r"^#+\s+Phase\s+([A-Za-z0-9+\-]+)", header.strip())
    if m:
        return m.group(1)
    m = re.match(r"^#+\s+([0-9]+\.[0-9a-z]+)\s", header.strip())
    if m:
        return m.group(1)
    return None


def route_phase(key: str) -> list[str]:
    if key.startswith("V-"):
        return _targets(key, V_SUB_TO_PLAN) or ["PLATFORM_FOUNDATION"]
    return _targets(key, PHASE_TO_PLAN)


def parse_monolith(text: str) -> dict[str, list[str]]:
    bucket: dict[str, list[str]] = {n: [] for n in ARCH_NAMES}

    # §0-§5 + documentation model -> PLATFORM_FOUNDATION
    m = re.search(r"^## Documentation model", text, re.MULTILINE)
    s6 = re.search(r"^## 6\. What to implement next", text, re.MULTILINE)
    if m and s6:
        _add(bucket, "PLATFORM_FOUNDATION", text[m.start() : s6.start()])

    # §6 top + §6.3 product backlog
    sec63 = re.search(r"^### 6\.3 End of plan", text, re.MULTILINE)
    sec64 = re.search(r"^### 6\.4 Historical", text, re.MULTILINE)
    if s6 and sec63:
        _add(bucket, "PLATFORM_FOUNDATION", text[s6.start() : sec63.start()])
    if sec63:
        end = sec64.start() if sec64 else len(text)
        _add(bucket, "PLATFORM_FOUNDATION", text[sec63.start() : end])
    if sec64:
        _add(bucket, "PLATFORM_FOUNDATION", text[sec64.start() :])

    # ### 6.1xx / 6.2xx subsections
    for header, body in split_by_pattern(text, r"(?=^### 6\.[0-9a-z]+ )"):
        key_m = re.match(r"^### (6\.[0-9a-z]+)", header.strip())
        if not key_m:
            continue
        key = key_m.group(1)
        block = header + body
        targets = _targets(key, SECTION_61_TO_PLAN) or _targets(key, SECTION_62_TO_PLAN)
        if not targets:
            continue
        for t in targets:
            _add(bucket, t, block)

    # M.6 registers (### M.6 P4 etc.)
    for header, body in split_by_pattern(text, r"(?=^### M\.6 )"):
        if "M.6" in header:
            _add(bucket, "INTEGRATIONS", header + body)

    # Orphan blocks between §5 and §6 (phase tails without ## header)
    m5 = re.search(r"^## 5\. Definition of Done", text, re.MULTILINE)
    if m5 and s6:
        orphan = text[m5.end() : s6.start()]
        if orphan.strip():
            for kw, plan in [
                ("GOV-DOC", "UNIFIED_EXECUTION_RUNTIME"),
                ("FAUDIT-32", "PLATFORM_FOUNDATION"),
                ("CRIT-V", "CRITIC_VERIFICATION"),
                ("FLOW-GAP", "NEXUS_EXECUTION_FLOW"),
                ("Appendix N", "NEXUS_EXECUTION_FLOW"),
            ]:
                if kw in orphan:
                    _add(bucket, plan, orphan)

    return bucket


def parse_phase_text(text: str, bucket: dict[str, list[str]]) -> None:
    for block in extract_blocks(text, "## Phase"):
        key = phase_key_from_header(block.split("\n", 1)[0])
        if not key:
            continue
        for t in route_phase(key):
            _add(bucket, t, block)
    for block in extract_blocks(text, "### Phase"):
        key = phase_key_from_header(block.split("\n", 1)[0])
        if not key:
            continue
        for t in route_phase(key):
            _add(bucket, t, block)


def parse_phase_file(path: Path, bucket: dict[str, list[str]]) -> None:
    text = path.read_text(encoding="utf-8")
    parse_phase_text(text, bucket)


def parse_historical_v_subsections(text: str, bucket: dict[str, list[str]]) -> None:
    for header, body in split_by_pattern(text, r"(?=^#### V-[A-Z0-9]+ )"):
        key_m = re.match(r"^#### (V-[A-Z0-9]+)", header.strip())
        if not key_m:
            continue
        key = key_m.group(1)
        for t in _targets(key, V_SUB_TO_PLAN):
            _add(bucket, t, header + body)


def parse_appendix(path: Path, bucket: dict[str, list[str]]) -> None:
    letter = path.stem.replace("appendix-", "")
    plan = APPENDIX_TO_PLAN.get(letter)
    if not plan:
        return
    content = path.read_text(encoding="utf-8")
    content = re.sub(
        r"\*\*Hub:\*\*.*\n",
        "",
        content,
        count=1,
    )
    _add(bucket, plan, f"## Appendix {letter.upper()}\n\n{content}")


def plan_header(name: str) -> str:
    title = name.replace("_", " ").title()
    return f"""# {title} — Implementation Plan

**Architecture (1:1):** [`architecture/{name}.md`](../architecture/{name}.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

"""


def load_phase_sources() -> list[str]:
    texts: list[str] = []
    if OLD_PHASES.exists():
        for p in OLD_PHASES.glob("*.md"):
            texts.append(p.read_text(encoding="utf-8"))
    else:
        for name in [
            "core-runtime.md",
            "governance-security.md",
            "integrations.md",
            "tools-skills.md",
            "llm-and-modality.md",
            "rag-context-memory.md",
            "observability-reliability.md",
            "registry-capability.md",
            "evaluation-adaptive-critic.md",
            "tier3-dx-aa.md",
            "platform-quality.md",
            "historical-phases.md",
        ]:
            t = git_show(f"plan/phases/{name}")
            if t:
                texts.append(t)
    return texts


def load_appendix_sources() -> list[Path]:
    if OLD_APPENDICES.exists():
        return sorted(OLD_APPENDICES.glob("appendix-*.md"))
    return []


def write_plan_files(bucket: dict[str, list[str]]) -> None:
    for text in load_phase_sources():
        parse_phase_text(text, bucket)
        parse_historical_v_subsections(text, bucket)

    for p in load_appendix_sources():
        parse_appendix(p, bucket)
    if not OLD_APPENDICES.exists():
        for letter in "abcdefghijklmn":
            t = git_show(f"plan/appendices/appendix-{letter}.md")
            if t:
                plan = APPENDIX_TO_PLAN.get(letter)
                if plan:
                    t = re.sub(r"\*\*Hub:\*\*.*\n", "", t, count=1)
                    _add(bucket, plan, f"## Appendix {letter.upper()}\n\n{t}")

    mono_text = None
    if MONOLITH.exists():
        mono_text = MONOLITH.read_text(encoding="utf-8")
    else:
        mono_text = git_show("INTERGRAX_IMPLEMENTATION_PLAN.md")
    if mono_text:
        mono_bucket = parse_monolith(mono_text)
        for name in ARCH_NAMES:
            bucket[name] = mono_bucket.get(name, []) + bucket.get(name, [])

    if OLD_PHASES.exists():
        shutil.rmtree(OLD_PHASES)
    if OLD_APPENDICES.exists():
        shutil.rmtree(OLD_APPENDICES)
    readme = PLAN_DIR / "README.md"
    if readme.exists():
        readme.unlink()
    arch_readme = ARCH_DIR / "README.md"
    if arch_readme.exists():
        arch_readme.unlink()

    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARCH_NAMES:
        blocks = bucket.get(name, [])
        body = "\n\n---\n\n".join(blocks) if blocks else "_No implementation tasks registered yet._\n"
        out = PLAN_DIR / f"{name}.md"
        out.write_text(plan_header(name) + body + "\n", encoding="utf-8")


def move_guides() -> None:
    GUIDES.mkdir(parents=True, exist_ok=True)
    moves = [
        "IDEAL_HARNESS_AI_ARCHITECTURE.md",
        "INTEGRAX_HARNESS_AUDIT_MAP.md",
        "INTERGRAX_DEVELOPMENT_STRATEGY.md",
    ]
    for fname in moves:
        src = DOCS / fname
        dst = GUIDES / fname
        if src.exists():
            shutil.move(str(src), str(dst))


def update_hub() -> None:
    hub = DOCS / "intergrax_runtime_architecture.md"
    lines = [
        "# Intergrax Runtime Architecture",
        "",
        "**Hub only** — domain architecture and implementation are paired 1:1 under `architecture/` and `plan/`.",
        "**Target:** [`guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)",
        "**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)",
        "**Audit:** [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md)",
        "**Authoring:** [`guides/`](guides/)",
        "",
        "---",
        "",
        "## Four tiers",
        "",
        "```text",
        "Tier-0  intergrax/          integrations · tools · skills · LLM · RAG · memory",
        "Tier-1  intergrax/runtime/    Nexus · AgentEngine · UAEP · policy",
        "Tier-2  agents/             domain capabilities",
        "Tier-3  applications/       deployable hosts",
        "```",
        "",
        "Stack: Integration → Tool → Skill → Agent",
        "Execution: [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md)",
        "",
        "---",
        "",
        "## Domain documents (architecture ↔ implementation 1:1)",
        "",
        "| Architecture | Implementation plan |",
        "|--------------|---------------------|",
    ]
    for name in ARCH_NAMES:
        title = name.replace("_", " ").title()
        lines.append(
            f"| [`architecture/{name}.md`](architecture/{name}.md) | [`plan/{name}.md`](plan/{name}.md) |"
        )
    lines += [
        "",
        "---",
        "",
        "## Reading order",
        "",
        "1. This hub → [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) + [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md)",
        "2. [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) + matching plan",
        "3. [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) + matching plan",
        "4. Your domain pair from the table above",
        "5. [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when building agents",
        "",
        "**Per-iteration rule:** pick one domain — read only its architecture + plan pair; do not load unrelated domains.",
        "",
        "Platform docs do not replace `agents/*/ARCHITECTURE.md` or `applications/*/ARCHITECTURE.md`.",
        "",
    ]
    hub.write_text("\n".join(lines), encoding="utf-8")


REPLACEMENTS = [
    (r"docs/INTERGRAX_IMPLEMENTATION_PLAN\.md", "docs/intergrax_runtime_architecture.md"),
    (r"INTERGRAX_IMPLEMENTATION_PLAN\.md", "intergrax_runtime_architecture.md"),
    (r"\]\(INTERGRAX_IMPLEMENTATION_PLAN\.md", "](intergrax_runtime_architecture.md"),
    (r"docs/IDEAL_HARNESS_AI_ARCHITECTURE\.md", "docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md"),
    (r"\]\(IDEAL_HARNESS_AI_ARCHITECTURE\.md\)", "](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)"),
    (r"docs/INTEGRAX_HARNESS_AUDIT_MAP\.md", "docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md"),
    (r"\]\(INTEGRAX_HARNESS_AUDIT_MAP\.md\)", "](guides/INTEGRAX_HARNESS_AUDIT_MAP.md)"),
    (r"docs/INTERGRAX_DEVELOPMENT_STRATEGY\.md", "docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md"),
    (r"\]\(INTERGRAX_DEVELOPMENT_STRATEGY\.md\)", "](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)"),
    (r"architecture/README\.md", "intergrax_runtime_architecture.md"),
    (r"plan/phases/core-runtime\.md", "plan/ORCHESTRATION.md"),
    (r"plan/phases/governance-security\.md", "plan/UNIFIED_EXECUTION_RUNTIME.md"),
    (r"plan/phases/integrations\.md", "plan/INTEGRATIONS.md"),
    (r"plan/phases/tools-skills\.md", "plan/TOOLS.md"),
    (r"plan/phases/llm-and-modality\.md", "plan/LLM_ADAPTERS.md"),
    (r"plan/phases/rag-context-memory\.md", "plan/MEMORY.md"),
    (r"plan/phases/observability-reliability\.md", "plan/OBSERVABILITY.md"),
    (r"plan/phases/registry-capability\.md", "plan/AGENT_CONTRACTS_AND_ASSEMBLY.md"),
    (r"plan/phases/evaluation-adaptive-critic\.md", "plan/CRITIC_VERIFICATION.md"),
    (r"plan/phases/tier3-dx-aa\.md", "plan/TIER3_APPLICATION_ENVIRONMENT.md"),
    (r"plan/phases/platform-quality\.md", "plan/PLATFORM_FOUNDATION.md"),
    (r"plan/phases/historical-phases\.md", "plan/PLATFORM_FOUNDATION.md"),
    (r"plan/phases/misc-phases\.md", "plan/LLM_ADAPTERS.md"),
    (r"plan/appendices/appendix-[a-z]\.md", "plan/PLATFORM_FOUNDATION.md"),
    (r"plan/phases/", "plan/"),
    (r"plan/appendices/", "plan/"),
    (r"\]\(\.\./IDEAL_HARNESS_AI_ARCHITECTURE\.md\)", "](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)"),
    (r"\]\(\.\./INTERGRAX_DEVELOPMENT_STRATEGY\.md\)", "](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)"),
    (r"\]\(\.\./INTEGRAX_HARNESS_AUDIT_MAP\.md\)", "](../guides/INTEGRAX_HARNESS_AUDIT_MAP.md)"),
    (r"\]\(INTEGRATIONS\.md\)", "](architecture/INTEGRATIONS.md)"),
]


def update_references() -> None:
    skip = {".git", ".venv", "__pycache__", "node_modules"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(p in path.parts for p in skip):
            continue
        if path.suffix not in {".md", ".mdc", ".txt", ".py", ".yml", ".yaml"}:
            continue
        if path.name == "migrate_plan_1to1.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        orig = text
        for pattern, repl in REPLACEMENTS:
            text = re.sub(pattern, repl, text)
        if text != orig:
            path.write_text(text, encoding="utf-8")


def fix_architecture_internal_links() -> None:
    for path in ARCH_DIR.glob("*.md"):
        if path.name == "README.md":
            path.unlink(missing_ok=True)
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "](../IDEAL_HARNESS_AI_ARCHITECTURE.md)",
            "](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)",
        )
        text = text.replace(
            "](INTERGRAX_IMPLEMENTATION_PLAN.md",
            f"](plan/{path.stem}.md",
        )
        path.write_text(text, encoding="utf-8")


def main() -> None:
    bucket: dict[str, list[str]] = {n: [] for n in ARCH_NAMES}
    write_plan_files(bucket)
    if MONOLITH.exists():
        MONOLITH.unlink()
    if not (GUIDES / "IDEAL_HARNESS_AI_ARCHITECTURE.md").exists():
        move_guides()
    update_hub()
    fix_architecture_internal_links()
    update_references()
    print("migrate_plan_1to1: done")


if __name__ == "__main__":
    main()
