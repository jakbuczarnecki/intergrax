# © Artur Czarnecki. All rights reserved.
"""Fix implementation plan hub links after phase decomposition."""

from __future__ import annotations

import re
from pathlib import Path

PLAN = Path(__file__).resolve().parents[1] / "docs" / "INTERGRAX_IMPLEMENTATION_PLAN.md"

PHASE_LINKS: dict[str, str] = {
    "phase-v-rem": "plan/phases/core-runtime.md",
    "phase-w-ml": "plan/phases/llm-and-modality.md",
    "phase-w-ops": "plan/phases/platform-quality.md",
    "phase-w-adapt": "plan/phases/evaluation-adaptive-critic.md",
    "phase-h-app": "plan/phases/tier3-dx-aa.md",
    "phase-dx": "plan/phases/tier3-dx-aa.md",
    "phase-aa": "plan/phases/tier3-dx-aa.md",
    "phase-mem": "plan/phases/rag-context-memory.md",
    "phase-mem-depth": "plan/phases/rag-context-memory.md",
    "phase-gov-audit": "plan/phases/governance-security.md",
    "phase-faudit-32": "plan/phases/platform-quality.md",
    "phase-orch": "plan/phases/core-runtime.md",
    "phase-flow": "plan/phases/core-runtime.md",
    "phase-ts": "plan/phases/tools-skills.md",
    "phase-int": "plan/phases/integrations.md",
    "phase-rag": "plan/phases/rag-context-memory.md",
    "phase-ctx": "plan/phases/rag-context-memory.md",
    "phase-leg": "plan/phases/tools-skills.md",
    "phase-pe": "plan/phases/registry-capability.md",
    "phase-clean": "plan/phases/core-runtime.md",
    "phase-as": "plan/phases/core-runtime.md",
    "phase-reg": "plan/phases/registry-capability.md",
    "phase-cg": "plan/phases/registry-capability.md",
    "phase-obs": "plan/phases/observability-reliability.md",
    "phase-obs-bus": "plan/phases/observability-reliability.md",
    "phase-rel": "plan/phases/observability-reliability.md",
    "phase-sec": "plan/phases/governance-security.md",
    "phase-cost": "plan/phases/governance-security.md",
    "phase-eval": "plan/phases/evaluation-adaptive-critic.md",
    "phase-crit-v": "plan/phases/evaluation-adaptive-critic.md",
    "phase-m-llm-r": "plan/phases/llm-and-modality.md",
}

APPENDIX_LINKS = {chr(i): f"plan/appendices/appendix-{chr(i).lower()}.md" for i in range(ord("a"), ord("n") + 1)}


def main() -> None:
    text = PLAN.read_text(encoding="utf-8")

    for slug, path in PHASE_LINKS.items():
        text = re.sub(
            rf"\(#({slug}[^)]*)\)",
            rf"]({path})",
            text,
            flags=re.I,
        )

    for letter, path in APPENDIX_LINKS.items():
        text = re.sub(
            rf"\[Appendix {letter.upper()}\]\([^)]*\)",
            rf"[Appendix {letter.upper()}]({path})",
            text,
        )
        text = re.sub(
            rf"\*\*Appendix {letter.upper()}\*\* \(below\)",
            rf"**[Appendix {letter.upper()}]({path})**",
            text,
        )

    text = text.replace(
        "canon [§55](intergrax_runtime_architecture.md#55-critic--verification-layer-cvl--pev-verify-addendum)",
        "canon [`CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md`](CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md)",
    )
    text = text.replace(
        "canon [§54](intergrax_runtime_architecture.md#54-adaptive-harness-intelligence-ahi--l4-runtime-addendum)",
        "canon [`ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md`](ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md)",
    )
    text = text.replace(
        "canon [§5.3](intergrax_runtime_architecture.md#53-harness-ai-alignment-conceptual-model)",
        "canon [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) §5.3",
    )
    text = text.replace("Architecture canon §7.1.1–§7.1.5", "[`INTEGRATIONS.md`](INTEGRATIONS.md) + [`architecture/INTEGRATIONS_ARCHITECTURE.md`](architecture/INTEGRATIONS_ARCHITECTURE.md)")
    text = text.replace("Architecture canon §7.1.6–§7.1.7, §22", "[`TOOLS.md`](TOOLS.md) + [`architecture/TOOLS_RUNTIME.md`](architecture/TOOLS_RUNTIME.md)")
    text = text.replace("Architecture §5.3, §7.1.6–§7.1.8", "[`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) §5.3 · [`TOOLS.md`](TOOLS.md) · [`SKILLS.md`](SKILLS.md)")
    text = text.replace("Architecture canon §7.1.9", "[`MODALITY.md`](MODALITY.md)")
    text = text.replace("Architecture canon §7.4.8–§7.4.10", "[`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md)")
    text = text.replace("canon §42.11", "[`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.11")
    text = text.replace("canon §42.3–§42.15, §42.43", "[`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md)")

    text = text.replace(
        "Do not maintain separate status/readiness/roadmap files. This plan is the **only** live **platform (Harness / Agent OS)** implementation document:",
        "This file is the **implementation plan hub**. Detailed phase registers live under [`plan/phases/`](plan/phases/). Appendices: [`plan/appendices/`](plan/appendices/).",
    )

    PLAN.write_text(text, encoding="utf-8")
    print("Fixed plan hub links")


if __name__ == "__main__":
    main()
