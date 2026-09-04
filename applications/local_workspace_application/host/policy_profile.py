# © Artur Czarnecki. All rights reserved.

"""Declarative policy profile for local_workspace_application (UE-11G-C1-R4-F-D1)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications.contracts.environment_profile import PolicyRulesProfile
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID

LKW_PRODUCT_POLICY_RULES_PATH = (
    Path(__file__).resolve().parents[1] / "policy" / "rules" / "product.yaml"
)

LKW_RAG_INGEST_ALLOW_RULE_ID = "lkw.product.rag.ingest_document.allow"


def build_local_workspace_policy_rules_profile() -> PolicyRulesProfile:
    """ENFORCE-mode declarative policy for LKW meaningful side-effect authorization."""
    if not LKW_PRODUCT_POLICY_RULES_PATH.is_file():
        raise FileNotFoundError(
            f"LKW product policy rules file missing: {LKW_PRODUCT_POLICY_RULES_PATH}",
        )
    return PolicyRulesProfile(
        rules_path=LKW_PRODUCT_POLICY_RULES_PATH,
        policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
    )


def local_workspace_meaningful_side_effect_tool_ids() -> tuple[str, ...]:
    """Meaningful side-effect tools explicitly authorized for LKW C1 indexing."""
    return (RAG_INGEST_TOOL_ID,)
