# © Artur Czarnecki. All rights reserved.

"""GR-10-ADR1 — PRE_MODEL identity boundary and execution intake ADR regression gates."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from intergrax.contracts.execution_intake import CanonicalExecutionIntakeRequest
from intergrax.contracts.runtime_policy_context import PreModelPhase

REPO_ROOT = Path(__file__).resolve().parents[4]
ADR_PATH = (
    REPO_ROOT
    / "docs/project/technical/adr/entries/2026-09-18/ADR-GR-10-001.md"
)
PRE_MODEL_EVAL_PATH = (
    REPO_ROOT / "intergrax/runtime/policy/pre_model_policy_evaluation.py"
)
GOVERNANCE_DIR = REPO_ROOT / "intergrax/runtime/governance"

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _read(rel: Path) -> str:
    return rel.read_text(encoding="utf-8")


def test_gr10_adr1_adr_exists_and_accepted() -> None:
    assert ADR_PATH.is_file()
    adr = _read(ADR_PATH)
    assert "**Status** | Accepted" in adr or "| **Status** | Accepted |" in adr
    assert "REVISION_REQUIRED" in adr
    assert "KEEP_REQUIRED" in adr
    assert "ACTIVE_EXECUTION_GOVERNANCE_IDENTITY_ACCEPTED" in adr


def test_gr10_adr1_adr_forbids_empty_and_fake_subjects() -> None:
    adr = _read(ADR_PATH)
    assert "Empty `agent_id`:** **not approved**" in adr or "not approved" in adr
    for sentinel in ("inference", "default", "unknown"):
        assert sentinel in adr


def test_gr10_adr1_intake_requires_governance_scope_fields() -> None:
    names = {f.name for f in fields(CanonicalExecutionIntakeRequest)}
    for required in ("tenant_id", "workspace_id", "principal_id"):
        assert required in names
    source = _read(REPO_ROOT / "intergrax/contracts/execution_intake.py")
    assert "workspace_id must be non-empty" in source
    assert "principal_id must be non-empty" in source


def test_gr10_adr1_evaluate_pre_llm_lacks_principal_until_c1() -> None:
    from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

    sig = inspect.signature(RuntimePolicyEngine.evaluate_pre_llm)
    assert "principal_id" not in sig.parameters, (
        "once GR-10-R2-C1 lands, update this gate to require principal_id"
    )


def test_gr10_adr1_pre_model_eval_does_not_derive_principal_from_evidence() -> None:
    text = _read(PRE_MODEL_EVAL_PATH)
    assert "peek_active_execution_evidence_context" in text
    assert "principal_id" not in text.split("peek_active_execution_evidence_context")[1].split(
        "def ",
        1,
    )[0]


def test_gr10_adr1_no_global_governance_identity_getter() -> None:
    for path in GOVERNANCE_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "get_current_governance_identity" not in text


def test_gr10_adr1_phase_none_valid_per_adr() -> None:
    adr = _read(ADR_PATH)
    assert "phase=None" in adr
    assert PreModelPhase.NEXUS_PLANNING.value == "nexus_planning"
    assert PreModelPhase.AGENT_STEP.value == "agent_step"


def test_gr10_adr1_known_inference_empty_agent_id_pending_r2_r1() -> None:
    """Documents GR-10-R2-R1 remediation target; must be removed after contract+wiring fix."""
    text = _read(PRE_MODEL_EVAL_PATH)
    assert 'agent_id=""' in text or "agent_id=''" in text


@pytest.mark.xfail(
    reason="GR-10-R2-R1: INFERENCE PRE_MODEL must not pass empty agent_id after C1",
    strict=True,
)
def test_gr10_adr1_gate_no_empty_agent_id_on_inference_path() -> None:
    text = _read(PRE_MODEL_EVAL_PATH)
    assert 'agent_id=""' not in text and "agent_id=''" not in text
