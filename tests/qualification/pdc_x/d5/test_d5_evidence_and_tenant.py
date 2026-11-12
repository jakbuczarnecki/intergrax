# © Artur Czarnecki. All rights reserved.

"""PDC-D5-R013 (run recorded as evidence) and R015 (cross-tenant denial)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.contracts.evaluation_contracts import (
    EvalDigestMismatchError,
    EvalSubjectOutput,
    EvalTenantBoundaryError,
)
from intergrax.eval.eval_comparison import compare_eval_runs
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.architecture.online_evaluation_registry import (
    FileOnlineEvaluationRegistry,
    InMemoryOnlineEvaluationRegistry,
    OnlineEvaluationTenantBoundaryError,
    TenantScopedOnlineEvaluationRegistry,
)
from intergrax.runtime.evidence.eval_evidence_contracts import (
    EvalEvidenceCheckKind,
    EvalEvidenceReport,
    EvalEvidenceStatus,
    build_eval_run_evidence_report,
    read_eval_comparison_evidence,
    read_eval_run_evidence,
    validate_eval_evidence_report,
)
from intergrax.runtime.evidence.eval_evidence_export import write_eval_evidence_report
from tests.qualification.pdc_x.d5._support import ANSWERS, build_run

pytestmark = pytest.mark.unit


def _observation(observation_id: str, run_id: str = "run-1") -> OnlineEvaluationObservation:
    return OnlineEvaluationObservation(
        observation_id=observation_id,
        run_id=run_id,
        agent_id="agent:research",
        mode=OnlineEvaluationMode.SHADOW,
        scenario_id="s-math",
        passed=True,
        score=1.0,
    )


def test_r013_run_and_comparison_are_recorded_as_canonical_evidence(tmp_path: Path) -> None:
    baseline = build_run()
    candidate = build_run()
    comparison = compare_eval_runs(baseline, candidate)
    report = build_eval_run_evidence_report(candidate, comparison)
    validate_eval_evidence_report(report)
    assert report.status is EvalEvidenceStatus.PASSED
    kinds = {item.check_kind for item in report.results}
    assert kinds == {EvalEvidenceCheckKind.EVAL_RUN, EvalEvidenceCheckKind.EVAL_COMPARISON}

    json_path, md_path = write_eval_evidence_report(report, tmp_path / "eval")
    reloaded = EvalEvidenceReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert md_path.is_file()
    (run,) = read_eval_run_evidence(reloaded, tenant_id="tenant-a")
    assert run == candidate
    assert run.result_digest == candidate.result_digest
    (restored,) = read_eval_comparison_evidence(reloaded, tenant_id="tenant-a")
    assert restored == comparison


def test_r013_failed_run_is_recorded_as_failed_evidence() -> None:
    answers = dict(ANSWERS)
    answers["s-math"] = EvalSubjectOutput(output_text="5")
    report = build_eval_run_evidence_report(build_run(answers=answers))
    assert report.status is EvalEvidenceStatus.FAILED


def test_r013_regressed_comparison_is_failed_evidence() -> None:
    answers = dict(ANSWERS)
    answers["s-math"] = EvalSubjectOutput(output_text="5")
    candidate = build_run(answers=answers)
    report = build_eval_run_evidence_report(candidate, compare_eval_runs(build_run(), candidate))
    comparison_result = next(
        item for item in report.results if item.check_kind is EvalEvidenceCheckKind.EVAL_COMPARISON
    )
    assert comparison_result.status is EvalEvidenceStatus.FAILED
    assert comparison_result.metadata["advisory"] == "true"


def test_r013_tampered_evidence_payload_is_rejected() -> None:
    run = build_run()
    report = build_eval_run_evidence_report(run)
    result = report.results[0]
    result.metadata["payload_json"] = result.metadata["payload_json"].replace('"seed":7', '"seed":8')
    with pytest.raises(EvalDigestMismatchError):
        read_eval_run_evidence(report, tenant_id="tenant-a")


def test_r013_existing_eval_evidence_checks_keep_their_basis_rules() -> None:
    report = build_eval_run_evidence_report(build_run())
    report.results[0].basis.clear()
    with pytest.raises(ValueError, match="basis must not be empty"):
        validate_eval_evidence_report(report)


def test_r015_evidence_read_across_tenants_is_denied() -> None:
    report = build_eval_run_evidence_report(build_run(tenant="tenant-a"))
    with pytest.raises(EvalTenantBoundaryError):
        read_eval_run_evidence(report, tenant_id="tenant-b")


def test_r015_cross_tenant_comparison_evidence_is_rejected() -> None:
    a_run = build_run(tenant="tenant-a")
    b_run = build_run(tenant="tenant-b")
    with pytest.raises(EvalTenantBoundaryError):
        build_eval_run_evidence_report(a_run, compare_eval_runs(b_run, b_run))


@pytest.mark.parametrize("kind", ["memory", "file"])
def test_r015_cross_tenant_observation_denied(kind: str, tmp_path: Path) -> None:
    registry = (
        InMemoryOnlineEvaluationRegistry() if kind == "memory" else FileOnlineEvaluationRegistry(tmp_path / "obs.json")
    )
    tenant_a = TenantScopedOnlineEvaluationRegistry(registry, "tenant-a")
    tenant_b = TenantScopedOnlineEvaluationRegistry(registry, "tenant-b")
    tenant_a.append(_observation("obs-a"))
    tenant_b.append(_observation("obs-b"))

    assert [o.observation_id for o in tenant_a.list_observations()] == ["obs-a"]
    assert [o.observation_id for o in tenant_b.list_observations()] == ["obs-b"]
    assert tenant_a.get_observation("obs-a").observation_id == "obs-a"
    with pytest.raises(OnlineEvaluationTenantBoundaryError):
        tenant_a.get_observation("obs-b")
    with pytest.raises(OnlineEvaluationTenantBoundaryError):
        registry.get_observation("obs-a", tenant_id="tenant-b")
    with pytest.raises(KeyError):
        tenant_a.get_observation("nope")


@pytest.mark.parametrize("kind", ["memory", "file"])
def test_r015_untenanted_and_blank_tenant_fail_closed(kind: str, tmp_path: Path) -> None:
    registry = (
        InMemoryOnlineEvaluationRegistry() if kind == "memory" else FileOnlineEvaluationRegistry(tmp_path / "obs.json")
    )
    registry.append(_observation("legacy"))  # legacy producer without a tenant
    scoped = TenantScopedOnlineEvaluationRegistry(registry, "tenant-a")
    assert scoped.list_observations() == []
    with pytest.raises(OnlineEvaluationTenantBoundaryError):
        registry.get_observation("legacy", tenant_id="tenant-a")
    with pytest.raises(ValueError, match="tenant_id"):
        TenantScopedOnlineEvaluationRegistry(registry, " ")
    with pytest.raises(ValueError, match="tenant_id"):
        registry.list_observations_for_tenant("")
    assert [o.observation_id for o in registry.list_observations()] == ["legacy"]


def test_r015_tenant_survives_the_file_roundtrip_and_clear_is_tenant_scoped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "obs.json"
    FileOnlineEvaluationRegistry(path).append_for_tenant(_observation("o1"), "tenant-a")
    reopened = FileOnlineEvaluationRegistry(path)
    reopened.append_for_tenant(_observation("o2"), "tenant-b")
    assert [o.observation_id for o in reopened.list_observations_for_tenant("tenant-a")] == ["o1"]
    TenantScopedOnlineEvaluationRegistry(reopened, "tenant-a").clear()
    assert [o.observation_id for o in reopened.list_observations_for_tenant("tenant-b")] == ["o2"]
    assert reopened.list_observations_for_tenant("tenant-a") == []


def test_r015_scoped_registry_refuses_foreign_tenant_stamp() -> None:
    from intergrax.runtime.architecture.online_evaluation_registry import (
        TenantOnlineEvaluationObservation,
    )

    registry = InMemoryOnlineEvaluationRegistry()
    foreign = TenantOnlineEvaluationObservation(**_observation("x").model_dump(), tenant_id="tenant-b")
    with pytest.raises(OnlineEvaluationTenantBoundaryError):
        TenantScopedOnlineEvaluationRegistry(registry, "tenant-a").append(foreign)


def test_r015_application_wiring_binds_registry_to_the_tenant(tmp_path: Path) -> None:
    from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="d5.tenant")
    env = env.model_copy(
        update={
            "evaluation_profile": env.evaluation_profile.model_copy(update={"registry_path": tmp_path / "wired.json"})
        }
    )
    wiring = wire_application_evaluation(env, tenant_id="tenant-a")
    assert isinstance(wiring.registry, TenantScopedOnlineEvaluationRegistry)
    assert wiring.tenant_id == "tenant-a"
    assert wire_application_evaluation(env).tenant_id is None
