# © Artur Czarnecki. All rights reserved.

"""In-memory adversarial probes (TENANT-X-P0 §14–15)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.tenant_x._classifiers import classify_candidate
from tests.qualification.tenant_x._types import (
    AdversarialProbeVerdict,
    TenantBlockerClass,
    TenantDiscoveryCandidate,
    TenantFailClosedDisposition,
    TenantIdentitySource,
    TenantPersistenceRole,
    TenantPropagationRole,
)


class SyntheticProbeId(StrEnum):
    A_CROSS_TENANT_STATE_KEY = "A"
    B_CHILD_WIDENS_TENANT = "B"
    C_MISSING_TENANT_DEFAULT = "C"
    D_CROSS_TENANT_PROVIDER_CONFIG = "D"
    E_CROSS_TENANT_CREDENTIAL = "E"
    F_CROSS_TENANT_EVIDENCE = "F"
    G_CROSS_TENANT_CHECKPOINT_RECOVERY = "G"
    H_PLUGIN_REWRITES_TENANT = "H"
    I_LEGAL_CROSS_TENANT_CONTRACT = "I"
    J_GLOBAL_HEALTH_METRIC = "J"
    K_UNKNOWN_TENANT_FIELD = "K"
    L_FUNCTION_PARAM_ONLY = "L"
    M_SILENT_TENANT_DEFAULT = "M"
    N_HARMLESS_GLOBAL_METADATA = "N"
    O_CREDENTIALS_PATH_ONLY = "O"
    P_STORAGE_IGNORES_TENANT = "P"
    Q_ACTUAL_TENANT_PREDICATE = "Q"


@dataclass(frozen=True, slots=True)
class SyntheticAdversarialScenario:
    probe_id: SyntheticProbeId
    description: str
    tenant_a: str
    tenant_b: str
    expected_verdict: AdversarialProbeVerdict


@dataclass(frozen=True, slots=True)
class ClassifierProbeScenario:
    probe_id: SyntheticProbeId
    candidate: TenantDiscoveryCandidate


SYNTHETIC_ADVERSARIAL_SCENARIOS: tuple[SyntheticAdversarialScenario, ...] = (
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.A_CROSS_TENANT_STATE_KEY,
        description="tenant A request → tenant B state key",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.B_CHILD_WIDENS_TENANT,
        description="tenant A authority → child tenant B",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.C_MISSING_TENANT_DEFAULT,
        description='missing tenant → "default"',
        tenant_a="",
        tenant_b="default",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.D_CROSS_TENANT_PROVIDER_CONFIG,
        description="tenant A config → provider binding tenant B",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.E_CROSS_TENANT_CREDENTIAL,
        description="tenant A credential ref → tenant B provider",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.F_CROSS_TENANT_EVIDENCE,
        description="tenant A evidence → tenant B assessment",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.G_CROSS_TENANT_CHECKPOINT_RECOVERY,
        description="tenant A checkpoint → recovery tenant B",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.H_PLUGIN_REWRITES_TENANT,
        description="plugin receives tenant A → returns tenant B",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_FAIL_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.I_LEGAL_CROSS_TENANT_CONTRACT,
        description="explicit legal cross-tenant typed+authorized contract",
        tenant_a="tenant-a",
        tenant_b="tenant-b",
        expected_verdict=AdversarialProbeVerdict.EXPECT_PASS_CLASSIFICATION,
    ),
    SyntheticAdversarialScenario(
        probe_id=SyntheticProbeId.J_GLOBAL_HEALTH_METRIC,
        description="global non-tenant health metric",
        tenant_a="",
        tenant_b="",
        expected_verdict=AdversarialProbeVerdict.EXPECT_PASS_CLASSIFICATION,
    ),
)


def _fixture_candidate(probe_file: str, symbol: str, kind: str) -> TenantDiscoveryCandidate:
    module_path = f"synthetic/probes/{probe_file}.py"
    return TenantDiscoveryCandidate(
        surface_id=f"{module_path}::{symbol}::{kind}",
        module_path=module_path,
        symbol=symbol,
        discovery_kind=kind,
        line_no=1,
    )


CLASSIFIER_PROBE_SCENARIOS: tuple[ClassifierProbeScenario, ...] = (
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.K_UNKNOWN_TENANT_FIELD,
        candidate=_fixture_candidate("k_unknown_tenant_field", "ProbeK", "class.field.tenant"),
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.L_FUNCTION_PARAM_ONLY,
        candidate=_fixture_candidate("l_function_param_only", "fn", "function.param.tenant"),
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.M_SILENT_TENANT_DEFAULT,
        candidate=TenantDiscoveryCandidate(
            surface_id="intergrax/runtime/execution/m_probe_fixture.py::line1::ambient_default_field",
            module_path="intergrax/runtime/execution/m_probe_fixture.py",
            symbol="L1",
            discovery_kind="ambient_default_field",
            line_no=1,
        ),
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.N_HARMLESS_GLOBAL_METADATA,
        candidate=_fixture_candidate("n_harmless_global_metadata", "health_status", "class.field.tenant"),
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.O_CREDENTIALS_PATH_ONLY,
        candidate=TenantDiscoveryCandidate(
            surface_id="synthetic/probes/o_credentials_path_only.py::Broker::class.field.tenant",
            module_path="synthetic/probes/o_credentials_path_only.py",
            symbol="Broker",
            discovery_kind="class.field.tenant",
            line_no=1,
        ),
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.P_STORAGE_IGNORES_TENANT,
        candidate=_fixture_candidate("p_storage_ignores_tenant", "Store", "class.field.tenant"),
    ),
    ClassifierProbeScenario(
        probe_id=SyntheticProbeId.Q_ACTUAL_TENANT_PREDICATE,
        candidate=_fixture_candidate("q_actual_tenant_predicate", "Repo", "class.field.tenant"),
    ),
)


def classify_synthetic_probe(scenario: SyntheticAdversarialScenario) -> TenantPropagationRole:
    probe = scenario.probe_id
    if probe == SyntheticProbeId.I_LEGAL_CROSS_TENANT_CONTRACT:
        return TenantPropagationRole.LEGAL_CROSS_TENANT_BOUNDARY
    if probe == SyntheticProbeId.J_GLOBAL_HEALTH_METRIC:
        return TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE
    if probe == SyntheticProbeId.C_MISSING_TENANT_DEFAULT:
        return TenantPropagationRole.IDENTITY_DERIVATION
    if probe == SyntheticProbeId.H_PLUGIN_REWRITES_TENANT:
        return TenantPropagationRole.TENANT_SCOPED_PROVIDER_SELECTION
    if probe in {
        SyntheticProbeId.A_CROSS_TENANT_STATE_KEY,
        SyntheticProbeId.G_CROSS_TENANT_CHECKPOINT_RECOVERY,
    }:
        return TenantPropagationRole.TENANT_SCOPED_WRITE
    if probe == SyntheticProbeId.B_CHILD_WIDENS_TENANT:
        return TenantPropagationRole.TENANT_SCOPED_CHILD_OPERATION
    if probe == SyntheticProbeId.D_CROSS_TENANT_PROVIDER_CONFIG:
        return TenantPropagationRole.TENANT_SCOPED_CONFIGURATION
    if probe == SyntheticProbeId.E_CROSS_TENANT_CREDENTIAL:
        return TenantPropagationRole.TENANT_SCOPED_CREDENTIAL_REFERENCE
    if probe == SyntheticProbeId.F_CROSS_TENANT_EVIDENCE:
        return TenantPropagationRole.TENANT_SCOPED_EVIDENCE
    return TenantPropagationRole.UNKNOWN


def synthetic_probe_fail_closed(scenario: SyntheticAdversarialScenario) -> TenantFailClosedDisposition:
    role = classify_synthetic_probe(scenario)
    if role == TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE:
        return TenantFailClosedDisposition.NON_TENANT_SURFACE
    if role == TenantPropagationRole.LEGAL_CROSS_TENANT_BOUNDARY:
        return TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH
    if scenario.probe_id == SyntheticProbeId.C_MISSING_TENANT_DEFAULT:
        return TenantFailClosedDisposition.SILENT_DEFAULT_SUBSTITUTION
    return TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH


def evaluate_synthetic_probe(scenario: SyntheticAdversarialScenario) -> bool:
    """Return True when classification matches P0 expected PASS/FAIL bucket."""
    role = classify_synthetic_probe(scenario)
    fail_closed = synthetic_probe_fail_closed(scenario)
    if scenario.expected_verdict == AdversarialProbeVerdict.EXPECT_PASS_CLASSIFICATION:
        return role in {
            TenantPropagationRole.LEGAL_CROSS_TENANT_BOUNDARY,
            TenantPropagationRole.GLOBAL_NON_TENANT_SURFACE,
        }
    if role == TenantPropagationRole.UNKNOWN:
        return False
    if scenario.probe_id == SyntheticProbeId.C_MISSING_TENANT_DEFAULT:
        return fail_closed == TenantFailClosedDisposition.SILENT_DEFAULT_SUBSTITUTION
    return fail_closed in {
        TenantFailClosedDisposition.FAIL_CLOSED_ON_MISMATCH,
        TenantFailClosedDisposition.SILENT_DEFAULT_SUBSTITUTION,
    }


def evaluate_classifier_probe(scenario: ClassifierProbeScenario) -> bool:
    record = classify_candidate(scenario.candidate)
    probe = scenario.probe_id
    if probe == SyntheticProbeId.K_UNKNOWN_TENANT_FIELD:
        return record.propagation_role == TenantPropagationRole.UNKNOWN
    if probe == SyntheticProbeId.L_FUNCTION_PARAM_ONLY:
        return (
            record.identity_source == TenantIdentitySource.EXPLICIT_PARAMETER
            and record.fail_closed_behavior == TenantFailClosedDisposition.UNKNOWN
            and record.persistence_role == TenantPersistenceRole.NOT_APPLICABLE
        )
    if probe == SyntheticProbeId.M_SILENT_TENANT_DEFAULT:
        return record.blocker_class == TenantBlockerClass.IN_SCOPE_BLOCKER
    if probe == SyntheticProbeId.N_HARMLESS_GLOBAL_METADATA:
        return record.blocker_class == TenantBlockerClass.NONE and record.tenant_required is False
    if probe == SyntheticProbeId.O_CREDENTIALS_PATH_ONLY:
        return record.propagation_role == TenantPropagationRole.UNKNOWN
    if probe == SyntheticProbeId.P_STORAGE_IGNORES_TENANT:
        return record.persistence_role != TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE
    if probe == SyntheticProbeId.Q_ACTUAL_TENANT_PREDICATE:
        return record.persistence_role == TenantPersistenceRole.TENANT_IN_KEY_OR_PREDICATE
    return False
